from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from model import MixtureOfExperts
from sklearn.model_selection import KFold

############################################
task = "Task09_Spleen"
# task = "Task06_Lung"
splits = 5
epochs = 500
############################################
print_config()

data_dir = f"./data/msd_datasets/{task}/"

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

kf = KFold(n_splits=splits, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(data_dicts)):

    root_dir = f"./results/{task}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    root_dir = os.path.join(root_dir, f"fold{fold+1}")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    print(f"Fold {fold+1}:")
    print(f"Train size: {len(train_files)}, Val size: {len(val_files)}\n")

    # train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    set_determinism(seed=0)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ]
    )

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=2,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    #     norm=Norm.BATCH,
    # ).to(device)
    model = MixtureOfExperts(num_classes = 2, 
                             unfreeze_epoch = 250, 
                             se_type = "simple", 
                             modality = "CT", 
                             decoder_dropout = 0.0, 
                             score_eval_n_epochs = 2,
                             results_dir=root_dir).to(device)
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs, epoch=epoch, loss_fn=loss_function, labels=labels)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # if (epoch + 1) % val_interval == 0:
        if (epoch + 1) == max_epochs:
            model.eval()
            with torch.no_grad():
                for idx, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (128, 128, 128)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

                plt.figure("train", (12, 6))
    
                plt.subplot(1, 2, 1)
                plt.title("Epoch Average Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("epoch")
                plt.grid()
                plt.plot(x, y)

                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [val_interval * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("epoch")
                plt.grid()
                plt.plot(x, y)
                
                # plt.show()
                plt.savefig(os.path.join(root_dir, "train_stats.png"), dpi=300)  # Save the figure as a PNG file
                plt.close()

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    print(f"train completed, last_metric: {metric:.4f} " f"at epoch: {epoch+1}")

    with open(os.path.join(root_dir, "results.txt"), "w") as f:
        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}", file=f)
        print(f"train completed, last_metric: {metric:.4f} " f"at epoch: {epoch+1}", file=f)

    