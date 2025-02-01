import torch
import torch.nn as nn
# from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer, ResidualSELayer
from monai.networks.layers.factories import Act, Pool

class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_prob=0.0):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        self.all_modules = nn.Sequential(
            self.conv,
            self.norm,
            self.nonlin,
            self.dropout
        )
    
    def forward(self, x):
        return self.all_modules(x)

class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, in_stride, out_stride, padding, dropout):
        super(StackedConvBlocks, self).__init__()
        self.convs = nn.Sequential(
            ConvDropoutNormReLU(in_channels, mid_channels, kernel_size=kernel_size, stride=in_stride, padding=padding, dropout_prob=dropout),
            ConvDropoutNormReLU(mid_channels, out_channels, kernel_size=kernel_size, stride=out_stride, padding=padding, dropout_prob=dropout)
        )
    
    def forward(self, x):
        return self.convs(x)

class PlainConvEncoder(nn.Module):
    def __init__(self, in_channels=1, dropout=0.0):
        super().__init__()
        self.stages = nn.Sequential(
            StackedConvBlocks(in_channels, 32, 32, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),  
            StackedConvBlocks(32, 64, 64, kernel_size=3, in_stride=2, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(64, 128, 128, kernel_size=3, in_stride=2, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(128, 256, 256, kernel_size=3, in_stride=2, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(256, 320, 320, kernel_size=3, in_stride=2, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(320, 320, 320, kernel_size=3, in_stride=2, out_stride=1, padding=1, dropout=dropout)
        )

    def forward(self, x):
        return self.stages(x)

class UNetDecoder(nn.Module):
    def __init__(self, encoder, num_classes=3, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        
        self.stages = nn.ModuleList([
            StackedConvBlocks(640, 320, 320, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(512, 256, 256, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),  
            StackedConvBlocks(256, 128, 128, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(128, 64, 64, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(64, 32, 32, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),  
        ])

        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
            nn.ConvTranspose3d(320, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
        ])
        
        self.seg_layers = nn.ModuleList([
            nn.Conv3d(320, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(256, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(128, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        ])

    def forward(self, x):
        encoder_outputs = []
        for stage in self.encoder.stages:
            x = stage(x)
            encoder_outputs.append(x)

        x = encoder_outputs[-1]  
        seg_outputs = []

        for i in range(len(self.stages)):
            x = self.transpconvs[i](x)          
            x = torch.cat([x, encoder_outputs[-2 - i]], dim=1)
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))

        return seg_outputs[-1]

class PlainConvUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, dropout=0.0):
        super().__init__()
        self.encoder = PlainConvEncoder(in_channels, dropout)
        self.decoder = UNetDecoder(self.encoder, num_classes, dropout)

    def forward(self, x):
        return self.decoder(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, dropout=0.0, pretrain_path=None):
        super().__init__()

        self.model = PlainConvUNet(in_channels, num_classes, dropout)

        if pretrain_path:
            print("Loading weights from pretrained model!")
            target_weights = torch.load(pretrain_path)["network_weights"]
            dict1 = target_weights
            dict2 = self.model.state_dict()
            new_dict = {new_key: value for new_key, value in zip(dict2.keys(), dict1.values())}
            self.model.load_state_dict(new_dict, strict=True)
        else:
            print("Model will be randomly initialized!")
    
    def forward(self, x):
        return self.model(x)

class ChannelSELayer(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, number_of_remaining_feature_maps: int = 2, acti_type_1: str = "leakyrelu", acti_type_2: str = "sigmoid"):
        super(ChannelSELayer, self).__init__()

        pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.avg_pool = pool_type(1)

        self.r = number_of_remaining_feature_maps
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            Act[acti_type_1](inplace=True),
            nn.Linear(in_channels, in_channels, bias=True),
            Act[acti_type_2](),
        )

    def forward(self, x: torch.Tensor):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        y = y.reshape(b, c)
        channel_indices = torch.topk(y, self.r).indices

        sliced_tensor = []
        for i in range(b):
            important_channels = channel_indices[i]
            sliced_tensor.append(torch.unsqueeze(x[i, important_channels, :, :, :], dim=0))

        sliced_tensor = torch.concat(sliced_tensor, dim=0)
        # sliced_tensor = x[:, channel_indices, :, :, :]
        
        return sliced_tensor

class MixtureOfExperts(nn.Module):
    def __init__(self):
        super().__init__()

        #############################################
        # Encoder ###################################
        #############################################

        # Organ
        self.unet_organ = UNet(pretrain_path="/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset291_TotalSegmentator_part1_organs_1559subj/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                               num_classes=25)
        for param in self.unet_organ.parameters():
            param.requires_grad = False        
        self.encoder1 = self.unet_organ.model.encoder
        
        # Vertebrae
        self.unet_vertebrae = UNet(pretrain_path="/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                               num_classes=27)
        for param in self.unet_vertebrae.parameters():
            param.requires_grad = False        
        self.encoder2 = self.unet_vertebrae.model.encoder
        
        # Cardiac
        self.unet_cardiac = UNet(pretrain_path="/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj/Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                               num_classes=19)
        for param in self.unet_cardiac.parameters():
            param.requires_grad = False        
        self.encoder3 = self.unet_cardiac.model.encoder
        
        # Muscle
        self.unet_muscle = UNet(pretrain_path="/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset294_TotalSegmentator_part4_muscles_1559subj/Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                num_classes=24)
        for param in self.unet_muscle.parameters():
            param.requires_grad = False
        self.encoder4 = self.unet_muscle.model.encoder

        # Ribs
        self.unet_ribs = UNet(pretrain_path="/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset295_TotalSegmentator_part5_ribs_1559subj/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                num_classes=27)
        for param in self.unet_ribs.parameters():
            param.requires_grad = False
        self.encoder5 = self.unet_ribs.model.encoder

        #############################################
        # Decoder ###################################
        #############################################
        dropout = 0.0 
        self.stages = nn.ModuleList([
            StackedConvBlocks(640, 320, 320, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(512, 256, 256, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),  
            StackedConvBlocks(256, 128, 128, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(128, 64, 64, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),
            StackedConvBlocks(64, 32, 32, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=dropout),  
        ])

        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
            nn.ConvTranspose3d(320, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
        ])
        
        num_classes = 2
        self.seg_layers = nn.ModuleList([
            nn.Conv3d(320, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(256, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(128, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        ])

        ###############################################
        # Squeeeze & Excitation #######################
        ###############################################

        self.se0 = ChannelSELayer(spatial_dims=3, in_channels=5*32, number_of_remaining_feature_maps=32)
        self.se1 = ChannelSELayer(spatial_dims=3, in_channels=5*64, number_of_remaining_feature_maps=64)
        self.se2 = ChannelSELayer(spatial_dims=3, in_channels=5*128, number_of_remaining_feature_maps=128)
        self.se3 = ChannelSELayer(spatial_dims=3, in_channels=5*256, number_of_remaining_feature_maps=256)
        self.se4 = ChannelSELayer(spatial_dims=3, in_channels=5*320, number_of_remaining_feature_maps=320)
        self.se5 = ChannelSELayer(spatial_dims=3, in_channels=5*320, number_of_remaining_feature_maps=320)        

    def forward(self, x):

        temp = x
        encoder1_outputs = []
        for stage in self.encoder1.stages:
            x = stage(x)
            encoder1_outputs.append(x)

        x = temp
        encoder2_outputs = []
        for stage in self.encoder2.stages:
            x = stage(x)
            encoder2_outputs.append(x)

        x = temp
        encoder3_outputs = []
        for stage in self.encoder3.stages:
            x = stage(x)
            encoder3_outputs.append(x)

        x = temp
        encoder4_outputs = []
        for stage in self.encoder4.stages:
            x = stage(x)
            encoder4_outputs.append(x)

        x = temp
        encoder5_outputs = []
        for stage in self.encoder5.stages:
            x = stage(x)
            encoder5_outputs.append(x)        

        encoder_outputs = []
        encoder_outputs.append(self.se0(torch.concat([encoder1_outputs[0], encoder2_outputs[0], encoder3_outputs[0], encoder4_outputs[0], encoder5_outputs[0]], dim=1)))
        encoder_outputs.append(self.se1(torch.concat([encoder1_outputs[1], encoder2_outputs[1], encoder3_outputs[1], encoder4_outputs[1], encoder5_outputs[1]], dim=1)))
        encoder_outputs.append(self.se2(torch.concat([encoder1_outputs[2], encoder2_outputs[2], encoder3_outputs[2], encoder4_outputs[2], encoder5_outputs[2]], dim=1)))
        encoder_outputs.append(self.se3(torch.concat([encoder1_outputs[3], encoder2_outputs[3], encoder3_outputs[3], encoder4_outputs[3], encoder5_outputs[3]], dim=1)))
        encoder_outputs.append(self.se4(torch.concat([encoder1_outputs[4], encoder2_outputs[4], encoder3_outputs[4], encoder4_outputs[4], encoder5_outputs[4]], dim=1)))
        encoder_outputs.append(self.se5(torch.concat([encoder1_outputs[5], encoder2_outputs[5], encoder3_outputs[5], encoder4_outputs[5], encoder5_outputs[5]], dim=1)))

        x = encoder_outputs[-1]
        
        seg_outputs = []
        for i in range(len(self.stages)):
            x = self.transpconvs[i](x)          
            x = torch.cat([x, encoder_outputs[-2 - i]], dim=1)
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))

        return seg_outputs[-1]


if __name__ == "__main__":

    input_tensor = torch.randn(8, 1, 128, 128, 128)  
    
    weights = "/home/johannes/Code/totalsegmentator/Dataset305_vertebrae_discs_1559subj/nnUNetTrainer_DASegOrd0__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"   

    weights = {"organs": "/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset291_TotalSegmentator_part1_organs_1559subj/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
               "vertebrae": "/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
               "cardiac": "/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj/Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
               "muscles": "/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset294_TotalSegmentator_part4_muscles_1559subj/Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
               "ribs": "/home/johannes/Code/totalsegmentator/data/pretrained_weights/Dataset295_TotalSegmentator_part5_ribs_1559subj/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
               "none": None}
    
    classes = {"organs": 25,
               "vertebrae": 27,
               "cardiac": 19,
               "muscles": 24,
               "ribs": 27,
               "none": 3}

    model = MixtureOfExperts()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    output = model(input_tensor)
    print(output.shape) 

    target_structure = "organs"
    model = UNet(pretrain_path=weights[target_structure], num_classes=classes[target_structure])    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    output = model(input_tensor)
    print(output.shape) 

    encoder = model.model.encoder
    output = encoder(input_tensor)
    print(output.shape)
