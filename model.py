import torch
import torch.nn as nn
from typing import Literal
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, Pool, split_args

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

class ChannelSELayerOwn(nn.Module):
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

class ChannelSELayer(nn.Module):
    """
    Re-implementation of the Squeeze-and-Excitation block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        r: int = 2,
        acti_type_1: tuple[str, dict] | str = ("relu", {"inplace": True}),
        acti_type_2: tuple[str, dict] | str = "sigmoid",
        add_residual: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: activation type of the hidden squeeze layer. Defaults to ``("relu", {"inplace": True})``.
            acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".

        Raises:
            ValueError: When ``r`` is nonpositive or larger than ``in_channels``.

        See also:

            :py:class:`monai.networks.layers.Act`

        """
        super().__init__()

        self.add_residual = add_residual

        pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

        channels = int(in_channels // r)
        if channels <= 0:
            raise ValueError(f"r must be positive and smaller than in_channels, got r={r} in_channels={in_channels}.")

        act_1, act_1_args = split_args(acti_type_1)
        act_2, act_2_args = split_args(acti_type_2)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels, bias=True),
            Act[act_1](**act_1_args),
            nn.Linear(channels, in_channels, bias=True),
            Act[act_2](**act_2_args),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
        """
        b, c = x.shape[:2]
        y: torch.Tensor = self.avg_pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        result = x * y

        # Residual connection is moved here instead of providing an override of forward in ResidualSELayer since
        # Torchscript has an issue with using super().
        if self.add_residual:
            result += x

        return result

class SEBlock(nn.Module):
    """
    Residual module enhanced with Squeeze-and-Excitation::

        ----+- conv1 --  conv2 -- conv3 -- SE -o---
            |                                  |
            +---(channel project if needed)----+

    Re-implementation of the SE-Resnet block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_chns_1: int,
        n_chns_2: int,
        n_chns_3: int,
        conv_param_1: dict | None = None,
        conv_param_2: dict | None = None,
        conv_param_3: dict | None = None,
        project: Convolution | None = None,
        r: int = 2,
        acti_type_1: tuple[str, dict] | str = ("relu", {"inplace": True}),
        acti_type_2: tuple[str, dict] | str = "sigmoid",
        acti_type_final: tuple[str, dict] | str | None = ("relu", {"inplace": True}),
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            n_chns_1: number of output channels in the 1st convolution.
            n_chns_2: number of output channels in the 2nd convolution.
            n_chns_3: number of output channels in the 3rd convolution.
            conv_param_1: additional parameters to the 1st convolution.
                Defaults to ``{"kernel_size": 1, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}``
            conv_param_2: additional parameters to the 2nd convolution.
                Defaults to ``{"kernel_size": 3, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}``
            conv_param_3: additional parameters to the 3rd convolution.
                Defaults to ``{"kernel_size": 1, "norm": Norm.BATCH, "act": None}``
            project: in the case of residual chns and output chns doesn't match, a project
                (Conv) layer/block is used to adjust the number of chns. In SENET, it is
                consisted with a Conv layer as well as a Norm layer.
                Defaults to None (chns are matchable) or a Conv layer with kernel size 1.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: activation type of the hidden squeeze layer. Defaults to "relu".
            acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".
            acti_type_final: activation type of the end of the block. Defaults to "relu".

        See also:

            :py:class:`monai.networks.blocks.ChannelSELayer`

        """
        super().__init__()

        if not conv_param_1:
            conv_param_1 = {"kernel_size": 1, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}
        self.conv1 = Convolution(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=n_chns_1, **conv_param_1
        )

        if not conv_param_2:
            conv_param_2 = {"kernel_size": 3, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}
        self.conv2 = Convolution(spatial_dims=spatial_dims, in_channels=n_chns_1, out_channels=n_chns_2, **conv_param_2)

        if not conv_param_3:
            conv_param_3 = {"kernel_size": 1, "norm": Norm.BATCH, "act": None}
        self.conv3 = Convolution(spatial_dims=spatial_dims, in_channels=n_chns_2, out_channels=n_chns_3, **conv_param_3)

        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims, in_channels=n_chns_3, r=r, acti_type_1=acti_type_1, acti_type_2=acti_type_2
        )

        if project is None and in_channels != n_chns_3:
            self.project = Conv[Conv.CONV, spatial_dims](in_channels, n_chns_3, kernel_size=1)
        elif project is None:
            self.project = nn.Identity()
        else:
            self.project = project

        if acti_type_final is not None:
            act_final, act_final_args = split_args(acti_type_final)
            self.act = Act[act_final](**act_final_args)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
        """
        residual = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, y = self.se_layer(x)
        x += residual
        x = self.act(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, num_classes: int = 2, 
                 unfreeze_epoch: int = 1000, 
                 se_type: Literal["simple", "advanced"] = None, 
                 modality: Literal["CT", "MRI"] = "CT", 
                 decoder_dropout: float = 0.0, 
                 score_eval_n_epochs: int = 10):
        super().__init__()

        self.num_classes = num_classes
        self.unfreeze_epoch = unfreeze_epoch          
        self.se_type = se_type
        self.modality = modality
        self.decoder_dropout = decoder_dropout
        self.score_eval_n_epochs = score_eval_n_epochs
        self.encoders = []           

        #############################################
        # Encoders ##################################
        #############################################

        if self.modality == "CT":
            # Organ
            self.unet_organ = UNet(pretrain_path="./data/pretrained_weights/CT/Dataset291_TotalSegmentator_part1_organs_1559subj/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                   num_classes=25)
            for param in self.unet_organ.parameters():
                param.requires_grad = False        
            # self.encoder1 = self.unet_organ.model.encoder
            self.encoders.append(self.unet_organ.model.encoder)
            
            # Vertebrae
            self.unet_vertebrae = UNet(pretrain_path="./data/pretrained_weights/CT/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                       num_classes=27)
            for param in self.unet_vertebrae.parameters():
                param.requires_grad = False        
            # self.encoder2 = self.unet_vertebrae.model.encoder
            self.encoders.append(self.unet_vertebrae.model.encoder)
            
            # Cardiac
            self.unet_cardiac = UNet(pretrain_path="./data/pretrained_weights/CT/Dataset293_TotalSegmentator_part3_cardiac_1559subj/Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                     num_classes=19)
            for param in self.unet_cardiac.parameters():
                param.requires_grad = False        
            # self.encoder3 = self.unet_cardiac.model.encoder
            self.encoders.append(self.unet_cardiac.model.encoder)
            
            # Muscle
            self.unet_muscle = UNet(pretrain_path="./data/pretrained_weights/CT/Dataset294_TotalSegmentator_part4_muscles_1559subj/Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                    num_classes=24)
            for param in self.unet_muscle.parameters():
                param.requires_grad = False
            # self.encoder4 = self.unet_muscle.model.encoder
            self.encoders.append(self.unet_muscle.model.encoder)

            # Ribs
            self.unet_ribs = UNet(pretrain_path="./data/pretrained_weights/CT/Dataset295_TotalSegmentator_part5_ribs_1559subj/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                                  num_classes=27)
            for param in self.unet_ribs.parameters():
                param.requires_grad = False
            # self.encoder5 = self.unet_ribs.model.encoder
            self.encoders.append(self.unet_ribs.model.encoder)
            
            # Heptic Vessel
            # TODO: implement
        
        elif self.modality == "MRI":
            pass
            # Organ
            # TODO: implement

            # Muscle
            # TODO: implement

            # Vertebrae
            # TODO: implement

        self.scores = torch.zeros(len(self.encoders))
        self.num_encoders = len(self.encoders)
        self.num_encoder_stages = len(self.encoders[0].stages)
        self.channel_attention = nn.ModuleList([nn.Identity()]*self.num_encoder_stages)
        self.encoders = nn.ModuleList(self.encoders)        

        #############################################
        # Decoder (shared) ##########################
        #############################################
        
        self.stages = nn.ModuleList([
            StackedConvBlocks(640, 320, 320, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=self.decoder_dropout),
            StackedConvBlocks(512, 256, 256, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=self.decoder_dropout),  
            StackedConvBlocks(256, 128, 128, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=self.decoder_dropout),
            StackedConvBlocks(128, 64, 64, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=self.decoder_dropout),
            StackedConvBlocks(64, 32, 32, kernel_size=3, in_stride=1, out_stride=1, padding=1, dropout=self.decoder_dropout)  
        ])

        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
            nn.ConvTranspose3d(320, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])
        
        self.seg_layers = nn.ModuleList([
            nn.Conv3d(320, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(256, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(128, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        ])

        ###############################################
        # Squeeeze & Excitation #######################
        ###############################################        
            
        if self.se_type == "simple":
            print("Using ChannelSELayer!")
            self.channel_attention = nn.ModuleList([
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*32, r=self.num_encoders), 
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*64, r=self.num_encoders),
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*128, r=self.num_encoders),
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*256, r=self.num_encoders),
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*320, r=self.num_encoders),
                ChannelSELayer(spatial_dims=3, in_channels=self.num_encoders*320, r=self.num_encoders)
            ])
        
        elif self.se_type == "advanced":
            print("Using SEBlock!")
            # SEBlock code needs to be tested!
            self.channel_attention = nn.ModuleList([
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*32, r=self.num_encoders), 
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*64, r=self.num_encoders),
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*128, r=self.num_encoders),
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*256, r=self.num_encoders),
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*320, r=self.num_encoders),
                SEBlock(spatial_dims=3, in_channels=self.num_encoders*320, r=self.num_encoders)
            ])   
        else:
            print("Not using Squeeze & Excitation!")

        ###############################################
        # 1x1x1 Convolution Channel Mixing ############
        ###############################################
        self.conv1x1x1 = nn.ModuleList([
            nn.Conv3d(in_channels=self.num_encoders*32, out_channels=32, kernel_size=1, bias=False),
            nn.Conv3d(in_channels=self.num_encoders*64, out_channels=64, kernel_size=1, bias=False),
            nn.Conv3d(in_channels=self.num_encoders*128, out_channels=128, kernel_size=1, bias=False),
            nn.Conv3d(in_channels=self.num_encoders*256, out_channels=256, kernel_size=1, bias=False),
            nn.Conv3d(in_channels=self.num_encoders*320, out_channels=320, kernel_size=1, bias=False),
            nn.Conv3d(in_channels=self.num_encoders*320, out_channels=320, kernel_size=1, bias=False)
        ])

    def forward(self, x, epoch=-1):

        # Encode x using all encoders and append stage ouputs to list
        temp = x
        encoder_outputs = []
        for encoder in self.encoders:
            x = temp
            for stage in encoder.stages:
                 x = stage(x)
                 encoder_outputs.append(x)       
        
        temp = []
        for i in range(self.num_encoder_stages):
            # Concat same stages of encoders on feature map dimension
            out = torch.concat(encoder_outputs[i::self.num_encoder_stages], dim=1)
            # Apply channel attention to concatenated features
            out = self.channel_attention[i](out)
            # Reduce feature map dimension to dimension of a single encoder
            out = self.conv1x1x1[i](out)
            temp.append(out)
        encoder_outputs = temp
        x = encoder_outputs[-1]

        if (epoch+1) % self.score_eval_n_epochs == 0:
            # Calculate encoder scores
            for i in range(self.num_encoder_stages):
                scores = [self.get_importance_from_weights(weights=self.conv1x1x1[i].weight)][0]
                self.scores = self.scores + scores
                self.scores = self.scores / torch.sum(self.scores)                       

        if epoch == self.unfreeze_epoch+1:
            
            enc_idx = torch.argmax(self.scores).item()
            for param in self.encoders[enc_idx].parameters():
                param.requires_grad = True 

            print(f"Unfreezing encoder: {enc_idx}")
            print(f"All encoder scores: {self.scores}")       

            with open('encoder_scores.txt', 'a') as file:     
                print(f"Unfreezing encoder: {enc_idx}", file=file)
                print(f"All encoder scores: {self.scores}", file=file)            
       
        seg_outputs = []
        for i in range(len(self.stages)):
            x = self.transpconvs[i](x)          
            x = torch.cat([x, encoder_outputs[-2 - i]], dim=1)
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))

        return seg_outputs[-1]
    
    def get_importance_from_weights(self, weights: torch.Tensor, print_results:bool = False) -> torch.Tensor:

        f = weights.shape[1]
        n = weights.shape[0]
        m = 5
        k = f//m

        # Extract the weight matrix (shape: [n, f, 1, 1, 1] -> squeeze to [n, f])
        weight_matrix = weights.detach().cpu().squeeze()

        # Compute feature importance: sum of absolute values of weights per feature map
        feature_importance = weight_matrix.abs().sum(dim=0)  # Shape (f,)

        # Get the top n most important feature map indices
        important_feature_indices = torch.argsort(feature_importance, descending=True)[:n]

        # Compute encoder contribution ratios
        encoder_counts = torch.zeros(m)  # Count of selected features per encoder
        for idx in important_feature_indices:
            encoder_id = idx // k  # Determine encoder index
            encoder_counts[encoder_id] += 1

        # Normalize to get ratios
        encoder_ratios = encoder_counts / n

        if print_results:
            print("Most important feature indices:", important_feature_indices.tolist())
            print("Feature importance scores:", feature_importance[important_feature_indices].tolist())
            print("Feature count per encoder:", encoder_counts.tolist())
            print("Encoder contribution ratios:", encoder_ratios.tolist())

        encoder_ratios = torch.tensor(encoder_ratios.tolist(), dtype=torch.float32)

        return encoder_ratios


if __name__ == "__main__":

    input_tensor = torch.randn(2, 1, 128, 128, 128)  
    
    model = MixtureOfExperts(num_classes=2,                             
                             unfreeze_epoch=1000, 
                             se_type="simple", 
                             modality="CT", 
                             score_eval_n_epochs=1)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    
    output = model(input_tensor)
    print(output.shape) 
    
    # weights = {"organs": "/home/johannes/Code/totalsegmentator/data/CT/pretrained_weights/Dataset291_TotalSegmentator_part1_organs_1559subj/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    #            "vertebrae": "/home/johannes/Code/totalsegmentator/CT/data/pretrained_weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    #            "cardiac": "/home/johannes/Code/totalsegmentator/CT/data/pretrained_weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj/Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    #            "muscles": "/home/johannes/Code/totalsegmentator/CT/data/pretrained_weights/Dataset294_TotalSegmentator_part4_muscles_1559subj/Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    #            "ribs": "/home/johannes/Code/totalsegmentator/CT/data/pretrained_weights/Dataset295_TotalSegmentator_part5_ribs_1559subj/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    #            "none": None}
    
    # classes = {"organs": 25,
    #            "vertebrae": 27,
    #            "cardiac": 19,
    #            "muscles": 24,
    #            "ribs": 27,
    #            "none": 3}   

    # target_structure = "organs"
    # model = UNet(pretrain_path=weights[target_structure], num_classes=classes[target_structure])    
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {total_params}")
    # output = model(input_tensor)
    # print(output.shape) 

    # encoder = model.model.encoder
    # output = encoder(input_tensor)
    # print(output.shape)
