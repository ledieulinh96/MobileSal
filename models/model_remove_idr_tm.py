import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import BatchNorm2d
import math

from MobileNetV2 import mobilenet_v2
from torch.nn import Parameter
import os,datetime

from cc_1 import RCCAModule
from co import CoNet
from coattention_net import CoattentionNet
from torchsummary import summary
import dataset as myDataLoader
import Transforms as myTransforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
# Define the source file path
source_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
# Get the current date and time formatted as YYYYMMDD-HHMMSS
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Define the destination file path, including the timestamp in the filename
destination_dir = '/home/linh/backup/'+current_file_name
destination_file = f'{timestamp}.py'
destination_path = os.path.join(destination_dir, destination_file)

# Ensure the destination directory exists, create if it doesn't
os.makedirs(destination_dir, exist_ok=True)

# Copy the file
shutil.copy(source_path, destination_path)

print(f"File copied from {source_path} to {destination_path}")


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, spectral_norm=False, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            elif spectral_norm:
                self.bn = SpectralNorm(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        #if self.scale > 1:
        #    out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileSal(nn.Module):
    
    def __init__(self, pretrained=True, use_carafe=True,
                 enc_channels=[16, 24, 32, 96, 320],
                 dec_channels=[16, 24, 32, 96, 320]):
        super(MobileSal, self).__init__()
        self.backbone = mobilenet_v2(pretrained)
        self.depthnet = DepthNet()

        #self.depth_fuse = DepthFuseNet(inchannels=320)
        #self.depth_fuse = CrissCrossAttention(320)
        #self.depth_fuse = RCCAModule(320)
        #self.depth_fuse = RCCAModule(320)

        self.fpn = MDecoder(enc_channels, dec_channels)

        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
    def loss(self, input, target):
        pass

    def forward(self, input, depth=None, test=True):

        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        #1: 10,16,160,160
        #2: 10,24,80,80
        #3: 10,32,40,40
        #4: 10,96,20,20
        #5: 10,320,10,10
        
        # RGB-D fuse & implicit depth restoration
        # if depth is not None:
        depth_features = self.depthnet(depth)
            #conv5 = self.depth_fuse(conv5, depth_features[-1]) #10,320,10,10
        #     depth_pred = None
        #     if test:
        #         depth_pred = None
        # else:
        #     depth_pred = None

        features = self.fpn([conv1, conv2, conv3, conv4, conv5], depth_features[-1])

        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:], #specifies the desired spatial dimensions, which are height and width of the input image).
                    mode='bilinear',
                    align_corners=False)
            )
            if test: #only compute the first saliency map
                break
        saliency_maps = torch.sigmoid(torch.cat(saliency_maps, dim=1))

        if test:
            return saliency_maps
        else:   
            return saliency_maps, depth_pred


class DepthNet(nn.Module):
    def __init__(self, pretrained=None,
                 use_gan=False):
        super(DepthNet, self).__init__()
        block = InvertedResidual
        input_channel = 1
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 2, 2, 1],
            [4, 32, 2, 2, 1],
            [4, 64, 2, 2, 1],
            [4, 96, 2, 2, 1],
            [4, 320, 2, 2, 1],
        ]
        features = []
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * 1.0)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 3, 5, 7, 9]:
                feats.append(x)
        return feats


class DepthFuseNet(nn.Module):
    def __init__(self, inchannels=320):
        super(DepthFuseNet, self).__init__()
        self.d_conv1 = InvertedResidual(inchannels, inchannels, residual=True)
        self.d_linear = nn.Sequential(
            nn.Linear(inchannels, inchannels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(inchannels, inchannels, bias=True),
        )
        self.d_conv2 = InvertedResidual(inchannels, inchannels, residual=True)

    def forward(self, x, x_d):
        x_f = self.d_conv1(x * x_d)
        x_d1 = self.d_linear(x.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3)
        x_f1 = self.d_conv2(torch.sigmoid(x_d1) * x_f * x_d)
        return x_f1



class CPR(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=[1,2,3], residual=True):
        super(CPR, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
                # Calculate the hidden dimension as an expansion of the input dimension.

        if self.stride == 1 and inp == oup:  # Use a residual connection if conditions are met.
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        self.conv1 = ConvBNReLU(inp, hidden_dim, ksize=1, pad=0, prelu=False)
        # Initial convolution layer to expand the channel dimension from 'inp' to 'hidden_dim'.

        # Define three separate dilated convolution layers for depthwise convolutions at different scales.

        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim, dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim, dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim, dilation=dilation[2])
                # Each convolution uses 'hidden_dim' groups to create a depthwise effect, processing each channel independently.

        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
                # Batch normalization followed by a ReLU activation function applied after the dilated convolutions.

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
                # Output convolution layer that reduces the channel dimension from 'hidden_dim' to 'oup'.


    def forward(self, x):
        m = self.conv1(x)
        m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
                # Apply all three dilated convolutions and sum the results, leveraging different receptive fields.

        m = self.hidden_bnact(m)
                # Normalize and activate the combined feature maps.

        if self.use_res_connect:
            return x + self.out_conv(m)
                    # If residual connections are used, add the input 'x' to the output of the final convolution.

        else:
            return self.out_conv(m)



class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, input_num=2):
        super(Fusion, self).__init__()
                # Conditionally initialize channel attention mechanism if two inputs are expected.
        if input_num == 2:
            self.channel_att = nn.Sequential(nn.Linear(in_channels, in_channels), # Fully connected layer to process channels.
                                             nn.ReLU(),
                                             nn.Linear(in_channels, in_channels),
                                             nn.Sigmoid()# Sigmoid activation to output weights between 0 and 1.
                                             )
                    # Initialize the main fusion module which consists of:

        self.fuse = nn.Sequential( CPR(in_channels, in_channels, expand_ratio=expansion, residual=True),             # CPR could be a custom layer (not standard in PyTorch) often used for channel-wise feature processing.

                                      ConvBNReLU(in_channels, in_channels, ksize=1, pad=0, stride=1)
                                      )


    def forward(self, low, high=None):
        if high is None:
                        # If only one input is provided, process it directly through the fusion module.

            final = self.fuse(low)
        else:
                        # If two inputs are provided, first upsample 'high' to match 'low's spatial dimensions.
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
                        # F.interpolate rescales the 'high' feature map to have the same width and height as 'low'.
            # Concatenate the upsampled 'high' and 'low' feature maps along the channel dimension.
            fuse = torch.cat((high_up, low), dim=1)
            
                        # Apply channel attention to the global average pooled features and reshape.
            # fuse.mean(dim=2).mean(dim=2) performs global average pooling by averaging across spatial dimensions.
            channel_weights = self.channel_att(fuse.mean(dim=2).mean(dim=2))
            channel_weights = channel_weights.unsqueeze(dim=2).unsqueeze(dim=2)
            # Reshape the channel weights to match the dimensions needed for element-wise multiplication.
            
            # Apply the channel-specific weights to the fused feature map and process through the fusion module.
            final = channel_weights * self.fuse(fuse)

        return final
    
class MDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDecoder, self).__init__()
        self.inners_a = nn.ModuleList()  # Lateral connections
        self.inners_b = nn.ModuleList()  # Top-down connections

        # Initialize convolutional layers for inners_a and inners_b
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        
    
        # Convolutional layer for the last channel in inners_a
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))

        # Replace with RCCA modules for fusion
        self.fuse = nn.ModuleList()
        for i in range(len(in_channels)):
            self.fuse.append(RCCAModule(out_channels[i]))
            

    def forward(self, features, depth):
        results = []
        # Process the top-most feature first and upsample
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]), depth)  # Top layer doesn't need RCCA input from above
        results.append(stage_result)
        
        # Process remaining features in reverse order
        for idx in range(len(features) - 2, -1, -1):
            # Upsample previous stage result to current feature map size
            upsampled_result = F.interpolate(stage_result, size=features[idx].shape[2:], mode='bilinear', align_corners=False)
            
            # Combine upsampled result with lateral and pass through RCCA
            inner_lateral = self.inners_a[idx](features[idx])
            inner_top_down = self.inners_b[idx](upsampled_result)

            stage_result = self.fuse[idx](inner_lateral, inner_top_down) + inner_top_down
            results.insert(0, stage_result)

        return results
    
class Last2Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDecoder, self).__init__()
        self.inners_a = nn.ModuleList()  # Lateral connections
        self.inners_b = nn.ModuleList()  # Top-down connections
        
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        self.fuse = nn.ModuleList([RCCAModule(out_channels[i]) for i in range(len(in_channels))])

    def forward(self, features, depth, depth2):
        results = []
        # Process the top-most feature first and upsample
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]), depth)  # Top layer doesn't need RCCA input from above
        results.append(stage_result)
        
        # Process remaining features in reverse order
        for idx in range(len(features) - 2, -1, -1):
            # Upsample previous stage result to current feature map size
            upsampled_result = F.interpolate(stage_result, size=features[idx].shape[2:], mode='bilinear', align_corners=False)
            if idx == len(features) - 2:
                upsampled_result = torch.cat([upsampled_result, depth2], dim=1)
            # Combine upsampled result with lateral and pass through RCCA
            inner_lateral = self.inners_a[idx](features[idx])
            inner_top_down = self.inners_b[idx](upsampled_result)

            stage_result = self.fuse[idx](inner_lateral, inner_top_down)
            results.insert(0, stage_result)

        return results
            

class CPRDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, teacher=False):
        super(CPRDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners_a = nn.ModuleList()   #  processing feature maps directly from the backbone (lateral connections)
        self.inners_b = nn.ModuleList() # processing the outputs that propagate from deeper layers in the network (top-down connections)
                        # Loop through the input channels except the last one to create corresponding convolutional layers.
        for i in range(len(in_channels) - 1):
             # `in_channels[i]` to `out_channels[i] // 2` reduction is typical in decoders to gradually increase abstraction.
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0)) #connects directly to a corresponding input feature map, reducing its channel size by half.
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0)) #processes the feature map from a deeper stage, also reducing its size by half for subsequent fusion
            # Similar processing for the top-down connection using output from the next layer.
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0)) # processing the last channel of input

        self.fuse = nn.ModuleList() # merge features from different stages
        for i in range(len(in_channels)):
            if i == len(in_channels) - 1:
                self.fuse.append(Fusion(out_channels[i], out_channels[i], input_num=1))
                                # If it's the last layer, use a single-input fusion module.

            else:
                                # Use a convolutional layer or a fusion layer based on the `teacher` flag.

                self.fuse.append(
                    ConvBNReLU(out_channels[i], out_channels[i]) if teacher else Fusion(out_channels[i], out_channels[i]) #teacher = true ->, regular convolutional layers are used
                    )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1): #            # Loop over remaining features in reverse order for top-down feature merging.

            #inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
            #                               size=features[idx].shape[2:],
            #                               mode='bilinear',
            #                               align_corners=False)
                        # Transform the result from the previous stage down to the current feature's resolution.
            inner_top_down = self.inners_b[idx](stage_result) #Applying transformation
            inner_lateral = self.inners_a[idx](features[idx]) #            # Process lateral connections at the current feature map level.

            stage_result = self.fuse[idx](inner_lateral, inner_top_down)#(torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results

class FPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners.append(ConvBNReLU(in_channels[i], out_channels[i], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        for i in range(len(out_channels)):
            self.fuse.append(
                ConvBNReLU(out_channels[i], out_channels[i]),
            )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            inner_lateral = self.inners[idx-1](features[idx])
            stage_result = self.fuse[idx](inner_top_down + inner_lateral)
            results.insert(0, stage_result)

        return results

# class ModelWrapper(nn.Module):
#     def __init__(self, model):
#         super(ModelWrapper, self).__init__()
#         self.model = model
    
#     def forward(self, x):
#         outputs = self.model(x)
#         # Assuming the main model's forward method returns a list of tensors
#         # We take the first tensor for summary
#         if isinstance(outputs, list):
#             return outputs[0]
#         return outputs



def test():
#     model = MobileSal(pretrained=False)
#     wrapped_model = ModelWrapper(model)
#     #y = net(torch.randn(1, 3, 32, 32))
#     #print(y.size())
#     wrapped_model.to('cuda')  # Move the model to GPU if available, adjust if using CPU
#     summary(wrapped_model,(3,224,224))
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     depthpred_params = sum([np.prod(p.size()) for p in model.idr.parameters()])
#     print('Total network parameters (excluding idr): ' + str(total_params - depthpred_params))

  
    mean = [0.406, 0.456, 0.485]  # [103.53,116.28,123.675]
    std = [0.225, 0.224, 0.229] # [57.375,57.12,58.395]

    # for multi-scale training
    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(256, 256),
        myTransforms.RandomCropResize(int(7. / 224. * 256)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    train_data_scale1 = myDataLoader.Dataset("NJU2K_NLPR_train", file_root='../data/', transform=trainDataset_scale1, use_depth=1)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        train_data_scale1,
        batch_size=10, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    
    # Get one batch of data
    data_iter = iter(trainLoader_scale1)
    images, labels, depth = next(data_iter)
    print("Image batch shape:", images.shape) #Image batch shape: torch.Size([10, 3, 256, 256])
    print("Label batch shape:", labels.shape) #Image batch shape: torch.Size([10, 1, 256, 256])

    # Move the data to the correct device if your model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = images.to(device), labels.to(device)

    model = MobileSal()  # Replace with your actual model
    model.to(device)  # Move model to the same device as your data
    writer.add_graph(model, images)
    writer.close()  
    outputs = model(images)  # Forward pass
    print("Output shape:", outputs.shape) #10, 1, 256, 256
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_params))
    #summary(model, input_size=images.shape[1:]) => errror vi co cai nhieu input

    
    

#test()