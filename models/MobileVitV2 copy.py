from transformers import ViTFeatureExtractor
from torch import nn
import torch
import timm

from transformers import MobileViTV2ForImageClassification
from PIL import Image


class MobileVitV2(nn.Module):
    def __init__(self, pretrained=None, num_classes=1000, depth = False):
        super(MobileVitV2, self).__init__()
        self.model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        
        if pretrained:
            print("Loading pre-trained weights...")
            self.model.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            print("Pre-trained weights loaded successfully.")
            
        #get_graph_node_names(model)
        #print(self.model)
        self.conv_stem = self.model.mobilevitv2.conv_stem
                # Adjust the input layer to accept 1 channel
        if (depth): 
            self.conv_stem.convolution = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.layer1 = self.model.mobilevitv2.encoder.layer[0]
        self.layer2 = self.model.mobilevitv2.encoder.layer[1]
        self.layer3 = self.model.mobilevitv2.encoder.layer[2]
        self.layer4 = self.model.mobilevitv2.encoder.layer[3]

    def forward(self, x):
        res = []
        
        # Early convolutional layer
        x = self.conv_stem(x)
        res.append(x)
        
        # First middle layer
        x = self.layer1(x)
        res.append(x)
        
        # Second middle layer
        x = self.layer2(x)
        res.append(x)
        
        # First deep layer
        x = self.layer3(x)
        res.append(x)
        
        # Second deep layer
        x = self.layer4(x)
        res.append(x)
        
        return res

def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    model = MobileViTBackbone(**kwargs)
    #model = MobileVitV2(**kwargs)
    
        # Print the names and shapes of the model parameters
    #print("\nModel parameters after loading weights:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape} {'(loaded)' if param.requires_grad else '(not loaded)'}")
    
    return model


# Define the MobileViT Backbone
class MobileViTBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(MobileViTBackbone, self).__init__()
        self.backbone = timm.create_model('mobilevitv2_150', pretrained=True, features_only=True, in_chans=in_channels)

    def forward(self, x):
        return self.backbone(x)

def save_features(features, stage):
    num_features = min(len(features), 5)  # Limit the number of features to display
    fig, axarr = plt.subplots(1, num_features, figsize=(15, 5))
    if num_features == 1:
        axarr = [axarr]  # Make it iterable if there's only one subplot

    for idx in range(num_features):
        axarr[idx].imshow(features[idx][0].cpu().detach().numpy(), cmap='viridis')
        axarr[idx].set_title(f'Stage {stage} Feature {idx+1}')
        axarr[idx].axis('off')
    
    plt.savefig(f'feature_maps/stage_{stage}_features.png')
    plt.close()
    

import matplotlib.pyplot as plt
import os

    # model = MobileVitV2()
# sample_input = torch.randn(1, 3, 224, 224)
# outputs = model(sample_input)
# for i, output in enumerate(outputs):
#     print(f"Stage {i+1} output shape: {output.shape}")
    
#Stage 1 output shape: torch.Size([1, 32, 112, 112])
# Stage 2 output shape: torch.Size([1, 64, 112, 112])
# Stage 3 output shape: torch.Size([1, 128, 56, 56])
# Stage 4 output shape: torch.Size([1, 256, 28, 28])
# Stage 5 output shape: torch.Size([1, 384, 14, 14])

# Initialize the backbone and inspect feature maps
model = MobileViTBackbone()
# Stage 1 output shape: torch.Size([1, 32, 112, 112])
# Stage 2 output shape: torch.Size([1, 64, 56, 56])
# Stage 3 output shape: torch.Size([1, 128, 28, 28])
# Stage 4 output shape: torch.Size([1, 192, 14, 14])
# Stage 5 output shape: torch.Size([1, 256, 8, 8])
sample_input = torch.randn(1, 3, 224, 224)
outputs = model(sample_input)
for i, output in enumerate(outputs):
    print(f"Stage {i+1} output shape: {output.shape}")
    
# Visualize features (e.g., first feature map of each stage) and save to file
# Create a directory to save the feature maps
os.makedirs('feature_maps', exist_ok=True)


# Example visualization of the first feature map of each stage
for i, output in enumerate(outputs):
    save_features(output.permute(1, 0, 2, 3), i+1)
        

