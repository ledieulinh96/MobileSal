import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import BatchNorm2d
import math

import os,datetime




# Define the source file path
source_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
# Get the current date and time formatted as YYYYMMDD-HHMMSS
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Define the destination file path, including the timestamp in the filename
destination_dir = '/root/linhle/DL/MobileSal/MobileSal/snapshots/'
destination_file = f'{timestamp}_{current_file_name}.py'
destination_path = os.path.join(destination_dir, destination_file)

# Ensure the destination directory exists, create if it doesn't
os.makedirs(destination_dir, exist_ok=True)

# Copy the file
shutil.copy(source_path, destination_path)

print(f"File copied from {source_path} to {destination_path}")
        #5: 10,320,10,10
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x): #x: 4D tensor: batch size, channels, height, and width.
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        # Reshape and transpose to align for matrix multiplication for height-wise attention:
        #permute: changes the order to (batch size, width, channels, height).
        #contiguous(): This function is used to ensure that the tensor is stored in a contiguous block. This step is required before reshaping the tensor.
        #view(m_batchsize*width, -1, height): This function reshapes the tensor. After permutation, the tensor is reshaped into a new form where:
            #The first dimension is m_batchsize * width, effectively treating each width slice across all batches as a separate entity.
            #The second dimension -1 tells PyTorch to calculate the necessary size to maintain the same number of elements in the tensor, which will be equivalent to the number of channels.
            #The third dimension is explicitly set to height
        #This final permutation swaps the second and third dimensions, resulting in a tensor with dimensions (m_batchsize*width, height, channels).
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) #reshape and hoan vi computes the attention scores separately for the height and width dimensions.
        # Reshape and transpose to align for matrix multiplication for width-wise attention:

        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x) 
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
                # Calculate attention energy scores, apply masking, and reshape:
        #batch matrix-multiplication between proj_query_H and proj_key_H
        #with each H×H matrix corresponding to the attention weights between all pairs of height positions for each width slice.
        #permute(0, 2, 1, 3): Rearranges the dimensions of the tensor to move the 'width' dimension. The tensor shape becomes [B,H,W,H].
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3) # computed by batch matrix multiplication
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
                # Concatenate the energies and apply softmax:
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) 
        
        
                # Separate the attention scores for height and width:
        #extracts the section of the tensor relevant to the height-wise attention, resulting in a tensor of shape [m_batchsize, width, height, height].
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)

        #print(concate)
        #print(att_H) 
        #extracts the section of the tensor relevant to the width-wise attention. It starts from height and extends to height + width, resulting in a tensor of shape [m_batchsize, width, height, width].
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        
                # Compute the final output by applying the attention weights to the value projections and adding input x:
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        print(out_H.size(),out_W.size())
                # Scale the output with gamma and add the original input (residual connection):
        return self.gamma*(out_H + out_W) + x #scaling the sum of the attention-modified feature maps from both dimensions using gamma, and adding this result back to the original input 


    def forward(self, x, x_d=None):
        if x_d is None:
            x_d = x
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x_d)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)

        proj_value = self.value_conv(x_d)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)

        energy_H = torch.bmm(proj_query_H, proj_key_H).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        att_H = concate[:, :, :, :height].permute(0, 2, 1, 3).contiguous().view(m_batchsize*width, height, height)
        att_W = concate[:, :, :, height:height+width].contiguous().view(m_batchsize*height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x
    


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RCCAModule(nn.Module):
    def __init__(self, in_channels):
        super(RCCAModule, self).__init__()
        #inter_channels = in_channels // 4
        # self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(in_channels)
        # self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            InPlaceABNSync(inter_channels))

    def forward(self, x, x_d, recurrence=1):
        #output = self.conva(x)
        output = self.cca(x, x_d)
        for i in range(recurrence-1):
            output = self.cca(output)
        #output = self.convb(output)
        #output = self.bottleneck(torch.cat([x, output], 1))
        return output
