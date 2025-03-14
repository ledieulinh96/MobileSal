import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PyramidDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidDecoder, self).__init__()
        self.inners = nn.ModuleList()  # Lateral connections: 1x1 conv to reduce channel dimensions
        self.outs = nn.ModuleList()    # Output layers

        for i in range(len(in_channels)):
            self.inners.append(nn.Conv2d(in_channels[i], out_channels, kernel_size=1))
            self.outs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
    def forward(self, features):
        results = []
        upsampled = None
        for idx in reversed(range(len(features))):
            lateral = self.inners[idx](features[idx])
            logging.debug(f"Lateral output at level {idx}: {lateral.shape}")
            
            if upsampled is not None:
                upsampled = F.interpolate(upsampled, size=lateral.shape[2:], mode='bilinear', align_corners=False)
                logging.debug(f"Upsampled output to level {idx}: {upsampled.shape}")
            
            combined = lateral if upsampled is None else lateral + upsampled
            upsampled = self.outs[idx](combined)
            results.insert(0, upsampled)
            logging.debug(f"Output at level {idx}: {upsampled.shape}")

        return results

                #  enc_channels=[16, 24, 32, 96, 320],
                #  dec_channels=[16, 24, 32, 96, 320]
# Example usage:
model = PyramidDecoder(in_channels=[256, 512, 1024, 2048], out_channels=256)
dummy_features = [torch.randn(1, 256, 64, 64), torch.randn(1, 512, 32, 32), torch.randn(1, 1024, 16, 16), torch.randn(1, 2048, 8, 8)]
outputs = model(dummy_features)
