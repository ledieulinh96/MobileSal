import torch
import torch.nn as nn
import torch.nn.functional as F

class CoNet(nn.Module):
    """
    Model that generates an output with the same channel size as the input image, based on another
    'question' image, using a parallel co-attention mechanism.
    """
    def __init__(self, img_channels, feature_dim=128, attention_dim=64):
        super().__init__()
        self.img_channels = img_channels  # Store the number of input channels
        question_channels = img_channels
        
        # Convolution layers for feature extraction
        self.image_conv = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, attention_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.question_conv = nn.Sequential(
            nn.Conv2d(question_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, attention_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Parallel co-attention module
        self.parallel_co_attention = ParallelCoAttention(attention_dim)
        
        # Additional layer to match the output channels to the input channels
        self.final_conv = nn.Conv2d(attention_dim, img_channels, kernel_size=1)

    def forward(self, image, question_image):
        # Extract features from both images
        image_features = self.image_conv(image)
        question_features = self.question_conv(question_image)

        # Compute attention map using parallel co-attention
        attention_features = self.parallel_co_attention(image_features, question_features)

        # Use final convolution layer to ensure output channels match the input image channels
        output = self.final_conv(attention_features)
        output = torch.sigmoid(output)  # Normalize the output
        return output

class ParallelCoAttention(nn.Module):
    """
    A parallel co-attention module that computes attention maps by aligning features from two inputs.
    """
    def __init__(self, channels):
        super().__init__()
        self.attention_map_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.attention_map_q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, V, Q):
        # Generate attention maps for both V (image features) and Q (question image features)
        attention_v = self.attention_map_v(V)
        attention_q = self.attention_map_q(Q)

        # Combine the attention maps
        combined_attention = attention_v * attention_q
        return combined_attention

# Example usage
if __name__ == "__main__":
    img_channels = 3  # Example for RGB images
    question_channels = 3  # Assuming the question image is also RGB
    model = CoattentionNet(img_channels, question_channels)
    image = torch.randn(1, img_channels, 256, 256)
    question_image = torch.randn(1, question_channels, 256, 256)
    output = model(image, question_image)
    print("Output size:", output.size())  # Should be [1, 3, 256, 256]
