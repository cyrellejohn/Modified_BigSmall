"""
DeepPhys - 2D Convolutional Attention Network
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff
"""

import torch
import torch.nn as nn

class Attention_mask(nn.Module):
    def __init__(self):
        """
        Initializes the Attention_mask module.
        Inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
        """
        super(Attention_mask, self).__init__()

    def forward(self, x):
        """
        Forward pass for the Attention_mask module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input, representing the attention mask.
        
        The method computes a spatial attention mask by normalizing the input tensor across its spatial dimensions.
        The mask is scaled by the product of the input's height and width, and a factor of 0.5.
        """
        # Sum the input tensor along the height dimension
        xsum = torch.sum(x, dim=2, keepdim=True)

        # Sum the result along the width dimension
        xsum = torch.sum(xsum, dim=3, keepdim=True)

        # Capture the shape of the input tensor
        xshape = tuple(x.size())

        # Normalize the input tensor by the summed values and scale by the spatial dimensions and a factor of 0.5
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """
        Returns the configuration of the Attention_mask module.
        
        This method is a placeholder and may be used for compatibility with frameworks that utilize configuration dictionaries.
        
        Returns:
            dict: Configuration dictionary of the module.
        """
        config = super(Attention_mask, self).get_config()
        return config

class DeepPhys(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        """
        Initializes the DeepPhys model.

        Args:
          in_channels (int): Number of input channels. Default is 3 for RGB images.
          nb_filters1 (int): Number of filters in the first set of convolutional layers.
          nb_filters2 (int): Number of filters in the second set of convolutional layers.
          kernel_size (int): Size of the convolutional kernel.
          dropout_rate1 (float): Dropout rate for the first set of dropout layers.
          dropout_rate2 (float): Dropout rate for the second set of dropout layers.
          pool_size (tuple): Size of the pooling window for average pooling layers.
          nb_dense (int): Number of units in the dense (fully connected) layer.
          img_size (int): Size of the input image (height/width). Supported sizes are 36, 72, and 96.

        Raises:
          Exception: If img_size is not supported.
        """
        super(DeepPhys, self).__init__()

        # Initialize parameters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        
        # Motion branch convolutional layers
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Appearance branch convolutional layers
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Average pooling layers
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        """
        Perform the forward pass of the DeepPhys model.

        Args:
            inputs (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
            params (optional): Additional parameters, not used in this method.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """

        # Split the input tensor into two parts: motion and appearance
        diff_input = inputs[:, :3, :, :] # First three channels for motion
        raw_input = inputs[:, 3:, :, :] # Remaining channels for appearance

        # Motion branch processing
        d1 = torch.tanh(self.motion_conv1(diff_input)) # First motion convolution + tanh
        d2 = torch.tanh(self.motion_conv2(d1)) # Second motion convolution + tanh

        # Appearance branch processing
        r1 = torch.tanh(self.apperance_conv1(raw_input)) # First appearance convolution + tanh
        r2 = torch.tanh(self.apperance_conv2(r1)) # Second appearance convolution + tanh

        # Attention mechanism for the first set of layers
        g1 = torch.sigmoid(self.apperance_att_conv1(r2)) # Attention convolution + sigmoid
        g1 = self.attn_mask_1(g1) # Apply attention mask
        gated1 = d2 * g1 # Element-wise multiplication for gating

        # Downsample and apply dropout to the gated motion output
        d3 = self.avg_pooling_1(gated1) # Average pooling
        d4 = self.dropout_1(d3) # Dropout

        # Downsample and apply dropout to the appearance output
        r3 = self.avg_pooling_2(r2) # Average pooling
        r4 = self.dropout_2(r3) # Dropout

        # Further motion branch processing
        d5 = torch.tanh(self.motion_conv3(d4)) # Third motion convolution + tanh
        d6 = torch.tanh(self.motion_conv4(d5)) # Fourth motion convolution + tanh

        # Further appearance branch processing
        r5 = torch.tanh(self.apperance_conv3(r4)) # Third appearance convolution + tanh
        r6 = torch.tanh(self.apperance_conv4(r5)) # Fourth appearance convolution + tanh

        # Attention mechanism for the second set of layers
        g2 = torch.sigmoid(self.apperance_att_conv2(r6)) # Attention convolution + sigmoid
        g2 = self.attn_mask_2(g2) # Apply attention mask
        gated2 = d6 * g2 # Element-wise multiplication for gating

        # Downsample, apply dropout, and reshape for the dense layer
        d7 = self.avg_pooling_3(gated2) # Average pooling
        d8 = self.dropout_3(d7) # Dropout
        d9 = d8.view(d8.size(0), -1) # Flatten the tensor

        # Fully connected layers with dropout
        d10 = torch.tanh(self.final_dense_1(d9)) # First dense layer + tanh
        d11 = self.dropout_4(d10) # Dropout
        out = self.final_dense_2(d11) # Final dense layer

        return out # Return the final output
    