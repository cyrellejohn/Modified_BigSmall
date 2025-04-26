"""
BigSmall: Multitask Network for AU / Respiration / PPG

BigSmall: Efficient Multi-Task Learning
For Physiological Measurements
Girish Narayanswamy, Yujia (Nancy) Liu, Yuzhe Yang, Chengqian (Jack) Ma, 
Xin Liu, Daniel McDuff, Shwetak Patel

https://arxiv.org/abs/2303.11573
"""

import torch
import torch.nn as nn


#######################################################################################
##################################### BigSmall Model ##################################
#######################################################################################
class BigSmall(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25, 
                 dropout_rate2=0.5, dropout_rate3=0.5, pool_size1=(2, 2), pool_size2=(4,4), nb_dense=128, 
                 out_size_au=12, out_size_ppg=1, out_size_emotion=8, n_segment=3):
        """
        Initialize the BigSmall model for multi-task learning.

        Parameters:
        - in_channels: Number of input channels (e.g., 3 for RGB images).
        - nb_filters1: Number of filters for the first set of convolutional layers.
        - nb_filters2: Number of filters for the second set of convolutional layers.
        - kernel_size: Size of the convolutional kernel.
        - dropout_rate1, dropout_rate2, dropout_rate3: Dropout rates for different layers.
        - pool_size1, pool_size2: Pooling sizes for average pooling layers.
        - nb_dense: Number of units in the dense (fully connected) layers.
        - out_size_ppg, out_size_au: Output sizes for BVP, respiration, and AU tasks.
        - out_size_emotion: Output size for emotion task.
        - n_segment: Number of segments for temporal processing.
        """

        super(BigSmall, self).__init__()
        
        # Model Parameters
        self.in_channels = in_channels
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2
        self.nb_dense = nb_dense

        self.out_size_au = out_size_au
        self.out_size_ppg = out_size_ppg
        self.out_size_emotion = out_size_emotion
        self.n_segment = n_segment

        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d(self.pool_size2)
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)

        # Time Shift Modules (TSM) Layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense, self.out_size_au, bias=True)

        # PPG Fully Connected Layers 
        self.ppg_fc1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.ppg_fc2 = nn.Linear(self.nb_dense, self.out_size_ppg, bias=True)

    def forward(self, inputs, params=None):
        """
        Forward pass of the BigSmall model.

        Parameters:
        - inputs: A tuple containing two elements:
            - inputs[0]: High-resolution input tensor.
            - inputs[1]: Low-resolution input tensor.
        - params: Optional parameters for the forward pass (not used in this implementation).

        Notes:
        - Read more about chunk, batch and segment here: https://tinyurl.com/mpmd94pu

        Returns:
        - au_output: Output tensor for the Action Unit (AU) task.
        - ppg_output: Output tensor for the Blood Volume Pulse (BVP) task.
        """

        # Extract big and small resolution inputs
        big_input = inputs[0] # High-resolution input
        small_input = inputs[1] # Low-resolution input
        
        # Reshape big_input to separate batch and segment dimensions
        BT, C, H, W = big_input.size() # BT is batch_size * frames per chunk
        n_batch = BT // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, C, H, W)
        big_input = torch.moveaxis(big_input, 1, 2) # Move color channel to index 1, sequence channel to index 2 
        big_input = big_input[:, :, 0, :, :] # Use only the first frame in sequences

        # Big Conv block 1: Apply two convolutional layers with ReLU activation, followed by average pooling and dropout to reduce overfitting.
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2: Similar to block 1, with additional convolutional layers.
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3: Final set of convolutional layers for the big input branch.
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)      

        # Stack and reshape the output to prepare for concatenation with the small branch.
        b13 = torch.stack((b12, b12, b12), 2) # TODO: This is hardcoded for n_segment | FINALIZE
        b14 = torch.moveaxis(b13, 1, 2)
        big_B, big_T, big_C, big_H, big_W = b14.size() # B: batch_size. T: frames per segment
        b15 = b14.reshape(int(big_B * big_T), big_C, big_H, big_W) 

        # Small Conv block 1: Process the low-resolution input through TSM and convolutional layers.
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2: Continue processing with additional TSM and convolutional layers.
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers: Combine the outputs from the big and small branches.
        concat = b15 + s8

        # Flatten the concatenated result to prepare for fully connected layers.
        share1 = concat.reshape(concat.size(0), -1) 
        # Alternative: share1 = concat.view(concat.size(0), -1)

        # AU Output Layers: Process through fully connected layers for AU task.
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_output = self.au_fc2(aufc1)

        # PPG Output Layers: Process through fully connected layers for PPG task.
        ppgfc1 = nn.functional.relu(self.ppg_fc1(share1))
        ppg_output = self.ppg_fc2(ppgfc1)

        # Return the outputs for AU and PPG tasks.
        return au_output, ppg_output


#####################################################
############ Wrapping Time Shift Module #############
#####################################################
class WTSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        """
        Initializes the WTSM (Wrapping Time Shift Module) class.

        Parameters:
        n_segment (int): The number of segments to divide the input into. Default is 3.
        fold_div (int): The divisor to determine how input channels are divided into folds. Default is 3.

        This constructor sets up the WTSM module with the specified number of segments and fold division,
        which are used in the forward pass to perform time-shift operations on input tensors.
        """
        super(WTSM, self).__init__()
        self.n_segment = n_segment # Number of segments for input division
        self.fold_div = fold_div # Divisor for channel folding

    def forward(self, x):
        """
        Perform a wrapping time shift operation on the input tensor.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (BT, C, H, W), where:
            - BT is the product of the batch size and the number of segments.
            - C is the number of channels.
            - H and W are the height and width of the input.

        Notes: 
        - Detailed explanation: https://tinyurl.com/2c7abmwz

        Returns:
        - torch.Tensor: Output tensor of the same shape as the input (BT, C, H, W),
        with time-shifted channels.

        The function divides the input channels into three folds and performs the following operations:
        - The first fold of channels is shifted left by one segment, with the last segment wrapping to the first position.
        - The second fold of channels is shifted right by one segment, with the first segment wrapping to the last position.
        - The third fold of channels remains unchanged.

        This operation is designed to help the model learn temporal dependencies by shifting parts of the input across time segments.
        """

        # Get the dimensions of the input tensor
        BT, C, H, W = x.size()

        # Calculate the number of batches by dividing the total batch size by the number of frames per chunk
        n_batch = BT // self.n_segment

        # Reshape the input tensor to separate the batch and segment dimensions
        x = x.view(n_batch, self.n_segment, C, H, W)

        # Calculate the number of channels per fold
        fold = C // self.fold_div

        # Initialize an output tensor with the same shape as the reshaped input
        output = torch.zeros_like(x)

        # Shift the first fold of channels to the left by one segment | SHIFT LEFT
        output[:, :-1, :fold] = x[:, 1:, :fold] 

        # Wrap the first fold of channels from the last segment to the first | WRAP LEFT
        output[:, -1, :fold] = x[:, 0, :fold]

        # Shift the second fold of channels to the right by one segment | SHIFT RIGHT
        output[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]

        # Wrap the second fold of channels from the first segment to the last | WRAP RIGHT
        output[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]

        # Keep the final fold of channels unchanged | NO SHIFT AND WRAP
        output[:, :, 2 * fold:] = x[:, :, 2 * fold:]  

        # Reshape the output tensor back to its original shape
        return output.view(BT, C, H, W)