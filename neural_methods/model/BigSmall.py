import torch
import torch.nn as nn

class BigSmall(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, dropout_rate3=0.5, pool_size1=(2, 2), pool_size2=(4, 4), nb_dense=128,
                 out_size_au=12, out_size_ppg=1, out_size_emotion=8, n_segment=3):
        super(BigSmall, self).__init__()

        self.n_segment = n_segment
        self.out_size_au = out_size_au
        self.out_size_ppg = out_size_ppg
        self.out_size_emotion = out_size_emotion

        # Big Branch
        self.big_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1)
        self.big_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.big_avg_pooling1 = nn.AvgPool2d(pool_size1)
        self.big_dropout1 = nn.Dropout(dropout_rate1)

        self.big_conv3 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.big_conv4 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1)
        self.big_avg_pooling2 = nn.AvgPool2d(pool_size1)
        self.big_dropout2 = nn.Dropout(dropout_rate2)

        self.big_conv5 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, padding=1)
        self.big_conv6 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, padding=1)
        self.big_avg_pooling3 = nn.AvgPool2d(pool_size2)
        self.big_dropout3 = nn.Dropout(dropout_rate3)

        # TSM modules
        self.TSM_1 = WTSM(n_segment=n_segment)
        self.TSM_2 = WTSM(n_segment=n_segment)
        self.TSM_3 = WTSM(n_segment=n_segment)
        self.TSM_4 = WTSM(n_segment=n_segment)

        # Small Branch
        self.small_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1)
        self.small_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.small_conv3 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.small_conv4 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1)

        # Modular Heads
        self.au_head = nn.Sequential(
            nn.Linear(5184, nb_dense),
            nn.ReLU(),
            nn.Linear(nb_dense, out_size_au)
        )
        self.ppg_head = nn.Sequential(
            nn.Linear(5184, nb_dense),
            nn.ReLU(),
            nn.Linear(nb_dense, out_size_ppg)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(5184, nb_dense),
            nn.ReLU(),
            nn.Linear(nb_dense, out_size_emotion)
        )

        # Learnable log variances for uncertainty-based loss weighting
        # self.log_sigma_au = nn.Parameter(torch.zeros(1))
        # self.log_sigma_ppg = nn.Parameter(torch.zeros(1))
        # self.log_sigma_emotion = nn.Parameter(torch.zeros(1))
        self.log_sigma_au = nn.Parameter(torch.tensor(1.2))
        self.log_sigma_ppg = nn.Parameter(torch.tensor(0.1))
        self.log_sigma_emotion = nn.Parameter(torch.tensor(1.5))

    def forward(self, inputs):
        big_input, small_input = inputs
        BT, C, H, W = big_input.size()
        n_batch = BT // self.n_segment

        big_input = big_input.view(n_batch, self.n_segment, C, H, W)
        big_input = torch.moveaxis(big_input, 1, 2)[:, :, 0, :, :]  # Use only the first frame

        # Big path
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        b13 = torch.stack((b12, b12, b12), dim=2)  # Replicate for segment compatibility
        b14 = torch.moveaxis(b13, 1, 2)
        b15 = b14.reshape(BT, b14.shape[2], b14.shape[3], b14.shape[4])

        # Small path
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Fusion
        concat = b15 + s8
        flat = concat.view(concat.size(0), -1)

        # Heads
        au_output = self.au_head(flat)
        ppg_output = self.ppg_head(flat)
        emotion_output = self.emotion_head(flat)

        return au_output, ppg_output, emotion_output, self.log_sigma_au, self.log_sigma_ppg, self.log_sigma_emotion


class WTSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        super(WTSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        BT, C, H, W = x.size()
        n_batch = BT // self.n_segment
        x = x.view(n_batch, self.n_segment, C, H, W)

        fold = C // self.fold_div
        out = torch.zeros_like(x)

        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, -1, :fold] = x[:, 0, :fold]

        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, 0, fold:2*fold] = x[:, -1, fold:2*fold]

        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(BT, C, H, W)