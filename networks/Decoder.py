
import torch
import torch.nn as nn
from networks.weight_initialization import weight_init

class Decoder(nn.Module):
    def __init__(self, latent_dim, k=3, conv_shape=None):
        super(Decoder, self).__init__()
        self.num_filters = 32
        self.latent_dim  = latent_dim
        self.k = k

        if conv_shape is None:
            # fallback to original hard-coded sizes
            conv_shape = (self.num_filters, 35, 35)

        self._conv_shape = conv_shape
        flatten_size = conv_shape[0] * conv_shape[1] * conv_shape[2]

        self.fc_1 = nn.Linear(self.latent_dim, flatten_size)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=k, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

        self.apply(weight_init)

    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = x.view(-1, self._conv_shape[0], self._conv_shape[1], self._conv_shape[2])
        x = self.deconvs(x)
        return x
