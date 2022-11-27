import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Network architecture from this reference. Note that we omit batch norm
    since we are using a shallow network to speed up training times.

    @article{zhang2017beyond,
      title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
      author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
      journal={IEEE Transactions on Image Processing},
      year={2017},
      volume={26},
      number={7},
      pages={3142-3155},
    }
    """

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=3, use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same', bias=use_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)