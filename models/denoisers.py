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

    def __init__(self, in_channels=3, out_channels=3, depth=17, hidden_channels=64,
                use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=use_bias))
            layers.append(torch.nn.BatchNorm2d(hidden_channels))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = x
        residual = self.net(x)
        return y - residual