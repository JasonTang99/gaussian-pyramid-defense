import torch
import torch.nn as nn
import torch.functional as F
import math

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

"""
Adapted from https://github.com/yjn870/REDNet-pytorch
"""
class REDNet10(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=5, num_features=64, use_bias=True):
        super(REDNet10, self).__init__()
        conv_layers = []
        deconv_layers = []

        # encoding layers
        conv_layers.append(nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(num_features)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                             nn.ReLU(inplace=True),
                                             nn.BatchNorm2d(num_features)))

        # decoding layers
        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                               nn.ReLU(inplace=True),
                                               nn.BatchNorm2d(num_features)))

        deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        resid = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += resid
        out = self.relu(out)
        return out


class REDNet20(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=10, num_features=64, use_bias=True):
        super(REDNet20, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(num_features)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                             nn.ReLU(inplace=True),
                                             nn.BatchNorm2d(num_features)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                               nn.ReLU(inplace=True),
                                               nn.BatchNorm2d(num_features)))

        deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x


# some reference
class DAE1(nn.Module):
    def __init__(self):
        super(DAE1, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
                
        return x

class DAE2(nn.Module):
    def __init__(self):
        super(DAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0),
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0),
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size = (3,3), stride = 1, padding = 1),
            nn.Sigmoid()
        )
        
    def forward(self, images):
        x = self.encoder(images)
        x = self.decoder(x)
        return x