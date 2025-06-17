from torch import nn

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []

        prev_conv = None

        for l in config.values():
            conv = l['conv2d']
            layers.append(nn.Conv2d(3 if not prev_conv else prev_conv['channels'], conv['channels'], conv['kernel'], padding=conv['padding'], stride=conv['stride'], bias=False))
            layers.append(nn.BatchNorm2d(conv['channels'], affine=True))
            layers.append(nn.LeakyReLU())

            if 'dropout' in l:
                dropout = l['dropout']
                layers.append(nn.Dropout2d(p=dropout['prob']))

            if 'maxpool' in l:
                maxpool = l['maxpool']
                layers.append(nn.MaxPool2d(maxpool['kernel'], stride=maxpool['stride'], padding=maxpool['padding']))

            prev_conv = conv


        layers.append(nn.Conv2d(prev_conv['channels'], 10, 1))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.cnn = nn.Sequential(*layers)

    def forward(self, images):
        return self.cnn(images).view(images.size(0), -1)
