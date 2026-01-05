from torch import nn
from funcs import get_layers

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cnn = nn.Sequential(*get_layers(config['generator']))

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            m.bias is not None and nn.init.constant_(m.bias.data, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            m.bias is not None and nn.init.constant_(m.bias.data, 0)

    
    def forward(self, z):
        return self.cnn(z)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cnn = nn.Sequential(*get_layers(config['discriminator']))

        self.apply(self._weights_init)

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            m.bias is not None and nn.init.constant_(m.bias.data, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            m.bias is not None and nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        return self.cnn(input).view(len(input), -1)