from torch import nn

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # l1
            nn.Conv2d(3, config['l1_c'], config['l1_cnk'], bias=False, padding=1),
            nn.BatchNorm2d(config['l1_c'], affine=True),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(config['l1_pk'], config['l1_ps'], padding=1),

            # l2
            nn.Conv2d(config['l1_c'], config['l2_c'], config['l2_cnk'], bias=False, padding=1),
            nn.BatchNorm2d(config['l2_c'], affine=True),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(config['l2_pk'], config['l2_ps'], padding=1),

            # l3
            nn.Conv2d(config['l2_c'], config['l3_c'], config['l3_cnk'], bias=False, padding=1),
            nn.BatchNorm2d(config['l3_c'], affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(config['l3_pk'], config['l3_ps']),

            # l4
            nn.Conv2d(config['l3_c'], config['l4_c'], config['l4_cnk'], bias=False),
            nn.BatchNorm2d(config['l4_c'], affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(config['l4_pk'], config['l4_ps']),

            # l5
            nn.Conv2d(config['l4_c'], config['l5_c'], config['l5_cnk'], bias=False, padding=1),
            nn.BatchNorm2d(config['l5_c'], affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(config['l5_pk'], config['l5_ps']),

            # # l6
            # nn.Conv2d(config['l5_c'], config['l6_c'], config['l6_cnk'], bias=False, padding=1),
            # nn.BatchNorm2d(config['l6_c'], affine=True),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(config['l6_pk'], config['l6_ps']),

            # # l7
            # nn.Conv2d(config['l6_c'], config['l7_c'], config['l7_cnk'], bias=False, padding=1),
            # nn.BatchNorm2d(config['l7_c'], affine=True),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(config['l7_pk'], config['l7_ps'], padding=1),

            # # l8
            # nn.Conv2d(config['l7_c'], config['l8_c'], config['l8_cnk'], padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(config['l8_pk'], config['l8_ps']),

            # # l9
            # nn.Conv2d(config['l8_c'], config['l9_c'], config['l9_cnk'], padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(config['l9_pk'], config['l9_ps'], padding=1),

            # # l10
            # nn.Conv2d(config['l9_c'], config['l10_c'], config['l10_cnk']),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(config['l10_pk'], config['l10_ps']),

            
            # output
            nn.Conv2d(config['l5_c'], 10, 1),   # 10 channel == 10 classes
            nn.AdaptiveAvgPool2d(1)
        )


    def forward(self, images):
        return self.cnn(images).view(images.size(0), -1)