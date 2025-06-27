from torch import nn

# Parametric Image Compressor
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.Sequential(*self._get_layers(config['encoder']))
        self.code = nn.Sequential(*self._get_layers(config['code']))
        self.decoder = nn.Sequential(*self._get_layers(config['decoder']))


    def _get_layers(self, kv):
        layers = []
        
        for l in kv.values():
            if 'conv2d' in l:
                conv = l['conv2d']

                layers.append(nn.Conv2d(conv['in'], conv['out'], conv['kernel'], padding=conv['padding'], stride=conv['stride'], bias=False))
                layers.append(nn.BatchNorm2d(conv['out'], affine=True))
                layers.append(nn.LeakyReLU())


            if 'convT2d' in l:
                convt = l['convT2d']
                layers.append(
                    nn.ConvTranspose2d(
                        convt['in'], 
                        convt['out'], 
                        convt['kernel'], 
                        convt['stride'], 
                        padding=convt['padding'], 
                        output_padding=convt['out_padding'], 
                        bias=False
                    )
                )

                if 'output' not in l:
                    layers.append(nn.BatchNorm2d(convt['out'], affine=True))
                    layers.append(nn.LeakyReLU())


            if 'dropout' in l:
                dropout = l['dropout']
                layers.append(nn.Dropout2d(p=dropout['prob']))


            if 'maxpool' in l:
                maxpool = l['maxpool']
                layers.append(nn.MaxPool2d(maxpool['kernel'], stride=maxpool['stride'], padding=maxpool['padding']))

        
        return layers


    def forward(self, images):
        return self.decoder(self.code(self.encoder(images)))


    def encode(self, input):
        return self.code(self.encoder(input))
    

    def decode(self, input):
        return self.decoder(self.code(input))
    