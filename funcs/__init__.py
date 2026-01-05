from torch import nn

def get_layers(setup):
    layers = []

    for l in setup:
        if 'conv2d' in l:
            conv = l['conv2d']
            layers.append(nn.Conv2d(
                conv['in'], 
                conv['out'], 
                conv['kernel'], 
                padding=conv['padding'], 
                stride=conv['stride'], 
                bias=conv['bias']
            ))
        
        if 'convT2d' in l:
            convt = l['convT2d']
            layers.append(nn.ConvTranspose2d(
                convt['in'],
                convt['out'],
                convt['kernel'],
                convt['stride'],
                padding=convt['padding'],
                bias=convt['bias']
            ))

        if 'leakyRelu' in l:
            x = l['leakyRelu']
            layers.append(nn.LeakyReLU(
                negative_slope=x['nslope'], 
                inplace=x['inplace']
            ))

        if 'relu' in l:
            x = l['relu']
            layers.append(nn.ReLU(
                inplace=x['inplace']
            ))
        
        if 'batchNorm2d' in l:
            x = l['batchNorm2d']
            layers.append(nn.BatchNorm2d(
                num_features=x['in']
            ))

        if 'layerNorm' in l:
            x = l['layerNorm']
            layers.append(nn.LayerNorm(
                normalized_shape=x['shape'],
                bias=x['bias']
            ))
        
        if 'adaptativeAvgPool2d' in l:
            x = l['adaptativeAvgPool2d']
            layers.append(nn.AdaptiveAvgPool2d(
                output_size=x['output_size']
            ))
        
        if 'linear' in l:
            linear = l['linear']
            layers.append(nn.Linear(linear['in'], linear['out']))

        if 'dropout' in l:
            dropout = l['dropout']
            layers.append(nn.Dropout2d(p=dropout['prob']))
        
        if 'maxpool2d' in l:
            maxpool = l['maxpool2d']
            layers.append(nn.MaxPool2d(maxpool['kernel'], stride=maxpool['stride'], padding=maxpool['padding']))

        if 'tanh' in l:
            layers.append(nn.Tanh())

        if 'sigmoid' in l:
            layers.append(nn.Sigmoid())


    return layers