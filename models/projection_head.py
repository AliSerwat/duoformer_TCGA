import torch
from torch import nn

class Projection(nn.Module):
    def __init__(self,num_layers = 2, proj_dim = 768, backbone = 'r50'):
        super().__init__()
        if backbone == 'r50' or backbone == 'r50_Swav' or backbone == 'r50_BT' or backbone =='r50_MoCoV2':
            if num_layers == 1:
                self.proj_heads = nn.Conv2d(2048, proj_dim, kernel_size=(1,1),stride=(1,1))
                # self.proj_heads = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads)
            elif num_layers == 2:
                self.proj_heads3 = nn.Conv2d(2048, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(1024, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
            elif num_layers == 3:
                self.proj_heads3 = nn.Conv2d(2048, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(1024, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads1 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
            elif num_layers == 4:
                self.proj_heads3 = nn.Conv2d(2048, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(1024, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads1 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads0 = nn.Conv2d(256, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
                self._initialize_weights(self.proj_heads0)
        elif backbone == 'r18':
            if num_layers == 1:
                self.proj_heads = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads)
            elif num_layers == 2:
                self.proj_heads3 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(256, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
            elif num_layers == 3:
                self.proj_heads3 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(256, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads1 = nn.Conv2d(128, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
            elif num_layers == 4:
                self.proj_heads3 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(256, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads1 = nn.Conv2d(128, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads0 = nn.Conv2d(64, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
                self._initialize_weights(self.proj_heads0)

        # else:
        #     self.proj_heads = nn.ModuleDict()
        #     for i in range(num_layers):
        #         # self.proj_heads[f'{i}'] = nn.Linear(256 * (2 ** i), proj_dim)
        #         self.proj_heads[f'{3-i}'] = nn.Conv2d(256 * (2 ** (3-i)), proj_dim, kernel_size=(1,1),stride=(1,1))
        #         self._initialize_weights(self.proj_heads[f'{i}'])

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # nn.init.xavier_uniform_(module.weight)  # nn.init.xavier_uniform_() or nn.init.xavier_normal_()
            nn.init.kaiming_normal_(module.weight)  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self,x):
        if len(x) != 1:
            proj_features = {}
            #print(x.shape)
            for k, fea in x.items():
                N,C,H,W = fea.shape
                if k == '3':
                    proj_features[k] = self.proj_heads3(fea)
                elif k == '2':
                    proj_features[k] = self.proj_heads2(fea)
                elif k == '1':
                    proj_features[k] = self.proj_heads1(fea)
                elif k == '0':
                    proj_features[k] = self.proj_heads0(fea)
        else:
            proj_features = self.proj_heads(x)
        return proj_features

class Projection_for_hybrid(nn.Module):
    def __init__(self, proj_dim = 768, backbone = 'r50'):
        super().__init__()
        if backbone == 'r50' or backbone == 'r50_Swav' or backbone == 'r50_BT' or backbone =='r50_MoCoV2':
            self.proj_head = nn.Conv2d(2048, proj_dim, kernel_size=(1,1),stride=(1,1))
        elif backbone == 'r18':
            self.proj_head = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
        self._initialize_weights(self.proj_head)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # nn.init.xavier_uniform_(module.weight)  # nn.init.xavier_uniform_() or nn.init.xavier_normal_()
            nn.init.kaiming_normal_(module.weight)  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self,x):
        proj_features = self.proj_head(x)
        return proj_features
    
class Channel_Projector_layer1(nn.Module):
    def __init__(self, backbone='r50'):
        super().__init__()
        # Convolutional layers to reduce spatial dimensions
        if backbone=='r50' or backbone == 'r50_Swav' or backbone == 'r50_BT' or backbone =='r50_MoCoV2':
            self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        elif backbone=='r18':
            self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Pooling layer to downsample to 7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights(self.conv1)
        self._initialize_weights(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.activation1(x)
        x = self.conv2(x)
        # x = self.norm2(x)
        # x = self.activation2(x)
        x = self.pool(x)
        return x

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)


class Channel_Projector_layer2(nn.Module):
    def __init__(self, backbone='r50'):
        super().__init__()
        # Convolutional layers to reduce spatial dimensions
        if backbone=='r50' or backbone == 'r50_Swav' or backbone == 'r50_BT' or backbone =='r50_MoCoV2':
            self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        elif backbone=='r18':
            self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.activation1(x)
        x = self.pool(x)
        return x
    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)
    
class Channel_Projector_layer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class Channel_Projector_All(nn.Module):
    def __init__(self, backbone='r50', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if backbone=='r50' or backbone == 'r50_Swav' or backbone == 'r50_BT' or backbone =='r50_MoCoV2':
            self.nConvs = _make_nConv(3840, 768, 4)
        elif backbone=='r18':
            self.nConvs = _make_nConv(960, 768, 4)

    def forward(self, x):
        return torch.flatten(self.nConvs(x), start_dim=2) 