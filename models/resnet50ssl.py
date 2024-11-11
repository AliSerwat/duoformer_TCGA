import torch
from torchvision.models.resnet import Bottleneck, ResNet
import os
from PathDino512 import get_pathDino_model

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ResNetTrunkByScale(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        return [x0, x1, x2, x3]

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    # URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url, model_zoo_registry.get(key)

def resnet50FeatureExtractor(pretrained, progress, key, **kwargs):
    model = ResNetTrunkByScale(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url, model_filename = get_pretrained_url(key)
        print(pretrained_url)
        if os.path.exists(model_filename):
            state_dict = torch.load(model_filename)
            verbose = model.load_state_dict(state_dict)
            print(f"Model exists. Loaded pretrained model from local file: {model_filename}")
        else:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
            torch.save(state_dict, model_filename)
            verbose = model.load_state_dict(state_dict)
            print(f"Downloaded and saved pretrained model: {model_filename}")
        print(verbose)
    return model

def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url, model_filename = get_pretrained_url(key)
        print(pretrained_url)
        if os.path.exists(model_filename):
            state_dict = torch.load(model_filename)
            verbose = model.load_state_dict(state_dict)
            print(f"Model exists. Loaded pretrained model from local file: {model_filename}")
        else:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
            torch.save(state_dict, model_filename)
            verbose = model.load_state_dict(state_dict)
            print(f"Downloaded and saved pretrained model: {model_filename}")
        print(verbose)
    return model

class ResNet50withFC(torch.nn.Module):
    def __init__(self, pretrained=True, progress=False, num_classes=4, key="BT", freeze=True):
        super().__init__()
        self.resnet_trunk = resnet50(pretrained=pretrained, progress=progress, key=key)
        if freeze: # freeze the pretrained feature extractor
            for param in self.resnet_trunk.parameters():
                param.requires_grad = False
            print('resnet_trunk frozen.')
        self.fc = torch.nn.Linear(512 * Bottleneck.expansion, num_classes)  # Add your custom linear layer

    def forward(self, x):
        x = self.resnet_trunk(x)
        x = self.resnet_trunk.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
#written by Bodong
class PathDinoClassifier(torch.nn.Module):
    def __init__(self, num_classes=4, weights_path="/uu/sci.utah.edu/tolga_scratch/bodong/projects/202404_MIL/DTFD_MIL_code/DTFD-MIL/man_pathdino/PathDino512.pth"):
        super(PathDinoClassifier, self).__init__()
        # Load the pre-trained DINO model without the classification head
        self.dino_encoder, _ = get_pathDino_model(weights_path=weights_path)
        # Assume the encoder outputs a 384-dimensional vector (from VisionTransformer's `embed_dim`)
        self.fc = torch.nn.Linear(384, num_classes)
        
    def forward(self, x):
        # Forward pass through the encoder
        features = self.dino_encoder(x)  # Using the output directly as it already corresponds to the CLS token
        # Forward pass through the classifier
        output = self.fc(features)
        return output
    
#for Bodong MIL
class ResNet50FrozenEncoderWithAvgPool(torch.nn.Module):
    def __init__(self, pretrained=True, progress=False, num_classes=4, key="SwAV", freeze=True):
        super().__init__()
        self.resnet_trunk = resnet50(pretrained=pretrained, progress=progress, key=key)
        if freeze: # freeze the pretrained feature extractor
            for param in self.resnet_trunk.parameters():
                param.requires_grad = False
            print('resnet_trunk frozen.')
        #self.fc = torch.nn.Linear(512 * Bottleneck.expansion, num_classes)  # Add your custom linear layer

    def forward(self, x):
        x = self.resnet_trunk(x)
        x = self.resnet_trunk.avgpool(x)
        x = torch.flatten(x, 1)#flatten to 2 dim
        #x = self.fc(x)
        return x

# def count_parameters(model):
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     return trainable_params/1000000, total_params/1000000

# if __name__ == "__main__":
    # initialize resnet50 trunk using BT pre-trained weight
    # model = resnet50(pretrained=True, progress=False, key="SwAV")
    # print(count_parameters(model))