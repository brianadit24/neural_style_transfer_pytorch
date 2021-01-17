from torch import nn
from torchvision.models import vgg16


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        
        model = vgg16(pretrained=True).eval()
        self.model = model.features
        self.freeze()
        
    def forward(self, x, layers):
        features = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in layers:
                features.append(x)
        return features
    
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False