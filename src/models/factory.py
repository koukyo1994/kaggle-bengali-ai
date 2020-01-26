import pretrainedmodels
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class BengaliClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        if "se_resnext" in self.model_name:
            self.base = getattr(pretrainedmodels,
                                self.model_name)(pretrained=pretrained)
            self.base.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.base.last_linear = nn.Linear(
                self.base.last_linear.in_features, self.num_classes)
        elif "resnet" in self.model_name:
            self.base = getattr(pretrainedmodels,
                                self.model_name)(pretrained=pretrained)
            self.base.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.base.fc = nn.Linear(self.base.fc.in_features,
                                     self.num_classes)
        elif "efficientnet" in self.model_name:
            if pretrained:
                self.base = EfficientNet.from_pretrained(self.model_name)
            else:
                self.base = EfficientNet.from_name(self.model_name)
            self.base._fc = nn.Linear(self.base._fc.in_features,
                                      self.num_classes)
        else:
            raise NotImplementedError

    def fresh_params(self):
        if "se_resnext" in self.model_name:
            return self.base.last_linear.parameters()
        elif "resnet" in self.model_name:
            return self.base.fc.parameters()
        elif "efficientnet" in self.model_name:
            return self.base._fc.parameters()
        else:
            raise NotImplementedError

    def base_params(self):
        params = []
        if "se_resnext" in self.model_name:
            fc_name = "last_linear"
        elif "resnet" in self.model_name:
            fc_name = "fc"
        elif "efficientnet" in self.model_name:
            fc_name = "_fc"
        else:
            raise NotImplementedError
        for name, param in self.net.named_parameters():
            if fc_name not in name:
                params.append(param)
        return params

    def forward(self, x):
        return self.base(x)
