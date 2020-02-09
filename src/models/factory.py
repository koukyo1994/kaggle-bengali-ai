import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from easydict import EasyDict as edict
from torch.nn.parameter import Parameter


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p),
                        (x.size(-2), x.size(-1))).pow(1. / p)


def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps).squeeze(-1).squeeze(-1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
            self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class SEResNext(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained=None,
                 head="linear",
                 in_channels=3,
                 outputs=["grapheme", "vowel", "consonant"]):
        super().__init__()
        self.num_classes = num_classes
        self.base = getattr(pretrainedmodels.models,
                            model_name)(pretrained=pretrained)
        self.head = head
        assert in_channels in [1, 3]
        assert head in ["linear", "custom", "scse"]
        for out in outputs:
            assert out in {"grapheme", "vowel", "consonant"}
        self.outputs = outputs
        if in_channels == 1:
            if pretrained == "imagenet":
                weight = self.base.layer0.conv1.weight
                self.base.layer0.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.base.layer0.conv1.weight = nn.Parameter(
                    data=torch.mean(weight, dim=1, keepdim=True),
                    requires_grad=True)
            else:
                self.base.layer0.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if head == "linear":
            n_in_features = self.base.last_linear.in_features
            self.base.last_linear = nn.Linear(n_in_features, self.num_classes)
        elif head == "custom":
            n_in_features = self.base.last_linear.in_features
            arch = list(self.base.children())
            for _ in range(2):
                arch.pop()
            self.base = nn.Sequential(*arch)
            if "grapheme" in self.outputs:
                self.grapheme_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Dropout(0.3),
                    nn.Linear(512, 168))
            if "vowel" in self.outputs:
                self.vowel_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Dropout(0.3),
                    nn.Linear(512, 11))
            if "consonant" in self.outputs:
                self.consonant_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Dropout(0.3),
                    nn.Linear(512, 7))
        elif head == "scse":
            n_in_features = self.base.last_linear.in_features
            arch = list(self.base.children())
            for _ in range(2):
                arch.pop()
            self.base = nn.Sequential(*arch)
            if "grapheme" in self.outputs:
                self.grapheme_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 168))
            if "vowel" in self.outputs:
                self.vowel_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 11))
            if "consonant" in self.outputs:
                self.consonant_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 7))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.head == "linear":
            return self.base(x)
        elif self.head == "custom" or self.head == "scse":
            x = self.base(x)
            outputs = []
            if "grapheme" in self.outputs:
                grapheme = self.grapheme_head(x)
                outputs.append(grapheme)
            if "vowel" in self.outputs:
                vowel = self.vowel_head(x)
                outputs.append(vowel)
            if "consonant" in self.outputs:
                consonant = self.consonant_head(x)
                outputs.append(consonant)
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError


class Resnet(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained=False,
                 head="linear",
                 in_channels=3,
                 outputs=["grapheme", "vowel", "consonant"]):
        super().__init__()
        self.num_classes = num_classes
        self.base = getattr(models, model_name)(pretrained=pretrained)
        self.head = head
        assert in_channels in [1, 3]
        assert head in ["linear", "custom", "scse"]
        for out in outputs:
            assert out in {"grapheme", "vowel", "consonant"}
        self.outputs = outputs
        if in_channels == 1:
            if pretrained:
                weight = self.base.conv1.weight
                self.base.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.base.conv1.weight = nn.Parameter(
                    data=torch.mean(weight, dim=1, keepdim=True),
                    requires_grad=True)
            else:
                self.base.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if head == "linear":
            n_in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(n_in_features, self.num_classes)
        elif head == "custom":
            n_in_features = self.base.fc.in_features
            arch = list(self.base.children())
            for _ in range(2):
                arch.pop()
            self.base = nn.Sequential(*arch)
            if "grapheme" in self.outputs:
                self.grapheme_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Linear(512, 168))
            if "vowel" in self.outputs:
                self.vowel_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Linear(512, 11))
            if "consonant" in self.outputs:
                self.consonant_head = nn.Sequential(
                    Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                    nn.BatchNorm2d(512), GeM(), nn.Linear(512, 7))
        elif head == "scse":
            n_in_features = self.base.fc.in_features
            arch = list(self.base.children())
            for _ in range(2):
                arch.pop()
            self.base = nn.Sequential(*arch)
            if "grapheme" in self.outputs:
                self.grapheme_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 168))
            if "vowel" in self.outputs:
                self.vowel_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 11))
            if "consonant" in self.outputs:
                self.consonant_head = nn.Sequential(
                    SCse(n_in_features), Mish(), nn.BatchNorm2d(512), GeM(),
                    nn.Dropout(0.3), nn.Linear(512, 7))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.head == "linear":
            return self.base(x)
        elif self.head == "custom" or self.head == "scse":
            x = self.base(x)
            outputs = []
            if "grapheme" in self.outputs:
                grapheme = self.grapheme_head(x)
                outputs.append(grapheme)
            if "vowel" in self.outputs:
                vowel = self.vowel_head(x)
                outputs.append(vowel)
            if "consonant" in self.outputs:
                consonant = self.consonant_head(x)
                outputs.append(consonant)
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError


def get_model(config: edict):
    params = config.model
    if "resnet" in params.model_name:
        return Resnet(**params)
    elif "se_resnext" in params.model_name:
        return SEResNext(**params)
    else:
        raise NotImplementedError
