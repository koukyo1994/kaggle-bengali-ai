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


class Resnet(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained=False,
                 head="linear",
                 in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.base = getattr(models, model_name)(pretrained=pretrained)
        self.head = head
        assert in_channels in [1, 3]
        assert head in ["linear", "custom"]
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
            self.grapheme_head = nn.Sequential(
                Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                nn.BatchNorm2d(512), GeM(), nn.Linear(512, 168))
            self.vowel_head = nn.Sequential(
                Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                nn.BatchNorm2d(512), GeM(), nn.Linear(512, 11))
            self.consonant_head = nn.Sequential(
                Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
                nn.BatchNorm2d(512), GeM(), nn.Linear(512, 7))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.head == "linear":
            return self.base(x)
        elif self.head == "custom":
            x = self.base(x)
            grapheme = self.grapheme_head(x)
            vowel = self.vowel_head(x)
            consonant = self.consonant_head(x)
            return torch.cat([grapheme, vowel, consonant], dim=1)
        else:
            raise NotImplementedError


def get_model(config: edict):
    params = config.model
    if "resnet" in params.model_name:
        return Resnet(**params)
    else:
        raise NotImplementedError
