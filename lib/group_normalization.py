import torch
from torch import nn


class GroupNormND(nn.Module):
    def __init__(self, ND, in_channels, G, channels_per_group=None, eps=1e-5):
        """
        :param ND: [int] the dimension of input feature maps. (B, C, H, W) is 2D.
        :param in_channels: [int] channel of input feature maps. (B, C, H, W) is C.
        :param G: [int or None] the group number. If G is set to int, then channels_per_group must be None.
        :param channels_per_group: [int or None] It it is set to int, then G must be None.
        :param eps: default to 1e-5
        """

        super(GroupNormND, self).__init__()

        assert G is None or channels_per_group is None
        assert not (G is None and channels_per_group is None)

        if G is not None:
            assert in_channels % G == 0
        else:
            assert in_channels % channels_per_group == 0

        self.ND = ND
        spatial_dimension = [1] * ND

        self.gamma = nn.Parameter(torch.ones(1, in_channels, *spatial_dimension))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, *spatial_dimension))
        self.eps = eps
        self.G = G
        self.channels_per_group = channels_per_group

    def forward(self, x):
        input_size = x.size()
        N, C= input_size[:2]

        if self.G is not None:
            groups = self.G
        elif self.channels_per_group is not None:
            groups = C // self.channels_per_group

        x = x.view(N, groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # the return of tf.nn.moments is mean and variance in the paper.
        # For simplicity and less computation, I directly use standard deviation to add eps.
        x = (x - mean) / (std+self.eps)

        # you can use the three code below to verify the mean is 0, std is 1.
        # print(x.size())
        # print(x.mean(dim=-1))
        # print(x.std(dim=-1))

        x = x.view(input_size)
        return x * self.gamma + self.beta


class GroupNorm1D(GroupNormND):
    def __init__(self, in_channels, G, channels_per_group=None, eps=1e-5):
        super(GroupNorm1D, self).__init__(1, in_channels, G, channels_per_group, eps)


class GroupNorm2D(GroupNormND):
    def __init__(self, in_channels, G, channels_per_group=None, eps=1e-5):
        super(GroupNorm2D, self).__init__(2, in_channels, G, channels_per_group, eps)


class GroupNorm3D(GroupNormND):
    def __init__(self, in_channels, G, channels_per_group=None, eps=1e-5):
        super(GroupNorm3D, self).__init__(3, in_channels, G, channels_per_group, eps)

if __name__ == '__main__':
    x = torch.randn(1, 10, 10)
    out = GroupNorm1D(10, 5, None)(x)
    print(out.size())

    x = torch.randn(1, 10, 10, 10)
    out = GroupNorm2D(10, 5, None)(x)
    print(out.size())

    x = torch.randn(1, 10, 3, 10, 10)
    out = GroupNorm3D(10, 5, None)(x)
    print(out.size())