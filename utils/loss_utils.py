import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)


class ColorBalanceLoss(nn.Module):
    def forward(self, restored):
        channel_mean = restored.mean(dim=(-2, -1))
        r, g, b = channel_mean[:, 0], channel_mean[:, 1], channel_mean[:, 2]
        return (torch.abs(r - g) + torch.abs(g - b) + torch.abs(r - b)).mean()


class ColorDistributionLoss(nn.Module):
    def forward(self, restored, target):
        restored = restored.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        restored_mean = restored.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
        restored_std = restored.std(dim=(-2, -1), unbiased=False)
        target_std = target.std(dim=(-2, -1), unbiased=False)
        return F.l1_loss(restored_mean, target_mean) + F.l1_loss(
            restored_std, target_std
        )


class GradientStructureLoss(nn.Module):
    def __init__(self):
        super(GradientStructureLoss, self).__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def _gradient(self, image):
        channels = image.shape[1]
        kernel_x = self.kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def forward(self, restored, target):
        return F.l1_loss(self._gradient(restored), self._gradient(target))


class EdgeIntensityLoss(GradientStructureLoss):
    def forward(self, restored, target):
        restored_grad = self._gradient(restored)
        target_grad = self._gradient(target)
        restored_mean = restored_grad.mean(dim=(-2, -1))
        target_mean = target_grad.mean(dim=(-2, -1))
        return F.l1_loss(restored_mean, target_mean)


class LocalContrastLoss(nn.Module):
    def __init__(self, kernel_size=9, eps=1e-6):
        super(LocalContrastLoss, self).__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def _local_std(self, image):
        image = image.clamp(0.0, 1.0)
        padding = self.kernel_size // 2
        mean = F.avg_pool2d(
            image, self.kernel_size, stride=1, padding=padding, count_include_pad=False
        )
        mean_sq = F.avg_pool2d(
            image * image,
            self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        return torch.sqrt((mean_sq - mean * mean).clamp_min(0.0) + self.eps)

    def forward(self, restored, target):
        return F.l1_loss(self._local_std(restored), self._local_std(target))


class HighFrequencyLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super(HighFrequencyLoss, self).__init__()
        self.blur = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def _high_freq(self, image):
        image = image.clamp(0.0, 1.0)
        return image - self.blur(image)

    def forward(self, restored, target):
        return F.l1_loss(self._high_freq(restored), self._high_freq(target))
