import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch
cv2.ocl.setUseOpenCL(False)


class Compose(object):
    """
    Composes several segsegtransforms together.

    Args:
        segtransforms (List[Transform]): list of segtransforms to compose.

    Example:
        segtransforms.Compose([
            segtransforms.RandScale([0.5, 2.0]),
            segtransforms.ToTensor()])
    """
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label, label1):
        for t in self.segtransforms:
            image, label, label1 = t(image, label, label1)
        return image, label, label1


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label, label1):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        label1 = torch.from_numpy(label1)
        if not isinstance(label1, torch.LongTensor):
            label1 = label1.long()
        return image, label, label1


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label, label1):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label, label1


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    'size' is a 2-element tuple or list in the order of (h, w)
    """
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label, label1):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        label1 = cv2.resize(label1, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label, label1





class RandRotate(object):
    """
    Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    """
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransforms.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label, label1):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
            label1 = cv2.warpAffine(label1, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label, label1


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, label1):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            label1 = cv2.flip(label1, 1)
        return image, label, label1



class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label, label1):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label, label1

