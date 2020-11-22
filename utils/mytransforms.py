import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import random


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, dataset='cifar'):
        if dataset.startswith('cifar'):
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            resos = [32, 28, 24]
        elif dataset.startswith('imagenet'):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            resos = [224, 192, 160, 128]
        else:
            raise NotImplemented('dataset not implemented.')
        self.reso_idx = [0, 1, 2]
        self.base_transform = base_transform
        self.fullnet_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.subnet_transforms = []
        for i in range(len(resos)):
            self.subnet_transforms.append(
                transforms.Compose([
                    transforms.Resize((resos[i], resos[i])),
                    transforms.ToTensor(),
                    normalize
                ])
            )

    def __call__(self, x):
        output_list = []
        output_list.append(self.fullnet_transform(self.base_transform(x)))
        for idx in self.reso_idx:
            # t = random.randint(0, len(self.subnet_transforms)-1)
            # output_list.append(self.subnet_transforms[random.randint(0, 3)](self.base_transform(x)))
            output_list.append(self.subnet_transforms[idx](self.base_transform(x)))
        return output_list

    def set_resoidx(self, reso_idx):
        self.reso_idx = reso_idx