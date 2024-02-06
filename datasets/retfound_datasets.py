import os

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)


def build_dataset(data_path, is_train, is_test):
    if is_train is True:
        split_set = "train"
    elif is_test is True:
        split_set = "test"
    else:
        split_set = "val"
    root = os.path.join(data_path, split_set)
    dataset = datasets.ImageFolder(root)
    return dataset


def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train is True:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,  # 224
            is_training=True,
            color_jitter=None,  # None
            auto_augment="rand-m9-mstd0.5-inc1",  # 'rand-m9-mstd0.5-inc1'
            interpolation="bicubic",
            re_prob=0.25,  # 0.25
            re_mode="pixel",  # 'pixel'
            re_count=1,  # 1
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    input_size = 224
    crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
