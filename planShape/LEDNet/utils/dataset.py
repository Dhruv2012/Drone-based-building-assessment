import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

EXTENSIONS = ['.jpg', '.png', '.JPG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, transform=None, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_path = image_path(self.images_root, filename, '.jpg')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.JPG')
        
        with open(img_path, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        
        # print('image shape:', image)
        # print('label shape:', label)

        if self.transform is not None:
            image, label = self.transform(image, label)
        # if self.input_transform is not None:
        #     image = self.input_transform(image)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')       
        #self.labels_root = os.path.join(root, 'gtCoarse/') 
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        valid_sets = ["train", "trainval", "val"]
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        voc_root = os.path.join(self.root, self.image_set)
        self.images_root = os.path.join(voc_root, 'images')
        self.labels_root = os.path.join(voc_root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # img = Image.open(self.images[index]).convert('RGB')
        # target = Image.open(self.masks[index])

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # return img, target
        filename = self.filenames[index]

        img_path = image_path(self.images_root, filename, '.jpg')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.JPG')
        
        with open(img_path, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        
        # print('image shape:', image)
        # print('label shape:', label)

        if self.transforms is not None:
            image, label = self.transforms(image, label)
        # if self.input_transform is not None:
        #     image = self.input_transform(image)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.filenames)