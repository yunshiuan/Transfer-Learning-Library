"""
@author: Yun-Shiuan Chuang
@modified based on: office31.py
"""
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31_v2(ImageList):
    """Office31_v2 Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, it requires the existence of the following directories and files. Note that the images can be downloaded using `office31.py`.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon_train.txt
                amazon_val.txt
                amazon_test.txt
                dslr_train.txt
                dslr_val.txt
                dslr_test.txt
                webcam_train.txt
                webcam_val.txt
                webcam_test.txt
    """
    image_list = {
        "A_train": "image_list/amazon_train.txt",
        "A_val": "image_list/amazon_val.txt",
        "A_test": "image_list/amazon.txt",

        "D_train": "image_list/dslr_train.txt",
        "D_val": "image_list/dslr_val.txt",
        "D_test": "image_list/dslr_test.txt",

        "W_train": "image_list/webcam_train.txt",
        "W_val": "image_list/webcam_val.txt",
        "W_test": "image_list/webcam_test.txt"        
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(Office31_v2, self).__init__(root, Office31_v2.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())