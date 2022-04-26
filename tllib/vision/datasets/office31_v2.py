"""
@author: Yun-Shiuan Chuang
@modified based on: office31.py
"""
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
        output_transform (callable, optional): A function/transform that takes in the output and transforms it.

    .. note:: In `root`, it requires the existence of the following directories and files. Note that the images can be downloaded using `office31.py`.
        ::
            office31/
                amazon/
                    images/
                        backpack/
                            *.jpg
                            ...
                dslr/
                webcam/
            office31_v2/
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
    DOMAINS = [
        "amazon",
        "dslr",
        "webcam"
    ]
    IMAGE_LIST_INPUT = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt"
    }
    IMAGE_LIST_OUTPUT = {
        "A_train": "image_list/amazon_train.txt",
        "A_val": "image_list/amazon_val.txt",
        "A_test": "image_list/amazon_test.txt",

        "D_train": "image_list/dslr_train.txt",
        "D_val": "image_list/dslr_val.txt",
        "D_test": "image_list/dslr_test.txt",

        "W_train": "image_list/webcam_train.txt",
        "W_val": "image_list/webcam_val.txt",
        "W_test": "image_list/webcam_test.txt"
    }
    LIST_PARTITION = ["train", "val", "test"]
    WEIGHT_PARTITION = [0.6, 0.2, 0.2]

    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    PATH_IMAGES_INPUT = "data/office31"
    PATH_IMAGES_OUTPUT = "data/office31_v2"

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.IMAGE_LIST_OUTPUT
        data_list_file = os.path.join(root, self.IMAGE_LIST_OUTPUT[task])
        # check and prepare the files
        self._process(root)

        super(Office31_v2, self).__init__(root, Office31_v2.CLASSES,
                                          data_list_file=data_list_file, **kwargs)

    def _process(self, root):
        # check if the image files exist
        for domain in self.DOMAINS:
            path_domain = os.path.join(self.PATH_IMAGES_INPUT, domain)
            # raise an error if the image files not yet exists
            if not os.path.isdir(path_domain):
                raise Exception(
                    "Should call `office31.py` to download the image files first.")

        # check if the image list txts exist
        for file in self.IMAGE_LIST_OUTPUT.values():
            # geneate the image list txts if not yet exist
            if not os.path.isfile(os.path.join(root, file)):
                self._generate_image_list()

    def _generate_image_list(self):
        # check if the input image list txts exist
        for file in self.IMAGE_LIST_INPUT.values():
            if not os.path.isfile(os.path.join(self.PATH_IMAGES_INPUT, file)):
                # raise an error if the input image files not yet exists
                raise Exception(
                    "Should call `office31.py` to download the image list files first.")
        # ------------------
        # generate the image lists
        # ------------------
        for input_domain in self.IMAGE_LIST_INPUT:
            # input file
            file_image_list_input = os.path.join(
                self.PATH_IMAGES_INPUT, self.IMAGE_LIST_INPUT[input_domain])

            # read the input file
            df_image_list_input = pd.read_csv(
                file_image_list_input, names=['raw'])
            # - split the lines into two columsn: 'file' and 'label'
            df_image_list_input = df_image_list_input.raw.str.split(
                expand=True)
            df_image_list_input.columns = ['file', 'label']

            # set the file dir to "../office31"
            df_image_list_input.file = os.path.join(
                '..', 'office31')+os.sep+df_image_list_input.file
            # ------------------
            # paritition the files into 'train/val/test' based on the given weights
            # - note that the partition should be done within each class to ensure the disribution of class is the same across the three partitions
            # ------------------
            # paritition into train vs. (val+test)
            file_train, file_test, label_train, label_test =\
                train_test_split(df_image_list_input['file'], df_image_list_input['label'],
                                 stratify=df_image_list_input['label'], random_state=1, test_size=self.WEIGHT_PARTITION[1]+self.WEIGHT_PARTITION[2])
            # paritition into val vs. test
            file_val, file_test, label_val, label_test =\
                train_test_split(file_test, label_test,
                                 stratify=label_test, random_state=1, test_size=(self.WEIGHT_PARTITION[2])/sum(self.WEIGHT_PARTITION[1:]))
            df_image_list_train = pd.DataFrame(
                {'file': file_train, 'label': label_train})
            df_image_list_val = pd.DataFrame(
                {'file': file_val, 'label': label_val})
            df_image_list_test = pd.DataFrame(
                {'file': file_test, 'label': label_test})

            # ------------------
            # output file
            # ------------------
            dict_df_image_list =\
                {"train": df_image_list_train, "val": df_image_list_val,
                    "test": df_image_list_test}
            path_output = os.path.join(self.PATH_IMAGES_OUTPUT,'image_list')
            if not os.path.exists(path_output):
                os.makedirs(path_output)

            for partition in self.LIST_PARTITION:
                file_image_list_output = os.path.join(
                    self.PATH_IMAGES_OUTPUT, self.IMAGE_LIST_OUTPUT[input_domain+"_"+partition])
                dict_df_image_list[partition].sort_values(
                    by=['label'], inplace=True)
                # dict_df_image_list[partition].to_csv(file_image_list_output)
                dict_df_image_list[partition].to_csv(
                    file_image_list_output, header=None, index=None, sep=' ', mode='a')

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
