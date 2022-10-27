import random

import numpy as np

# import keras
from keras.utils import all_utils
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from source.utils import read_img


class DataGenerator(all_utils.Sequence):
    """
    Data Generator yielding #batch_size of (img_array(224,224,3), img_label
    e.g ) np.array(shape of (224,224,3)), np.array([0,0,0,0,1,0])
    """

    def __init__(self, data, config, is_train):
        self.data = data
        self.config = config
        self.batch_size = config["BATCH_SIZE"]
        self.n_classes = config["NUM_CLASSES"]
        self.dim = config["INPUT_SIZE"]
        self.is_train = is_train
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))

    #         if self.shuffle:
    #             np.random.shuffle(self.indexes)
    # shuffle not necessary b/c shuffle on the level of data formation and arrangement
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size)) + 1

    def __data_generation(self, X_list, y_list):
        """Generates data containing batch_size samples"""
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=np.int8)

        for i, (img, label) in enumerate(zip(X_list, y_list)):

            try:
                # img = plt.imread(img)[:,:,:3]
                pil_img = Image.open(img)
                np_img = np.asarray(pil_img)[:, :, :3]
            except:
                print(img)

            X[i] = read_img(np_img, self.is_train)
            y[i] = label

        return X, y

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.is_train:
            cls_list = list(self.data["class"].unique())

            pick = random.sample(cls_list, self.batch_size % len(cls_list))
            ls = []
            # randomly pick index from each class -> pick equal or similar number of files
            for cls in cls_list:
                trg = self.data[self.data["class"] == cls]
                sample_size = self.batch_size // len(cls_list)
                if cls in pick:
                    sample_size += 1  # number of files in each class has only 1 difference at maximum
                ls.extend(list(trg.sample(sample_size).index))

            random.shuffle(ls)
            assert len(ls) == self.batch_size

            indexes = self.indexes[ls]
        else:
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]

        X_list = [self.data["image"][k] for k in indexes]
        y_list = [self.data["label"][k] for k in indexes]

        X, y = self.__data_generation(X_list, y_list)

        return X, y
