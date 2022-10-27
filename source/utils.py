import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.util import random_noise
from sodaflow import tracking
from tensorflow.keras.callbacks import ModelCheckpoint

from source import config

##Preprocessing


class CustomModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        print(
            f"-----> End epoch ({epoch}). monitor: {self.monitor}, current: {current}, log_values: {logs}"
        )
        tracking.log_metrics(step=epoch, **logs)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                # current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            tracking.log_model(filepath)
                        else:
                            self.model.save(filepath, overwrite=True)
                            tracking.log_model(filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    tracking.log_model(filepath)
                else:
                    self.model.save(filepath, overwrite=True)
                    tracking.log_model(filepath)


"Apply same preprocessing for every model"


def show_im(img):
    """simply show image"""
    plt.figure(figsize=(10, 10))
    img = plt.imread(img)
    plt.imshow(img)
    plt.show()


def remove_padding(image, threshold=0):
    """"""
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def crop_or_pad(image, size=(224, 224)):
    """standard image-preprocessing"""
    image = remove_padding(image)
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    image_max = max(h, w)
    scale = float(min(size)) / image_max

    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    h, w = image.shape[:2]
    top_pad = (size[1] - h) // 2
    bottom_pad = size[1] - h - top_pad
    left_pad = (size[0] - w) // 2
    right_pad = size[0] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode="constant", constant_values=0)

    # Fix image normalization
    if np.nanmax(image) > 1:
        image = np.divide(image, 255.0)

    return image


def read_img(image, is_train=True):
    """
    input image: image file name,
    is_train: boolean,
    return single preprocessed (remove_padding & crop_or_pad) image array
    if is_train: apply sklearn.image.random_noise
    """
    image = crop_or_pad(image)
    noise_mode = np.random.choice(
        ["gaussian", "speckle", "s&p", "None"]
    )  ##randomly pick augmentation type

    if (noise_mode == "None") or not is_train:
        noised_image = image

    else:
        noised_image = random_noise(image, noise_mode)

    return noised_image


def _rotate_image(mat, angle, padColor=0):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # set pad color
    if len(mat.shape) == 3 and not isinstance(
        padColor, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        padColor = [padColor] * 3

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderValue=padColor
    )
    return rotated_mat


@tf.function
def number():
    noise = tf.random.uniform(shape=(), minval=0, maxval=4)
    return int(noise)


def rotate_img(image, is_train=False):

    if is_train:
        rot_label = number()
        angle = int(rot_label * 90)
        rotated_img = _rotate_image(image, angle, padColor=0)
    else:
        rotated_img = image
        rot_label = 0

    return rotated_img, np.eye(4)[rot_label]


def prepare_data(input_string):
    """
    return preprocessed image array or list of arrays.
    """
    model_config = config.model_info()

    if os.path.isfile(input_string):
        img = plt.imread(input_string)
        img = crop_or_pad(img, model_config["INPUT_SIZE"][:2])
        img = np.expand_dims(img, 0)
        return img

    if os.path.isdir(input_string):

        ####Recursive
        img_array = []
        for filename in Path(input_string).rglob("*.jpg"):
            img = plt.imread(filename)
            img = crop_or_pad(img, model_config["INPUT_SIZE"][:2])
            img = np.expand_dims(img, 0)
            img_array.append(img)
        return img_array

    else:
        raise ValueError("Invalid Input. Please enter valid file name.")


# Load DataFrame
def read_data_df(data_files):
    data = pd.DataFrame(
        {"image": data_files, "class": [file.split("/")[-2] for file in data_files]}
    )
    data["label"] = pd.get_dummies(data["class"]).values.tolist()

    # shuffle every epoch
    data = data.reindex(np.random.permutation(data.index))  # Shuffle and reindex
    data.index = [i for i in range(len(data))]

    return data


def to_df(file_dir, is_train):
    """return dataframe based on file structure; 'data/train_or_val/classname/filename.jpg'
    {'image': filename, 'class': classname, 'label': one hot encoding}
    """
    if is_train:
        piece = "train"
    else:
        piece = "eval"
    files = glob(os.path.join(file_dir, piece, "*/*.*"))
    data = pd.DataFrame(
        {"image": files, "class": [file.split("/")[-2] for file in files]}
    )

    data["label"] = pd.get_dummies(data["class"]).values.tolist()

    # shuffle every epoch
    data = data.reindex(np.random.permutation(data.index))  # Shuffle and reindex
    data.index = [i for i in range(len(data))]

    return data


# Evaluation


def plot_cm(gt, prediction, ticks, title, save_dir):

    from sklearn.metrics import confusion_matrix, f1_score

    cm = confusion_matrix(gt, prediction)

    f1score = f1_score(gt, prediction, average="weighted")

    """return confusion matrix plot."""
    import itertools

    import matplotlib.pyplot as plt

    PLT_SIZE = 10 + int(20.0 / float(len(ticks)))

    plt.rc("font", size=PLT_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=PLT_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=PLT_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=PLT_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=PLT_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=PLT_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=PLT_SIZE)  # fontsize of the figure title

    keep_cm = cm
    cm_sum = cm.sum(axis=1)[:, np.newaxis]

    cm = cm.astype("float") / cm_sum
    acc = np.sum(cm.diagonal()) / np.sum(cm)
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    # plt.title(title + " | {:.2f}%".format(acc * 100))
    plt.title(title + " | F1 score : {:.4f}".format(f1score))

    plt.xticks(np.arange(len(ticks)), ticks, rotation=0)
    plt.yticks(np.arange(len(ticks)), ticks)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(
            j,
            i - 0.1,
            format(cm[i, j], ".2f"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

        plt.text(
            j,
            i + 0.1,
            format(keep_cm[i, j], "d") + " / " + str(cm_sum[i][0]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("true")
    plt.xlabel("predicted")
    if save_dir:
        plt.savefig(os.path.join(save_dir, title))
    plt.show()
    plt.close()

    return acc, f1score
