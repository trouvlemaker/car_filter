# coding: utf-8

from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class ModelLib:
    """
    return Classification Model based on the source.config.model_info()
    assigning model by source.config['MODEL_NAME']
    """

    def __init__(self, model_config):
        self.config = model_config
        self.base_model = None

    def build_model(self):
        network = {
            "DenseNet121": DenseNet121,
            "DenseNet169": DenseNet169,
            "MobileNetV2": MobileNetV2,
            "ResNet50": ResNet50,
        }

        self.base_model = network[self.config["MODEL_NAME"]](
            include_top=False,
            weights=None,
            #                                weights='imagenet',
            pooling="avg",
            input_shape=self.config["INPUT_SIZE"],
        )

        base_model = self.base_model

        base_model.trainable = True

        model = Sequential()
        model.add(base_model)
        model.add(Dropout(0.3))  # Standardize weights
        model.add(
            Dense(self.config["NUM_CLASSES"], activation="softmax", name="output")
        )

        adam = Adam(learning_rate=self.config["LEARNING_RATE"])

        # Compile
        model.compile(
            optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Summary
        model.summary()

        return model
