import os
from glob import glob
from pprint import pprint

import numpy as np
import sodaflow
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

from source import utils
from source.data_generator import DataGenerator
from source.model import ModelLib
from source.utils import CustomModelCheckpoint

USING_GPU = "0"


@sodaflow.main(config_path="configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    train_config = OmegaConf.to_container(cfg)
    data_dirs = train_config["sodaflow"]["dataset_path"]
    init_ckpts_dir = train_config["sodaflow"]["start_ckpt_path"]
    pprint(train_config)

    # Data
    train_file_list = list()
    valid_file_list = list()

    for ds_path in data_dirs:
        print(f"Read datasets from {ds_path}")
        train_file_list.extend(glob(os.path.join(ds_path, "train", "*/*.*")))
        valid_file_list.extend(glob(os.path.join(ds_path, "val", "*/*.*")))

    train_data = utils.read_data_df(train_file_list)
    val_data = utils.read_data_df(valid_file_list)

    # Generator
    train_generator = DataGenerator(train_data, train_config["train_inputs"], True)
    val_generator = DataGenerator(val_data, train_config["train_inputs"], False)

    # Callback options
    ckpt_list = glob(os.path.join(init_ckpts_dir, "*.h5"))
    model = ModelLib(train_config["train_inputs"])
    model = model.build_model()
    if len(ckpt_list) > 0:
        ckpt_path = ckpt_list[-1]
        print(f"Load initial model checkpoint from {ckpt_path}")
        model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
        init_epoch = 0
    else:
        [
            os.remove(x)
            for x in glob(os.path.join(train_config["MODEL_OUTPUT_DIR"], "*tfevents*"))
        ]
        init_epoch = 0

    model_output_path = os.path.join(
        train_config["MODEL_OUTPUT_DIR"],
        "filter-epoch-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}.h5",
    )

    if not os.path.exists(train_config["MODEL_OUTPUT_DIR"]):
        os.makedirs(train_config["MODEL_OUTPUT_DIR"], mode=511, exist_ok=True)

    # Checkpoint file format, option setter
    ckpt = CustomModelCheckpoint(
        model_output_path,
        monitor="val_accuracy",
        save_weights_only=False,
        save_best_only=True,
        verbose=1,
    )

    # Learning Rate modifier
    lrplateau = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=10, verbose=1
    )
    callbacks_list = [ckpt, lrplateau]
    steps_per_epoch = train_data.shape[0] // train_config["train_inputs"]["BATCH_SIZE"]
    val_step = val_data.shape[0] // train_config["train_inputs"]["BATCH_SIZE"]

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=train_config["train_inputs"]["NUM_EPOCHS"],
        callbacks=callbacks_list,
        initial_epoch=init_epoch,
        verbose=1,
        validation_data=val_generator,
        validation_steps=val_step,
    )

    # run test
    print("Run Evaluation...")

    true = np.array([], dtype=int)
    pred = np.array([], dtype=int)
    num_batch = val_data.shape[0] // val_generator.batch_size
    remain = val_data.shape[0] % val_generator.batch_size
    for idx in tqdm(range(num_batch + 1)):
        img, label = val_generator.__getitem__(idx)
        if idx == num_batch:
            if remain != 0:
                img = img[:remain]
                label = label[:remain]
            else:
                continue
        output = model.predict(img)
        true = np.concatenate((true, np.argmax(label, axis=1)))
        pred = np.concatenate((pred, np.argmax(output, axis=1)))

    tmp_dict = {
        "filepath": val_data["image"],
        "filename": val_data["image"].apply(lambda x: os.path.basename(x)),
        "ground_truth": true,
        "prediction": pred,
    }

    cm = confusion_matrix(true, pred)
    acc = np.sum(cm.diagonal()) / np.sum(cm)

    best_eval_file = os.path.join(train_config["LOGS_DIR"], "best_eval.txt")
    if os.path.isfile(best_eval_file):
        f = open(best_eval_file, "w")
        f.write(str(acc))
        f.close()


if __name__ == "__main__":
    run_app()
