import json
import os
import warnings
from glob import glob

import sodaflow

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.models import load_model
from omegaconf import DictConfig
from sodaflow import tracking
from tqdm import tqdm

from source import utils
from source.data_generator import DataGenerator

gpu_options = tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=1.0, allow_growth=True
)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)


def save_json(config, config_path):
    """
    dict 형태의 데이터를 받아 json파일로 저장해주는 함수
    """
    with open(config_path, "w", encoding="utf-8-sig") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


@sodaflow.main(config_path="./configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    """return confusion matrix plot in save_dir"""
    save_dir = cfg.EVAL_OUT
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = glob(os.path.join(cfg.sodaflow.dataset_path[0], "test/*/*"))
    print("file count:", len(files))
    val_data = pd.DataFrame(
        {"image": files, "class": [file.split("/")[-2] for file in files]}
    )
    val_data["label"] = pd.get_dummies(val_data["class"]).values.tolist()
    ticks = sorted(val_data["class"].unique().tolist())

    latest_model = sorted(
        glob(os.path.join(cfg.sodaflow.test_model_path, "*.h5")), key=os.path.getctime
    )[-1]
    model = load_model(latest_model)
    print("model loaded")

    gt = [np.argmax(lb) for lb in val_data["label"]]
    val_generator = DataGenerator(val_data, cfg.train_inputs, False)
    predictions = model.predict(
        val_generator,
        verbose=1,
        max_queue_size=50,
        workers=40,
        use_multiprocessing=True,
    )

    predictions = np.argmax(predictions, 1)[: len(files)]

    plot_title = "Twincar_Filter_model_CM.jpg"
    acc, f1score = utils.plot_cm(
        gt,
        predictions,
        title=plot_title,
        ticks=ticks,
        save_dir=cfg.EVAL_OUT,
    )
    plot_save_path = os.path.join(cfg.EVAL_OUT, plot_title)
    tracking.log_model(plot_save_path)
    tracking.log_outputs(
        accuracy="{:0.3f}".format(acc),
        f1score="{:3.3f}".format(f1score),
    )
    print(f"Accuracy => {acc:0.3f}, F1score => {f1score:0.3f}")
    # tracking.log_metric("Accuracy", acc)
    # tracking.log_metric("F1 score", f1score)
    output_dict = {}

    for idx, datas in tqdm(enumerate(zip(val_data["image"], predictions, gt))):
        img_path, result, g = datas
        basename = os.path.basename(img_path)
        output_dict[idx] = {
            "filename": basename,
            "predict": int(result),
            "ground_truth": int(g),
        }

    save_json_path = os.path.join(cfg.EVAL_OUT, "predict_result.json")
    save_json(output_dict, save_json_path)
    tracking.log_model(save_json_path)


if __name__ == "__main__":
    run_app()
