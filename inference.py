import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from keras.models import load_model

from source import config, utils

arg = argparse.ArgumentParser(description="inference: file or directory. ")
arg.add_argument("--inf", dest="inf", help="inference file or files in directory")
args = arg.parse_args()

model_config = config.model_info()


def inference(file=args.inf):
    """return DataFrame of {filename, softmax, prediction} for single or multiple files"""

    latest_model = sorted(
        glob(os.path.join(model_config["MODEL_DIR"], "*.h5")), key=os.path.getctime
    )[
        -1
    ]  # load latest weight

    model = load_model(latest_model)

    ###input file or directory

    if os.path.isdir(file):

        softmax = [
            np.round(model.predict(utils.prepare_data(i)), 3)
            for i in Path(args.inf).rglob("*.jpg")
        ]

        prediction = [np.argmax(i) for i in softmax]

        return pd.DataFrame(
            {
                "filename": [i for i in Path(file).rglob("*.jpg")],
                "softmax": softmax,
                "prediction": prediction,
            }
        )

    if os.path.isfile(file):

        softmax = np.round(model.predict(utils.prepare_data(file)), 3)

        prediction = np.argmax(softmax)

        return pd.DataFrame(
            {"filename": file, "softmax": [softmax], "prediction": prediction}
        )
        # return softmax and max score


if __name__ == "__main__":

    result = inference()
    # print(result)
    result.to_json(
        path_or_buf="./test_output.json", orient="table", default_handler=str
    )
