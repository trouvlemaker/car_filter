import os
import sys

APP_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_BASE_PATH)
sys.path.append("/opt/tritonserver/backends/python")

from glob import glob

import numpy as np
import triton_python_backend_utils as pb_utils
from sodaflow import api as soda_api
from tensorflow.keras.models import load_model

from source import config, utils


class TritonPythonModel(soda_api.SodaPythonModel):
    def load_model(self, args):
        # Load Model
        if os.environ.get("TEST"):
            print("--------------------> TEST env")
            ckpt_path = "./export/train_best.h5"
        else:
            ckpt_path = glob(os.path.join(self.weight_path, "*.h5"))[0]
            print(ckpt_path)

        self.config = config.model_info()
        self.model = load_model(ckpt_path)

        print("Restore model weight from %s" % ckpt_path)

    def build_model_signature(self):
        self.batch_size = 10
        self.add_input("INPUT_IMAGE", 4, "TYPE_FP32")
        self.add_output("OUTPUT_SOFTMAX", 3, "TYPE_FP32")
        self.add_output("OUTPUT_CLASS", 2, "TYPE_INT32")

    def execute(self, requests):
        responses = []
        for request in requests:
            data_np = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            data_np = data_np.as_numpy()

            result_softmax = []
            result_batch = []

            for dp in data_np:
                dp = utils.crop_or_pad(dp, self.config["INPUT_SIZE"][:2])
                dp = np.expand_dims(dp, 0)

                softmax = self.model.predict(dp)
                predict_class = np.argmax(softmax)

                result_softmax.append(softmax)
                result_batch.append(predict_class)

            out_tensor_0 = pb_utils.Tensor("OUTPUT_SOFTMAX", np.array(result_softmax))
            out_tensor_1 = pb_utils.Tensor("OUTPUT_CLASS", np.array(result_batch))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses


if __name__ == "__main__":
    # For Testing
    print("For test export python model.")
    args = {"model_name": "twincar-filter-v1", "output_path": "export/pymodels"}

    model = TritonPythonModel(
        model_name=args["model_name"], output_path=args["output_path"]
    )

    model.build_config_pbtxt()
