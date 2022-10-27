def model_info():

    return {
        "MODEL_NAME": "DenseNet169",
        "MODEL_DIR": "./tmp_model",
        "INPUT_SIZE": (224, 224, 3),
        "NUM_CLASSES": 6,
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 2,
        "LEARNING_RATE": 1e-5,
        "USING_GPU": "0",
        "TARGET_WEIGHT": "./logs/",
        "RESULT_SAVE_DIR": "./result",
        "GPU_MEMORY_FRACTION": 1.0,
    }


def print_info():
    print(model_info())
