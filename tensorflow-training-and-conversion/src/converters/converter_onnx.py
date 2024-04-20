from os.path import dirname, join as path_join

import onnx
import tf2onnx

from ..config import models_dir
from ..model import CNN


MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number")
ONNX_MODEL_FILE_PATH = path_join(dirname(MODEL_WEIGHTS_FILE_PATH), "model.onnx")


if __name__ == "__main__":
    keras_model = CNN()
    keras_model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    onnx_model, _ = tf2onnx.convert.from_keras(keras_model)
    
    onnx.save(onnx_model, ONNX_MODEL_FILE_PATH)
    print(f"ONNX model saved at: {ONNX_MODEL_FILE_PATH}")
