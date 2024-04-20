from os.path import join as path_join

from keras.models import load_model
import onnx
from tensorflowjs import converters
import tf2onnx

from ..config import models_dir


MODEL_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number.model")
ONNX_MODEL_FILE_PATH = MODEL_FILE_PATH[:-6] + ".onnx"


if __name__ == "__main__":
    keras_model = load_model(MODEL_FILE_PATH)
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model)
    
    onnx.save(onnx_model, ONNX_MODEL_FILE_PATH)
    print(f"ONNX model saved at: {ONNX_MODEL_FILE_PATH}")
