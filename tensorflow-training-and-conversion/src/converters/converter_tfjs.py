from os.path import dirname, join as path_join

from tensorflowjs import converters

from ..config import models_dir
from ..model import CNN


MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "tensorflow---2024-04-21-22-04-45", "epoch-9.h5")
TFJS_MODEL_FILE_PATH = path_join(dirname(MODEL_WEIGHTS_FILE_PATH), "model-tfjs")


if __name__ == "__main__":
    keras_model = CNN()
    keras_model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    converters.save_keras_model(keras_model, TFJS_MODEL_FILE_PATH)
    print(f"TensorFlow JS model saved at: {TFJS_MODEL_FILE_PATH}")
