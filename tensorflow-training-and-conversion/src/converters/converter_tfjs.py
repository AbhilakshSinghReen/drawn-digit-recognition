from os.path import join as path_join

from keras.models import load_model
from tensorflowjs import converters

from ..config import models_dir


MODEL_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number.model")
TFJS_MODEL_FILE_PATH = MODEL_FILE_PATH[:-6] + "-tfjs"


if __name__ == "__main__":
    model = load_model(MODEL_FILE_PATH)
    converters.save_keras_model(model, TFJS_MODEL_FILE_PATH)
    print(f"TensorFlow JS model saved at: {TFJS_MODEL_FILE_PATH}")
