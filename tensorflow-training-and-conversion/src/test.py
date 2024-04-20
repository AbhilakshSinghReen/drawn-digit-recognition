from os.path import join as path_join

from .config import models_dir
from .dataset import data_loaders
from .model import CNN


MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number")


if __name__ == "__main__":
    model = CNN()
    model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    x_test, y_test = data_loaders['test']
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
