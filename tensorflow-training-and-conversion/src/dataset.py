from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import normalize


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# x_train = x_train / 255
# x_test = x_test / 255


data_loaders = {
    "train": [x_train, y_train],
    "test": [x_test, y_test],
}
