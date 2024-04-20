from json import load as json_load
from os import listdir
from os.path import join as path_join

import cv2
import numpy as np

from .config import models_dir, data_dir
from .dataset import data_loaders
from .model import CNN

from tensorflow.keras.utils import normalize


MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number")


if __name__ == "__main__":
    model = CNN()
    model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    ### Test on MNIST Test Data
    x_test, y_test = data_loaders['test']
    loss, accuracy = model.evaluate(x_test, y_test)

    print("Testing on MNIST Test Set")
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    print()

    ### Test on Custom Images
    # Load Images
    test_images_dir = path_join(data_dir, "test_images")
    
    test_images_labels_file_path = path_join(data_dir, "test_images_labels.json")
    with open(test_images_labels_file_path, 'r') as test_images_labels_file:
        test_images_labels = json_load(test_images_labels_file)

    test_images = []
    test_labels = []
    for image_name in listdir(test_images_dir):
        image_path = path_join(test_images_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        test_images.append(image)

        test_labels.append(test_images_labels[image_name])

    # Preprocess
    for i in range(len(test_images)):
        test_images[i] = np.array([test_images[i]])
        test_images[i] = normalize(test_images[i], axis=1)
    
    # Predict
    num_correct = 0
    for image, label in zip(test_images, test_labels):
        prediction = model.predict(image, verbose=None)
        predicted_label = int(np.argmax(prediction))

        # print(f"{predicted_label} : {label}")
        num_correct += predicted_label == label
    
    accuracy = num_correct / len(test_images)
    print(f"Correct: {num_correct} / {len(test_images)}, Accuracy: {accuracy}")
