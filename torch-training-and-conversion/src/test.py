from json import load as json_load
from os import listdir
from os.path import join as path_join
from time import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import data_dir, models_dir
from .dataset import data_loaders
from .model import CNN


# MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_numbe")
MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "torch---2024-04-23-08-25-15", "epoch-9.pt")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model = model.to(device)

    model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE_PATH, map_location=device))

    model.eval()

    preprocessing_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Testing on MNIST Test Set")

    num_correct = 0
    with torch.no_grad():
        for data, target in data_loaders["test"]:
            data = data.to(device)
            target = target.to(device)

            if num_correct == 0:
                print(type(data[0]))
                print(data[0].dtype)
                print(data[0].shape)
                print(torch.min(data[0]))
                print(torch.max(data[0]))

            output = model(data)

            prediction = output.argmax(dim=1, keepdim=True)

            num_correct += prediction.eq(target.view_as(prediction)).sum().item()
    
    accuracy = num_correct / len(data_loaders['test'].dataset)
    print(f"    Accuracy: {accuracy}")
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
        # test_images[i] = np.array([test_images[i]])
        # test_images[i] = normalize(test_images[i], axis=1)
        test_images[i] = preprocessing_transforms(test_images[i])
        test_images[i] = test_images[i].unsqueeze(0)

    print(type(test_images[0]))
    print(test_images[0].dtype)
    print(test_images[0].shape)
    print(torch.min(test_images[0]))
    print(torch.max(test_images[0]))

    # Predict
    num_correct = 0
    for preprocessed_image, label in zip(test_images, test_labels):
        output = model(preprocessed_image)
        prediction = output.argmax(dim=1, keepdim=True)
        prediction = int(prediction)

        num_correct += prediction == label
    
    accuracy = num_correct / len(test_images)
    print(f"Correct: {num_correct} / {len(test_images)}, Accuracy: {accuracy}")
