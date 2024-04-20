from os.path import dirname, join as path_join
from sys import path

path.insert(0, path_join(dirname(dirname(__file__)), "src"))

from os import listdir
from json import load as json_load
import unittest

import requests

from config import data_dir


class TestAPIInference(unittest.TestCase):
    def setUp(self):
        self.url = "http://localhost:8000/api/run-inference"
        self.images_dir = path_join(data_dir, "test_images")
        self.labels_json_file_path = path_join(data_dir, "test_images_labels.json")

        with open(self.labels_json_file_path, 'r') as labels_json_file:
            self.labels = json_load(labels_json_file)

    def test_inference(self):
        num_correct_predictions = 0
        num_bad_responses = 0
        all_image_names = listdir(self.images_dir)

        for image_name in all_image_names:
            image_file_path = path_join(self.images_dir, image_name)

            with open(image_file_path, "rb") as file:
                response = requests.post(self.url, files={"file": file})

            if response.status_code != 200:
                num_bad_responses += 1

            self.assertEqual(response.status_code, 200)

            response_data = response.json()

            if response_data['predicted_label'] == self.labels[image_name]:
                num_correct_predictions += 1

            print(
                f"Image: {image_name}, "
                f"Predicted Label: {response_data['predicted_label']}, "
                f"Correct: {response_data['predicted_label'] == self.labels[image_name]}"
            )

        num_inferred = len(all_image_names) - num_bad_responses
        accuracy = num_correct_predictions / num_inferred

        print(f"Failed to infer: {num_bad_responses} / {len(all_image_names)}")
        print(f"Correct Predictions: {num_correct_predictions} / {num_inferred}, Accuracy: {accuracy}")


if __name__ == '__main__':
    unittest.main()
