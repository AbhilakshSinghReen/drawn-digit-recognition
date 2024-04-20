import requests
import os

url = "http://localhost:8000/api/run-inference"

image_file_path = "1.png"

for image_file_path in os.listdir(os.path.dirname(__file__)):
    if not image_file_path.endswith(".png"):
        continue

    with open(image_file_path, "rb") as file:
        response = requests.post(url, files={"file": file})

    if response.status_code == 200:
        print(f"Predicted Label: {image_file_path}, ", response.json()["predicted_label"])
    else:
        print("Error:", response.text)
