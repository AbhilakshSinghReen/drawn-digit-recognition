import requests

url = "http://localhost:8000/api/run-inference"

image_file_path = "1.png"

with open(image_file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

if response.status_code == 200:
    print("Predicted Label:", response.json()["predicted_label"])
else:
    print("Error:", response.text)
