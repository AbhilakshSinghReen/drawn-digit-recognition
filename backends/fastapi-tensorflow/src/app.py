from os.path import join as path_join

import cv2
from fastapi import FastAPI, File, UploadFile
import numpy as np

from .config import models_dir
from .model import CNN


app = FastAPI()

MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number")
MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "tensorflow---2024-04-20-18-35-16", "epoch-9.h5")

model = CNN()
model.load_weights(MODEL_WEIGHTS_FILE_PATH)


def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    print(np.unique(image))
    image = image.astype(np.float64)
    image = image / 255
    image = np.array([image])
    return image


@app.post("/api/run-inference")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)

    predicted_label = int(np.argmax(prediction))

    return {
        "predicted_label": predicted_label,
    }
