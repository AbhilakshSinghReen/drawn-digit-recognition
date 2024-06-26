from os.path import join as path_join

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np

from .config import models_dir
from .model import CNN


app = FastAPI()

MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_number.h5")

model = CNN()
model.load_weights(MODEL_WEIGHTS_FILE_PATH)


def normalize(x, axis=-1, order=2):
    "Taken from: https://github.com/keras-team/keras/blob/v3.2.1/keras/utils/numerical_utils.py#L7-L34"

    norm = np.atleast_1d(np.linalg.norm(x, order, axis))
    norm[norm == 0] = 1

    axis = axis or -1
    return x / np.expand_dims(norm, axis)


def preprocess_image(image):
    image = image.astype(np.float32)
    image = np.array([image])
    image = normalize(image, axis=1)
    return image


@app.post("/api/run-inference")
async def run_inference(file: UploadFile = File(...)):
    file_contents = await file.read()
    np_arr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)

    predicted_label = int(np.argmax(prediction))

    return JSONResponse(
        content={
            "predicted_label": predicted_label,
        },
        status_code=200
    )
