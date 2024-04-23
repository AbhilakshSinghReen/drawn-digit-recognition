from os.path import join as path_join

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime

from .config import models_dir


app = FastAPI()

ONNX_MODEL_FILE_PATH = path_join(models_dir, "training_id", "model.onnx")
ONNX_MODEL_FILE_PATH = path_join(models_dir, "torch---2024-04-23-08-25-15", "model.onnx")

ort_session = onnxruntime.InferenceSession(ONNX_MODEL_FILE_PATH)
ort_session_input_name = ort_session.get_inputs()[0].name
ort_session_output_name = ort_session.get_outputs()[0].name

print(ort_session_input_name)
print(ort_session_output_name)


def normalize(x, axis=-1, order=2):
    "Taken from: https://github.com/keras-team/keras/blob/v3.2.1/keras/utils/numerical_utils.py#L7-L34"

    norm = np.atleast_1d(np.linalg.norm(x, order, axis))
    norm[norm == 0] = 1

    axis = axis or -1
    return x / np.expand_dims(norm, axis)


def preprocess_image_1(image):
    image = image.astype(np.float32)
    image = image[np.newaxis, :, :, np.newaxis]
    image = normalize(image, axis=1)
    return image


def preprocess_image_2(image):
    image = image.astype(np.float32)
    image = image[np.newaxis, np.newaxis, :, :]
    image = normalize(image, axis=1)
    return image


@app.post("/api/run-inference")
async def run_inference(file: UploadFile = File(...)):
    file_contents = await file.read()
    np_arr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    preprocessed_image = preprocess_image_2(image)
    print(preprocessed_image.shape)
    print()
    print()

    prediction = ort_session.run(
        [ort_session_output_name],
        {
            ort_session_input_name: preprocessed_image,
        }
    )

    predicted_label = int(np.argmax(prediction))

    return JSONResponse(
        content={
            "predicted_label": predicted_label,
        },
        status_code=200
    )
