
from os.path import join as path_join

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import torch
from torchvision import transforms

from .config import models_dir
from .model import CNN


# MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_numbe")
MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "torch---2024-04-23-08-25-15", "epoch-9.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = FastAPI()

model = CNN()
model = model.to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE_PATH, map_location=device))
model.eval()

preprocessing_transforms = transforms.Compose([
    transforms.ToTensor(),
])


@app.post("/api/run-inference")
async def run_inference(file: UploadFile = File(...)):
    file_contents = await file.read()
    np_arr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    preprocessed_image = preprocessing_transforms(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)

    output = model(preprocessed_image)
    prediction = output.argmax(dim=1, keepdim=True)
    prediction = int(prediction)

    return JSONResponse(
        content={
            "predicted_label": prediction,
        },
        status_code=200
    )
