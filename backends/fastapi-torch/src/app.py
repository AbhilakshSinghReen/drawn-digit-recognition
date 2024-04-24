from os.path import join as path_join

import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from torchvision import transforms

from .config import models_dir
from .model import CNN


# MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_numbe")
MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "torch---2024-04-23-08-25-15", "epoch-9.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CNN()
model = model.to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE_PATH, map_location=device))
model.eval()

preprocessing_transforms = transforms.Compose([
    transforms.ToTensor(),
])


@app.post("/api/run-inference")
async def run_inference(file: UploadFile = File(...), image_provider: str = Query(None)):
    file_contents = await file.read()
    np_arr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(np_arr,  cv2.IMREAD_UNCHANGED)
    
    # Remove the alpha channel and make the digit black on white
    if image_provider == "konva":
        image = image[:, :, 3]
    
    cv2.imwrite("original.png", image)
    
    image = cv2.resize(image, (28, 28))
    cv2.imwrite("resized.png", image)

    if len(image.shape) > 2 and image.shape[-1] > 1:
        image = image[:, :, 0]

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
