from os.path import dirname, join as path_join

import torch

from ..config import models_dir
from ..dataset import data_loaders
from ..model import CNN


# MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "training_id", "epoch-epoch_numbe")
MODEL_WEIGHTS_FILE_PATH = path_join(models_dir, "torch---2024-04-23-08-25-15", "epoch-9.pt")
ONNX_MODEL_FILE_PATH = path_join(dirname(MODEL_WEIGHTS_FILE_PATH), "model.onnx")


if __name__ == "__main__":
    device = torch.device("cpu")

    torch_model = CNN()
    torch_model = torch_model.to(device)

    torch_model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE_PATH, map_location=device))

    torch_model.eval()

    for data, target in data_loaders["test"]:
        data = data.to(device)
        target = target.to(device)
        break

    sample_input = data[0].unsqueeze(0)

    onnx_model = torch.onnx.dynamo_export(
        torch_model,
        sample_input
    )

    onnx_model.save(ONNX_MODEL_FILE_PATH)
