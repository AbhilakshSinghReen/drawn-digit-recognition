const fs = require("fs").promises;
const path = require("path");

const express = require("express");
const multer = require("multer");
const ort = require("onnxruntime-node");

const { modelsDir } = require("./config");
const { loadImageAsUint8Array, imageArrayToTensor } = require("./imageProcessing");
const { softmax, indexMax } = require("./mathUtils");

const PORT = process.env.PORT || 8000;

// const modelPath = path.join(modelsDir, "trainingId", "model.onnx")
const modelPath = path.join(modelsDir, "tensorflow---2024-04-20-18-35-16", "model.onnx");

const app = express();
const multipartMiddleware = multer({ dest: "cache/" });

async function startServer() {
  const ortSession = await ort.InferenceSession.create(modelPath);

  app.post("/api/run-inference", multipartMiddleware.single("file"), async (req, res) => {
    if (!req.file) {
      return res.status(400).send("No file uploaded.");
    }

    const imagePath = req.file.path;

    const imageArr = await loadImageAsUint8Array(imagePath);
    const imageTensor = imageArrayToTensor(imageArr);

    const feeds = {
      [ortSession.inputNames[0]]: imageTensor,
    };

    const outputData = await ortSession.run(feeds);
    const output = outputData[ortSession.outputNames[0]];

    const outputSoftmax = softmax(Array.prototype.slice.call(output.data));
    const outputClass = indexMax(outputSoftmax);

    await fs.unlink(imagePath);

    res.status(200);
    res.send({
      predicted_label: outputClass,
    });
  });

  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
}

startServer();
