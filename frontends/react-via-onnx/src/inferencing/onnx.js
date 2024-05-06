import ort from "onnxruntime-web";

import { extractGreyscaleImageFromKonvaStage, imageArrayToTensor } from "./imageProcessing";
import { softmax, indexMax } from "../utils/mathUtils";

const modelUrl = process.env.PUBLIC_URL + "/model.onnx";
const ortSession = await ort.InferenceSession.create(modelUrl);

async function runInference(imageDataUri) {
  const greyscaleImage = await extractGreyscaleImageFromKonvaStage(imageDataUri, "tempCanvas");
  const imageTensor = imageArrayToTensor(greyscaleImage);

  const feeds = {
    [ortSession.inputNames[0]]: imageTensor,
  };

  const outputData = await ortSession.run(feeds);
  const output = outputData[ortSession.outputNames[0]];

  const outputSoftmax = softmax(Array.prototype.slice.call(output.data));

  const outputClass = indexMax(outputSoftmax);
  return outputClass;
}

export default runInference;
