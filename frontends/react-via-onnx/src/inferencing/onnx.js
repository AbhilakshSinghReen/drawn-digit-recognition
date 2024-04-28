import ort from "onnxruntime-web";

import { imageArrayToTensor } from "./imageProcessing";
import { softmax, indexMax } from "../utils/mathUtils";

// try {
// const ort = await import("onnxruntime-web");
// } catch (e) {
//   console.log(e);
// }

const modelPath = "";

const ortSession = await ort.InferenceSession.create(modelPath);

async function runInference(imageArr) {
  // resize image
  const imageTensor = imageArrayToTensor(imageArr);

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
