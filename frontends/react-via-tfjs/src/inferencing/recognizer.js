import { loadLayersModel } from "@tensorflow/tfjs";

import { indexMax } from "../utils/mathUtils";

const model = await loadLayersModel("http://localhost:3000/model-tfjs/model.json");

function recognizeDigit(preprocessedImageData) {
  const probabilities = model.predict(preprocessedImageData).dataSync();

  const prediction = indexMax(probabilities);

  return {
    success: true,
    prediction: prediction,
  };
}

export { recognizeDigit };
