import { tensor4d } from "@tensorflow/tfjs";

function preprocessImageData(imageDataArr) {
  const redChannel = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; ++i) {
    redChannel[i] = imageDataArr[i * 4] / 255;
  }

  const redChannelTensor = tensor4d(redChannel, [1, 28, 28, 1]);

  return redChannelTensor;
}

export { preprocessImageData };
