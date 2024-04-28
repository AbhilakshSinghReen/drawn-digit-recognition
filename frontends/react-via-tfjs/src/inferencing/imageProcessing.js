import { tensor4d } from "@tensorflow/tfjs";

function preprocessImageData(imageData) {
  console.log(imageData.length);
  for (let i = 0; i < imageData.length; i++) {
    // console.log(imageData(i));
  }

  //   console.log("");
  //   console.log("");
  //   console.log("");

  const redChannel = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; ++i) {
    redChannel[i] = imageData[i * 4] / 255;
    // console.log(redChannel[i]);
  }

  const redChannelTensor = tensor4d(redChannel, [1, 28, 28, 1]);

  return redChannelTensor;
}

export { preprocessImageData };
