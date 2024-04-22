const ort = require("onnxruntime-node");
const sharp = require("sharp");

const inputImageShape = [28, 28];
const inputShape = [1, 28, 28, 1];

async function loadImageAsUint8Array(imagePath) {
  const image = sharp(imagePath).resize(inputImageShape[0], inputImageShape[1]).greyscale();
  const rawImage = image.raw();
  const imageBuffer = await rawImage.toBuffer();

  const imageUint8Array = new Uint8Array(imageBuffer);
  return imageUint8Array;
}

function imageArrayToTensor(imageUint8Array) {
  let tensorValues = new Float32Array(imageUint8Array.length);

  for (let i = 0; i < imageUint8Array.length; i++) {
    tensorValues[i] = imageUint8Array[i] / 255;
  }

  const imageTensor = new ort.Tensor("float32", tensorValues, inputShape);
  return imageTensor;
}

module.exports = { loadImageAsUint8Array, imageArrayToTensor };
