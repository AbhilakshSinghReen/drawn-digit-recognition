import ort from "onnxruntime-web";

const inputImageShape = [28, 28];
// const inputShape = [1, 28, 28, 1];
const inputShape = [1, 1, 28, 28];

function imageArrayToTensor(imageUint8Array) {
  let tensorValues = new Float32Array(imageUint8Array.length);

  for (let i = 0; i < imageUint8Array.length; i++) {
    tensorValues[i] = imageUint8Array[i] / 255;
  }

  const imageTensor = new ort.Tensor("float32", tensorValues, inputShape);
  return imageTensor;
}

export { imageArrayToTensor };
