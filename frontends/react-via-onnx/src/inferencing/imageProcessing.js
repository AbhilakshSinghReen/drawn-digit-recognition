import ort from "onnxruntime-web";

const inputImageShape = [28, 28];
const inputShape = [1, 1, 28, 28];

function extractGreyscaleImageFromKonvaStage(imageDataUri, imageProcessingCanvasId) {
  const canvas = document.getElementById(imageProcessingCanvasId);
  const ctx = canvas.getContext("2d");

  const img = new Image();
  img.src = imageDataUri;

  return new Promise((resolve, reject) => {
    img.onload = () => {
      canvas.width = inputImageShape[0];
      canvas.height = inputImageShape[1];

      ctx.drawImage(img, 0, 0, inputImageShape[0], inputImageShape[1]);

      const imageData = ctx.getImageData(0, 0, inputImageShape[0], inputImageShape[1]);
      const imageArray = imageData.data;
      // r(1, 1), g(1, 1), b(1, 1), a(1, 1), r(2, 1), g(2, 1), b(2, 1), a(2, 1), ...

      const alphaChannelOnlyImageData = new ImageData(
        new Uint8ClampedArray(imageArray.length),
        inputImageShape[0],
        inputImageShape[1]
      );

      for (let i = 3; i < imageArray.length; i += 4) {
        alphaChannelOnlyImageData.data[i] = 255;
        alphaChannelOnlyImageData.data[i - 1] = imageArray[i];
        alphaChannelOnlyImageData.data[i - 2] = imageArray[i];
        alphaChannelOnlyImageData.data[i - 3] = imageArray[i];
      }

      ctx.putImageData(alphaChannelOnlyImageData, 0, 0); // Debug only

      const imageUint8Array = new Uint8Array(alphaChannelOnlyImageData.data.length / 4);

      // Take any one channel R, G, or B
      for (let i = 0; i < alphaChannelOnlyImageData.data.length; i += 4) {
        imageUint8Array[i / 4] = alphaChannelOnlyImageData.data[i];
      }

      resolve(imageUint8Array);
    };

    img.onerror = (error) => {
      reject(error);
    };
  });
}

function imageArrayToTensor(imageUint8Array) {
  console.log(imageUint8Array.length);
  let tensorValues = new Float32Array(imageUint8Array.length);

  for (let i = 0; i < imageUint8Array.length; i++) {
    tensorValues[i] = imageUint8Array[i] / 255;
  }

  const imageTensor = new ort.Tensor("float32", tensorValues, inputShape);
  return imageTensor;
}

export { extractGreyscaleImageFromKonvaStage, imageArrayToTensor };
