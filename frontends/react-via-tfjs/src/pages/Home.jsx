import { useRef, useState } from "react";
import { Stage, Layer, Line } from "react-konva";

import { preprocessImageData } from "../inferencing/imageProcessing";
import { recognizeDigit } from "../inferencing/recognizer";
import { dataURIToBlob } from "../utils/fileUtils";

export default function Home() {
  const stageSize = Math.min(window.innerWidth * 0.9, window.innerHeight * 0.5);

  const [tool, setTool] = useState("pen");
  const [lines, setLines] = useState([]);
  const [strokeWidth, setStrokeWidth] = useState(15);
  const [prediction, setPrediction] = useState(null);

  const stageRef = useRef(null);
  const isDrawing = useRef(false);
  const canvasRef = useRef(null);

  const handleMouseDown = (e) => {
    isDrawing.current = true;
    const pos = e.target.getStage().getPointerPosition();
    setLines([...lines, { tool, strokeWidth, points: [pos.x, pos.y] }]);

    setPrediction(null);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing.current) {
      return;
    }

    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    let lastLine = lines[lines.length - 1];

    lastLine.points = lastLine.points.concat([point.x, point.y]);

    lines.splice(lines.length - 1, 1, lastLine);
    setLines(lines.concat());
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  const handlePredictButtonClick = async (e) => {
    const stageImageDataUri = stageRef.current.toDataURL();
    const stageImageBlob = dataURIToBlob(stageImageDataUri);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();

    img.src = URL.createObjectURL(stageImageBlob);

    img.onload = () => {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 28;
      tempCanvas.height = 28;
      const tempCtx = tempCanvas.getContext("2d");

      // Draw the image on the temporary canvas
      tempCtx.drawImage(img, 0, 0, 28, 28);

      // Get the image data of the temporary canvas
      const imageData = tempCtx.getImageData(0, 0, 28, 28);
      const { data } = imageData;

      // Create a new ImageData object with only alpha channel data
      const inputImageData = new ImageData(new Uint8ClampedArray(data.length), 28, 28);
      // r1, g1, b1, a1, r2, g2, b2, a2, ...

      // Extract alpha channel from the original image data
      for (let i = 3; i < data.length; i += 4) {
        inputImageData.data[i] = 255;
        inputImageData.data[i - 1] = data[i];
        inputImageData.data[i - 2] = data[i];
        inputImageData.data[i - 3] = data[i];
      }

      // Draw the alpha channel data on the main canvas
      ctx.putImageData(inputImageData, 0, 0);

      const preprocessedImageData = preprocessImageData(inputImageData.data);

      const inferencingResult = recognizeDigit(preprocessedImageData);
      if (!inferencingResult.success) {
        window.alert("Could not recognize digit.");
        return;
      }

      setPrediction(inferencingResult.prediction);
    };
  };

  return (
    <div
      style={{
        width: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-start",
        alignItems: "center",
      }}
    >
      <h1>Drawn Digit Prediction</h1>

      <Stage
        ref={stageRef}
        width={stageSize}
        height={stageSize}
        onMouseDown={handleMouseDown}
        onMousemove={handleMouseMove}
        onMouseup={handleMouseUp}
        style={{
          border: "1px solid black",
        }}
      >
        <Layer>
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line.points}
              stroke="#000000"
              strokeWidth={line.strokeWidth}
              tension={0.5}
              lineCap="round"
              lineJoin="round"
              globalCompositeOperation={line.tool === "eraser" ? "destination-out" : "source-over"}
            />
          ))}
        </Layer>
      </Stage>

      <div
        style={{
          width: stageSize,
          marginTop: 10,
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-evenly",
          alignItems: "center",
        }}
      >
        <select
          value={tool}
          onChange={(e) => {
            setTool(e.target.value);
          }}
          style={{
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "16px",
          }}
        >
          <option value="pen">Pen</option>
          <option value="eraser">Eraser</option>
        </select>

        <select
          value={strokeWidth}
          onChange={(e) => {
            setStrokeWidth(parseInt(e.target.value));
          }}
          style={{
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "16px",
          }}
        >
          <option value="1">1</option>
          <option value="3">3</option>
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="15">15</option>
          <option value="20">20</option>
          <option value="30">30</option>
          <option value="40">40</option>
          <option value="50">50</option>
        </select>

        <button
          onClick={() => setLines([])}
          style={{
            padding: "8px 12px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            background: "#ffffff",
            color: "#333",
            fontSize: "16px",
            cursor: "pointer",
          }}
        >
          Clear
        </button>
      </div>

      <button
        onClick={handlePredictButtonClick}
        style={{
          padding: "8px 12px",
          borderRadius: "4px",
          border: "1px solid #444444",
          background: "#eeeeee",
          color: "#333",
          fontSize: "16px",
          cursor: "pointer",
          marginTop: 10,
          marginBottom: 5,
        }}
      >
        Predict
      </button>

      {prediction !== null && (
        <h4
          style={{
            margin: 0,
          }}
        >
          Probably a {prediction}
        </h4>
      )}

      <canvas ref={canvasRef} width={28} height={28}></canvas>
    </div>
  );
}
