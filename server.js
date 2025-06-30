const express = require("express");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const faceapi = require("@vladmandic/face-api");
const canvas = require("canvas");
require("@tensorflow/tfjs-node"); // <— garante que o back‑end tfjs‑node seja carregado

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(fileUpload());

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const BASE_PATH = path.join(__dirname, "base_faces");
let labeledDescriptors = [];

// Carrega modelos e base uma única vez
async function init() {
  const modelPath = path.join(__dirname, "models");
await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');


  const files = fs.readdirSync(BASE_PATH);
  for (const file of files) {
    const img = await canvas.loadImage(path.join(BASE_PATH, file));
    const det = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (det) labeledDescriptors.push({ file, descriptor: det.descriptor });
  }
}

// Rota POST /compare
app.post("/compare", async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send("No image uploaded");
  }
  const imgBuffer = req.files.image.data;
  const img = await canvas.loadImage(imgBuffer);
  const det = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!det) return res.status(404).json({ error: "No face detected" });

  const results = labeledDescriptors.map(base => {
    const dist = faceapi.euclideanDistance(base.descriptor, det.descriptor);
    return { file: base.file, percent: ((1 - dist) * 100).toFixed(2) };
  });
  results.sort((a, b) => b.percent - a.percent);
  res.json(results.slice(0, 3));
});

// Inicializa e sobe o servidor
init().then(() => {
  app.listen(PORT, () => {
    console.log(`🔥 Face API running on port ${PORT}`);
  });
});
