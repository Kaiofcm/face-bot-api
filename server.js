const express = require("express");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const faceapi = require("@vladmandic/face-api");
const canvas = require("canvas");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(fileUpload());

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const BASE_PATH = path.join(__dirname, "base_faces");
let labeledDescriptors = [];

// Carrega os modelos uma única vez
async function loadModels() {
  const modelPath = path.join(__dirname, "models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
}

// Lê todas as imagens de base e gera os descritores
async function loadBase() {
  const files = fs.readdirSync(BASE_PATH);
  for (const file of files) {
    const imgPath = path.join(BASE_PATH, file);
    const img = await canvas.loadImage(imgPath);
    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (detection) {
      labeledDescriptors.push({ file, descriptor: detection.descriptor });
    }
  }
}

// Rota POST /compare — recebe a imagem e retorna top 3 matches
app.post("/compare", async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send("No image uploaded");
  }
  const imgBuffer = req.files.image.data;
  const img = await canvas.loadImage(imgBuffer);
  const detection = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!detection) return res.status(404).json({ error: "No face detected" });

  const results = labeledDescriptors.map(base => {
    const dist = faceapi.euclideanDistance(base.descriptor, detection.descriptor);
    const percent = (1 - dist) * 100;
    return { file: base.file, percent: percent.toFixed(2) };
  });
  results.sort((a, b) => b.percent - a.percent);
  res.json(results.slice(0, 3));
});

// Inicialização
(async () => {
  await loadModels();
  await loadBase();
  app.listen(PORT, () => {
    console.log(`🔥 Face API running on http://localhost:${PORT}`);
  });
})();
