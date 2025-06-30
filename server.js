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

const BASE_PATH = "./base_faces";

let labeledDescriptors = [];

async function loadModels() {
    const modelPath = "./models";
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
}

async function loadBase() {
    const files = fs.readdirSync(BASE_PATH);
    for (const file of files) {
        const img = await canvas.loadImage(path.join(BASE_PATH, file));
        const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        if (detection) {
            labeledDescriptors.push({ file, descriptor: detection.descriptor });
        }
    }
}

app.post("/compare", async (req, res) => {
    if (!req.files || !req.files.image) {
        return res.status(400).send("No image uploaded");
    }

    const imgBuffer = req.files.image.data;
    const img = await canvas.loadImage(imgBuffer);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
    if (!detection) return res.status(404).json({ error: "No face detected" });

    const results = labeledDescriptors.map(base => ({
        file: base.file,
        percent: (1 - faceapi.euclideanDistance(base.descriptor, detection.descriptor)) * 100
    }));

    results.sort((a, b) => b.percent - a.percent);
    res.json(results.slice(0, 3));
});

(async () => {
    await loadModels();
    await loadBase();
    app.listen(PORT, () => console.log(`🔥 Face API running on http://localhost:${PORT}`));
})();