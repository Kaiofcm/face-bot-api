import express from 'express';
import * as faceapi from '@vladmandic/face-api';
import canvas from 'canvas';
import multer from 'multer';

// Monkey‐patch para usar canvas no Node
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer(); // sem armazenamento em disco, só buffer

const MODEL_URL = 'https://unpkg.com/@vladmandic/face-api@1.7.2/model/';

async function init() {
  console.log('🔄 Carregando modelos face-api via CDN...');
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  console.log('✅ Modelos carregados com sucesso!');
}

app.get('/', (req, res) => {
  res.send('🚀 Face API Bot rodando!');
});

// Rota /compare agora usa multer para ler o arquivo 'image' em req.file
app.post('/compare', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Envie o campo "image" com o arquivo.' });
    }

    // req.file.buffer é o buffer binário da imagem
    const img = await canvas.loadImage(req.file.buffer);
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!detections.length) {
      return res.status(404).json({ error: 'Nenhuma face detectada.' });
    }

    const descriptors = detections.map(det => det.descriptor);
    res.json({ count: descriptors.length, descriptors });
  } catch (err) {
    console.error('❌ Erro na rota /compare:', err);
    res.status(500).json({ error: 'Erro interno ao processar imagem.' });
  }
});

async function start() {
  await init();
  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`🚀 Face API Bot está no ar em http://0.0.0.0:${port}`);
  });
}

start();
