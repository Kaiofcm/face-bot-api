import express from 'express';
import * as faceapi from '@vladmandic/face-api';
import canvas from 'canvas';
import path from 'path';

// Patching environment for face-api to use canvas in Node.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const MODEL_URL = 'https://unpkg.com/@vladmandic/face-api@1.7.2/model/';

// Inicialização dos modelos
async function init() {
  console.log('🔄 Carregando modelos face-api via CDN...');
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  console.log('✅ Modelos carregados com sucesso!');
}

// Rota de healthcheck
app.get('/', (req, res) => {
  res.send('🚀 Face API Bot rodando!');
});

// Rota de comparação
app.post('/compare', async (req, res) => {
  try {
    if (!req.files || !req.files.image) {
      return res.status(400).json({ error: 'Envie o campo "image" com o arquivo.' });
    }

    const imgBuffer = req.files.image.data;
    const img = await canvas.loadImage(imgBuffer);
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!detections.length) {
      return res.status(404).json({ error: 'Nenhuma face detectada.' });
    }

    // Aqui você pode comparar contra uma base de descritores conhecida
    // Exemplo simples: retorna número de faces detectadas e seus descritores
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
