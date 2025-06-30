// server.js
import express from 'express';
import * as faceapi from '@vladmandic/face-api';
import tf from '@tensorflow/tfjs-node';
import multer from 'multer';
import fetch from 'node-fetch';

// Monkey‑patch para usar tfjs-node no face‑api
faceapi.env.monkeyPatch({
  fetch: fetch,
  Canvas: null,
  Image: null,
  ImageData: null,
  createCanvas: null,
});

// Multer em memória
const upload = multer();

// CDN dos modelos
const MODEL_URL = 'https://unpkg.com/@vladmandic/face-api@1.7.2/model/';

async function initModels() {
  console.log('🔄 Carregando modelos via CDN...');
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  console.log('✅ Modelos carregados!');
}

async function startServer() {
  await initModels();

  const app = express();

  app.get('/', (_req, res) => {
    res.send('🚀 Face API Bot no ar!');
  });

  app.post(
    '/compare',
    upload.single('image'),
    async (req, res) => {
      try {
        if (!req.file) {
          return res
            .status(400)
            .json({ error: 'Envie o campo "image" com o arquivo.' });
        }

        // Decodifica imagem para tensor
        const imgTensor = tf.node.decodeImage(req.file.buffer);

        const detections = await faceapi
          .detectAllFaces(imgTensor)
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (detections.length === 0) {
          return res
            .status(404)
            .json({ error: 'Nenhuma face detectada.' });
        }

        const descriptors = detections.map(d =>
          Array.from(d.descriptor)
        );

        res.json({ count: descriptors.length, descriptors });
      } catch (err) {
        console.error('❌ Erro na /compare:', err);
        res.status(500).json({ error: 'Erro interno ao processar imagem.' });
      }
    }
  );

  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`🚀 Bot rodando em http://0.0.0.0:${port}`);
  });
}

startServer();
