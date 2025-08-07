import express from 'express';
import { testModel } from '../controllers/modelTest.controller.js';
import axios from 'axios';
const router = express.Router();


router.get('/test-model',testModel);
router.post('/folder-path', async (req, res) => {
  const { folderPath } = req.body;

  if (!folderPath) return res.status(400).json({ error: 'folderPath is required' });

  try {
    // Forward to Python backend
    // const pythonRes = await axios.post('http://localhost:8000/process-folder', { folderPath });

    // const pythonRes={data:{temp:"data"}};

    res.status(200).json({ message: '✅ Sent to Python backend', data: folderPath });
  } catch (err) {
    console.error('❌ Error forwarding to Python:', err.message);
    res.status(500).json({ error: 'Failed to send to Python backend' });
  }
});
    

export default router;