import express from 'express';
import { streamModelLogs, predictFutureFrames, getAvailableSequences } from '../controllers/modelTest.controller.js';

const router = express.Router();

// Existing routes
router.post('/folder-path', streamModelLogs);
router.post('/predict-frames', predictFutureFrames);

// Add this new route for getting available sequences
router.get('/available-sequences', getAvailableSequences);

export default router;