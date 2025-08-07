import express from 'express';
import { streamModelLogs  } from '../controllers/modelTest.controller.js';
const router = express.Router();


router.post('/folder-path', streamModelLogs);
    

export default router;