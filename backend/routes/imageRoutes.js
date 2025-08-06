import express from 'express';
import { uploadImage } from '../controllers/imageUpload.controller.js';
import { upload } from '../config/cloudinary.config.js';

const router = express.Router();

// Legacy Cloudinary upload route
router.post('/cloudinary-upload', upload.single('image'), uploadImage);

export default router;