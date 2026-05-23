import express from 'express';
import multer from 'multer';
import { uploadToR2 } from '../controllers/r2upload.controller.js';

const router = express.Router();

const upload = multer({ 
  dest: 'uploads/',
  limits: { 
    fileSize: 100 * 1024 * 1024
  },
  fileFilter: (req, file, cb) => {
    console.log('Multer file filter - File received:', {
      fieldname: file.fieldname,
      originalname: file.originalname,
      mimetype: file.mimetype,
      size: file.size
    });
    
    const allowedExtensions = [
      '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp', '.gif',
      '.h5', '.hdf5', '.nc', '.csv', '.json', '.xml',
      '.pdf', '.txt', '.doc', '.docx'
    ];
    
    const fileExt = '.' + file.originalname.toLowerCase().split('.').pop();
    
    if (allowedExtensions.includes(fileExt)) {
      console.log('File type allowed:', fileExt);
      cb(null, true);
    } else {
      console.log('File type rejected:', fileExt);
      cb(new Error(`File type not supported: ${fileExt}. Allowed: ${allowedExtensions.join(', ')}`), false);
    }
  }
});

// Debug middleware to log all requests
router.use('/upload', (req, res, next) => {
  console.log('Upload endpoint hit:', {
    method: req.method,
    headers: req.headers,
    body: req.body,
    files: req.files,
    file: req.file
  });
  next();
});

// Single consistent endpoint using 'file' field
router.post('/upload', upload.single('file'), (req, res, next) => {
  console.log('After multer processing:', {
    file: req.file,
    body: req.body,
    multerError: req.multerError
  });
  next();
}, uploadToR2);

router.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'R2 service is running',
    timestamp: new Date().toISOString(),
    config: {
      bucket: !!process.env.R2_BUCKET_NAME,
      bucketName: process.env.R2_BUCKET_NAME,
      publicUrl: !!process.env.R2_PUBLIC_URL,
      publicUrlValue: process.env.R2_PUBLIC_URL,
      endpoint: !!process.env.R2_ENDPOINT,
      accessKey: !!process.env.R2_ACCESS_KEY_ID,
      secretKey: !!process.env.R2_SECRET_ACCESS_KEY
    }
  });
});

// Test endpoint to verify multer is working
router.post('/test-upload', upload.single('file'), (req, res) => {
  console.log('Test upload - File received:', req.file);
  res.json({
    success: true,
    message: 'Test upload successful',
    file: req.file ? {
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
      fieldname: req.file.fieldname
    } : null
  });
});

export default router;