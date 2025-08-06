import express from 'express'
import {config} from 'dotenv'
import cors from 'cors';
import fs from 'fs';

import {connectDB} from './utils/db.js'
import r2Routes from './routes/r2upload.routes.js'

config();
const app = express();

// Environment check function
const checkEnvironment = () => {
  console.log('🔍 Environment Check:');
  console.log(`PORT: ${process.env.PORT || 'not set'}`);
  console.log(`MONGO_URL: ${process.env.MONGO_URL ? '✅ Set' : '❌ Not set'}`);
  
  console.log('\n🔗 R2 Configuration:');
  console.log(`R2_BUCKET_NAME: ${process.env.R2_BUCKET_NAME ? '✅ ' + process.env.R2_BUCKET_NAME : '❌ Not set'}`);
  console.log(`R2_PUBLIC_URL: ${process.env.R2_PUBLIC_URL ? '✅ ' + process.env.R2_PUBLIC_URL : '❌ Not set'}`);
  console.log(`R2_ENDPOINT: ${process.env.R2_ENDPOINT ? '✅ Set' : '❌ Not set'}`);
  console.log(`R2_ACCESS_KEY_ID: ${process.env.R2_ACCESS_KEY_ID ? '✅ Set' : '❌ Not set'}`);
  console.log(`R2_SECRET_ACCESS_KEY: ${process.env.R2_SECRET_ACCESS_KEY ? '✅ Set' : '❌ Not set'}`);
  
  // Check for missing R2 variables
  const requiredR2Vars = ['R2_BUCKET_NAME', 'R2_PUBLIC_URL', 'R2_ENDPOINT', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY'];
  const missingVars = requiredR2Vars.filter(varName => !process.env[varName]);
  
  if (missingVars.length > 0) {
    console.log(`\n❌ Missing R2 environment variables: ${missingVars.join(', ')}`);
    return false;
  }
  
  console.log('\n✅ All R2 environment variables are set');
  return true;
};

// Create uploads directory
const uploadsDir = './uploads';
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
  console.log('📁 Created uploads directory');
}

// Basic middleware
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true,
}));

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'Project Kalam Backend API',
    status: 'running',
    timestamp: new Date().toISOString(),
    endpoints: {
      'POST /api/v1/upload': 'Upload files to Cloudflare R2 (field: "file")',
      'POST /api/v1/test-upload': 'Test upload endpoint (debugging)',
      'GET /api/v1/health': 'R2 service health check'
    }
  });
});

app.use('/api/v1', r2Routes);

// Global error handling
app.use((error, req, res, next) => {
  console.error('Global Error Handler:', {
    message: error.message,
    code: error.code,
    name: error.name,
    stack: process.env.NODE_ENV === 'development' ? error.stack : 'Stack trace hidden'
  });
  
  // Multer errors
  if (error.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      success: false,
      message: 'File too large. Maximum size is 100MB.'
    });
  }
  
  if (error.code === 'LIMIT_UNEXPECTED_FILE') {
    return res.status(400).json({
      success: false,
      message: 'Unexpected file field. Use "file" as the field name.'
    });
  }
  
  if (error.message && error.message.includes('File type not supported')) {
    return res.status(415).json({
      success: false,
      message: error.message
    });
  }
  
  // Default error
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: error.message
  });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, async () => {
  console.log('🚀 Starting Project Kalam Backend...');
  
  // Check environment variables
  const envOk = checkEnvironment();
  
  try {
    await connectDB();
    console.log(`\n✅ Server running on: http://localhost:${PORT}`);
    console.log(`📁 Uploads directory: ${uploadsDir}`);
    console.log(`🌐 CORS enabled for: http://localhost:5173`);
    console.log(`\n📋 Available endpoints:`);
    console.log(`   - POST /api/v1/upload (field: "file")`);
    console.log(`   - POST /api/v1/test-upload (debugging)`);
    console.log(`   - GET  /api/v1/health`);
    
    if (!envOk) {
      console.log(`\n⚠️  WARNING: Some R2 environment variables are missing. File uploads will fail.`);
    }
  } catch (error) {
    console.error('❌ Failed to start server:', error.message);
  }
});