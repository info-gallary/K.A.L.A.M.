import express from 'express'
import {config} from 'dotenv'
import cors from 'cors';
import fs from 'fs';
import path from 'path';

import  { spawn } from 'child_process';
import {WebSocketServer }  from 'ws'

import {connectDB} from './utils/db.js'
import r2Routes from './routes/r2upload.routes.js'
import testRoutes from './routes/modelTest.routes.js'

import { setClient } from './utils/websocket.js';

config();
const app = express();

// Basic middleware
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true,
}));

// ADD THIS: Static file serving for predicted images
const testOutputPath = "D:\\Hackathon\\ISRO\\pre_final\\inference_outputs\\test";
app.use('/api/prediction-images', express.static(testOutputPath, {
  setHeaders: (res, path) => {
    // Set proper headers for images
    if (path.endsWith('.png')) {
      res.set('Content-Type', 'image/png');
    }
    res.set('Cache-Control', 'public, max-age=86400'); // Cache for 1 day
  }
}));

// Setup WebSocket server on different port
const wss = new WebSocketServer({ port: 3001 });

wss.on('connection', (ws) => {
  console.log('🧠 WebSocket client connected');
  setClient(ws); 
});

// Create uploads directory
const uploadsDir = './uploads';
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
  console.log('📁 Created uploads directory');
}

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'Project Kalam Backend API',
    status: 'running',
    timestamp: new Date().toISOString(),
    endpoints: {
      'POST /api/v1/upload': 'Upload files to Cloudflare R2 (field: "file")',
      'POST /api/v1/test-upload': 'Test upload endpoint (debugging)',
      'GET /api/v1/health': 'R2 service health check',
      'GET /api/prediction-images/*': 'Serve predicted images'
    }
  });
});

app.use('/api/v1', r2Routes);
app.use('/api/v1', testRoutes);

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

const PORT = process.env.PORT || 3000;

app.listen(PORT, async () => {
  console.log('🚀 Starting Project Kalam Backend...');
  
  try {
    await connectDB();
    console.log(`\n✅ Server running on: http://localhost:${PORT}`);
    console.log(`📁 Prediction images served from: ${testOutputPath}`);
    console.log(`🌐 CORS enabled for: http://localhost:5173`);
    console.log(`\n📋 Available endpoints:`);
    console.log(`   - POST /api/v1/upload (field: "file")`);
    console.log(`   - POST /api/v1/predict-frames`);
    console.log(`   - GET  /api/prediction-images/* (static files)`);

  } catch (error) {
    console.error('❌ Failed to start server:', error.message);
  }
});