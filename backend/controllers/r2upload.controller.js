import { PutObjectCommand } from '@aws-sdk/client-s3';
import { s3Client } from '../config/r2.config.js';
import File from '../models/File.model.js';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import fs from 'fs';

// Helper function to determine file type
const getFileType = (extension, mimeType) => {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp', '.gif'];
  const dataExtensions = ['.h5', '.hdf5', '.nc', '.csv', '.json', '.xml'];
  const documentExtensions = ['.pdf', '.txt', '.doc', '.docx'];

  if (imageExtensions.includes(extension)) return 'image';
  if (dataExtensions.includes(extension)) return 'data';
  if (documentExtensions.includes(extension)) return 'document';
  
  // Fallback to MIME type checking
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('text/') || mimeType.includes('document')) return 'document';
  
  return 'other';
};

export const uploadToR2 = async (req, res) => {
  try {
    const file = req.file;
    
    if (!file) {
      console.error('No file received in upload request');
      return res.status(400).json({ 
        success: false, 
        message: 'No file uploaded. Make sure to use "file" as the field name.',
        expectedField: 'file'
      });
    }

    console.log('File received successfully:', {
      originalName: file.originalname,
      size: file.size,
      mimetype: file.mimetype,
      fieldname: file.fieldname
    });

    const extension = path.extname(file.originalname).toLowerCase();
    const baseName = path.basename(file.originalname, extension);
    const uniqueFileName = `${uuidv4()}_${baseName}${extension}`;
    
    const fileType = getFileType(extension, file.mimetype);
    
    const supportedExtensions = [
      '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp', '.gif',
      '.h5', '.hdf5', '.nc', '.csv', '.json', '.xml',
      '.pdf', '.txt', '.doc', '.docx'
    ];
    
    if (!supportedExtensions.includes(extension)) {
      if (fs.existsSync(file.path)) {
        fs.unlinkSync(file.path);
      }
      return res.status(415).json({
        success: false,
        message: `Unsupported file type: ${extension}`,
        supportedTypes: supportedExtensions
      });
    }

    // Check if R2 is properly configured
    if (!process.env.R2_BUCKET_NAME || !process.env.R2_PUBLIC_URL) {
      console.error('R2 configuration missing');
      if (fs.existsSync(file.path)) {
        fs.unlinkSync(file.path);
      }
      return res.status(500).json({
        success: false,
        message: 'Server configuration error: R2 storage not configured'
      });
    }

    const uploadParams = {
      Bucket: process.env.R2_BUCKET_NAME,
      Key: uniqueFileName,
      Body: fs.createReadStream(file.path),
      ContentType: file.mimetype,
    };

    console.log('Starting R2 upload...');

    await s3Client.send(new PutObjectCommand(uploadParams));

    const fileUrl = `${process.env.R2_PUBLIC_URL}/${uniqueFileName}`;

    const savedFile = await File.create({
      filename: uniqueFileName,
      originalName: file.originalname,
      url: fileUrl,
      size: file.size,
      mimeType: file.mimetype,
      fileType: fileType,
      extension: extension,
      storage: 'r2',
      metadata: {
        uploadedAt: new Date(),
      }
    });

    // Clean up temp file
    if (fs.existsSync(file.path)) {
      fs.unlinkSync(file.path);
    }
    
    console.log('Upload completed successfully:', {
      id: savedFile._id,
      originalName: file.originalname,
      url: fileUrl
    });
    
    res.status(200).json({ 
      success: true, 
      message: `File uploaded successfully to R2`,
      data: {
        _id: savedFile._id,
        url: savedFile.url,
        filename: savedFile.filename,
        originalName: savedFile.originalName,
        size: savedFile.size,
        fileType: savedFile.fileType,
        createdAt: savedFile.createdAt
      }
    });

  } catch (err) {
    console.error('R2 Upload error:', err);
    
    // Clean up temp file
    if (req.file && req.file.path && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    let statusCode = 500;
    let errorMessage = 'Upload failed';

    if (err.code === 'NoSuchBucket') {
      errorMessage = 'Storage bucket not found. Check R2 bucket configuration.';
    } else if (err.code === 'AccessDenied') {
      errorMessage = 'Storage access denied. Check R2 credentials.';
    } else if (err.name === 'ValidationError') {
      statusCode = 400;
      errorMessage = 'Database validation error: ' + err.message;
    }
    
    res.status(statusCode).json({ 
      success: false, 
      message: errorMessage,
      error: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
  }
};