import { cloudinary, upload } from '../config/cloudinary.config.js';
import Image from '../models/Image.model.js'; // Assuming this model is in models/Image.js

export const uploadImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No image file provided',
      });
    }

    console.log('Received File:', req.file);

    const imageUrl = req.file.path; // Cloudinary URL
    const publicId = req.file.filename; // Cloudinary public_id

    // Create and save image in DB
    const image = new Image({
      url: imageUrl,
      publicId: publicId,
    });

    await image.save();

    res.status(201).json({
      success: true,
      message: 'Image uploaded and saved successfully',
      data: {
        _id: image._id,
        url: image.url,
        publicId: image.publicId,
        createdAt: image.createdAt,
      },
    });
  } catch (error) {
    console.error('Upload Image Error:', error);

    let statusCode = 500;
    let errorMessage = 'Failed to upload image';

    if (error.name === 'CastError') {
      statusCode = 400;
      errorMessage = 'Invalid ID format';
    } else if (error.message.includes('file size')) {
      statusCode = 413;
      errorMessage = 'File size exceeds 10MB limit';
    } else if (error.message.includes('image files')) {
      statusCode = 415;
      errorMessage = 'Only JPG, JPEG, or PNG images are allowed';
    }

    res.status(statusCode).json({
      success: false,
      message: errorMessage,
      error: process.env.NODE_ENV === 'development' ? error.message : undefined,
    });
  }
};
