import mongoose from 'mongoose';

const FileSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  url: {
    type: String,
    required: true
  },
  size: {
    type: Number,
    required: true
  },
  mimeType: {
    type: String,
    required: true
  },
  fileType: {
    type: String,
    enum: ['image', 'data', 'document', 'other'],
    required: true
  },
  extension: {
    type: String,
    required: true
  },
  storage: {
    type: String,
    enum: ['r2', 'cloudinary'],
    default: 'r2'
  },
  metadata: {
    type: Object,
    default: {}
  },
  tags: [{
    type: String
  }],
  isActive: {
    type: Boolean,
    default: true
  }
}, { 
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for formatted file size
FileSchema.virtual('formattedSize').get(function() {
  const bytes = this.size;
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
});

// Index for better query performance
FileSchema.index({ fileType: 1, createdAt: -1 });
FileSchema.index({ originalName: 'text', filename: 'text' });

const File = mongoose.model('File', FileSchema);
export default File;