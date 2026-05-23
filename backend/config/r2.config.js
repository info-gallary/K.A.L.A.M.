import { S3Client } from '@aws-sdk/client-s3';
import dotenv from 'dotenv';

dotenv.config();

// Validate required environment variables
const requiredEnvVars = [
  'R2_ENDPOINT',
  'R2_ACCESS_KEY_ID', 
  'R2_SECRET_ACCESS_KEY',
  'R2_BUCKET_NAME',
  'R2_PUBLIC_URL'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);

if (missingEnvVars.length > 0) {
  console.error('❌ Missing required R2 environment variables:', missingEnvVars);
  console.error('Please check your .env file');
}

export const s3Client = new S3Client({
  region: 'auto',
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
  forcePathStyle: false, // Use virtual-hosted-style URLs
});

// Test connection function
export const testR2Connection = async () => {
  try {
    const { ListBucketsCommand } = await import('@aws-sdk/client-s3');
    const result = await s3Client.send(new ListBucketsCommand({}));
    console.log('✅ R2 connection successful');
    console.log(`📊 Available buckets: ${result.Buckets?.length || 0}`);
    return true;
  } catch (error) {
    console.error('❌ R2 connection failed:', error.message);
    console.error('Check your R2 credentials and endpoint configuration');
    return false;
  }
};

// Optional: Test connection on startup
if (process.env.NODE_ENV !== 'test') {
  testR2Connection();
}