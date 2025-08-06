import axios from 'axios';

export const axiosInstance = axios.create({
  baseURL: 'http://localhost:3000/api/v1', // Updated to match your PORT=3000
  withCredentials: true, // Enable cookies for authentication
  timeout: 120000, // 2 minutes timeout for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
axiosInstance.interceptors.request.use(
  (config) => {
    console.log('🚀 API Request:', {
      method: config.method?.toUpperCase(),
      url: config.url,
      baseURL: config.baseURL,
      timeout: config.timeout
    });
    return config;
  },
  (error) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for debugging and error handling
axiosInstance.interceptors.response.use(
  (response) => {
    console.log('✅ API Response:', {
      status: response.status,
      url: response.config.url,
      data: response.data?.success ? 'Success' : 'Check data'
    });
    return response;
  },
  (error) => {
    console.error('❌ API Error:', {
      status: error.response?.status,
      message: error.response?.data?.message || error.message,
      url: error.config?.url
    });
    
    // Handle common errors
    if (error.code === 'ECONNABORTED') {
      error.message = 'Request timeout - file may be too large or connection is slow';
    } else if (error.response?.status === 413) {
      error.message = 'File too large - maximum size is 100MB';
    } else if (error.response?.status === 415) {
      error.message = error.response.data.message || 'Unsupported file type';
    } else if (!error.response) {
      error.message = 'Network error - please check your connection';
    }
    
    return Promise.reject(error);
  }
);

export default axiosInstance;