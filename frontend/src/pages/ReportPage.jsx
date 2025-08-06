/* eslint-disable no-undef */
import React, { useState, useCallback, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';

import { useFileSystem } from '../hooks/useFileSystem.jsx';

import EnhancedFileBrowser from '../components/EnhancedFileBrowser';
import Sidebar from '../components/Sidebar';
import FileAssignmentSection from '../components/FileAssignmentSection';
import ReportHistorySection from '../components/ReportHistorySection';
import Resizer from '../components/Resizer';

import { axiosInstance } from '../libs/axios.js';
import toast from 'react-hot-toast';

const ReportPage = () => {

  // Helper function to load from localStorage
  const loadFromStorage = (key, defaultValue) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error);
      return defaultValue;
    }
  };

  // Helper function to save to localStorage
  const saveToStorage = (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error);
    }
  };

  const [isDarkMode, setIsDarkMode] = useState(() => loadFromStorage('isDarkMode', true));
  const [selectedDate, setSelectedDate] = useState(() => loadFromStorage('selectedDate', ''));
  const [selectedTime, setSelectedTime] = useState(() => loadFromStorage('selectedTime', ''));
  const [uploadedFiles, setUploadedFiles] = useState(() => loadFromStorage('uploadedFiles', {
    'Past Image Frame 4 (T-3)': null,
    'Past Image Frame 3 (T-2)': null,
    'Past Image Frame 2 (T-1)': null,
    'Current Image Frame 1 (T)': null
  }));

  // Layout sizing states
  const [leftWidth, setLeftWidth] = useState(() => loadFromStorage('leftWidth', 280));
  const [middleHeight, setMiddleHeight] = useState(() => loadFromStorage('middleHeight', 300));
  const [isDragging, setIsDragging] = useState(null);

  // UI states
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Use file system hook
  const {
    rootDirectoryHandle,
    currentDirectoryHandle,
    currentPath,
    pathHistory,
    currentFiles,
    loading,
    error,
    isSupported,
    selectRootDirectory,
    disconnectFolder,
    navigateToDirectory,
    navigateBack
  } = useFileSystem();

  const [reportHistory, setReportHistory] = useState(() => loadFromStorage('reportHistory', []));

  // Save state to localStorage whenever it changes
  useEffect(() => {
    saveToStorage('isDarkMode', isDarkMode);
  }, [isDarkMode]);

  useEffect(() => {
    saveToStorage('selectedDate', selectedDate);
  }, [selectedDate]);

  useEffect(() => {
    saveToStorage('selectedTime', selectedTime);
  }, [selectedTime]);

  useEffect(() => {
    saveToStorage('uploadedFiles', uploadedFiles);
  }, [uploadedFiles]);

  useEffect(() => {
    saveToStorage('leftWidth', leftWidth);
  }, [leftWidth]);

  useEffect(() => {
    saveToStorage('middleHeight', middleHeight);
  }, [middleHeight]);

  useEffect(() => {
    saveToStorage('reportHistory', reportHistory);
  }, [reportHistory]);

  // Restore file previews on component mount
  useEffect(() => {
    const restoreFilePreviews = () => {
      Object.entries(uploadedFiles).forEach(([key, file]) => {
        if (file && !file.localPreview && !file.uploading) {
          // Check file type for preview logic
          const ext = file.name?.split('.').pop()?.toLowerCase() || '';
          const isPreviewableImage = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
          
          if (isPreviewableImage) {
            // For R2 files, use the cloud URL as preview
            if (file.source === 'r2' && file.url) {
              setUploadedFiles(prev => ({
                ...prev,
                [key]: {
                  ...prev[key],
                  localPreview: file.url // Use R2 URL as preview
                }
              }));
            }
            // For local files, we can't restore the blob URL after page reload
            // The File object data is lost in localStorage serialization
          }
        }
      });
    };

    // Small delay to ensure component is fully mounted
    const timer = setTimeout(restoreFilePreviews, 100);
    return () => clearTimeout(timer);
  }, []);

  // Cleanup function for local preview URLs (but not R2 URLs)
  useEffect(() => {
    return () => {
      Object.values(uploadedFiles).forEach(file => {
        if (file?.localPreview && file?.localPreview?.startsWith('blob:')) {
          URL.revokeObjectURL(file.localPreview);
        }
      });
    };
  }, []);

  const imageTypes = [
    { key: 'Past Image Frame 4 (T-3)', label: 'Image at T-3 ', description: '90 mins ago' },
    { key: 'Past Image Frame 3 (T-2)', label: 'Image at T-2 ', description: '60 mins ago' },
    { key: 'Past Image Frame 2 (T-1)', label: 'Image at T-1 ', description: '30 mins ago' },
    { key: 'Current Image Frame 1 (T)', label: 'Image at T', description: 'Current Image' },
  ];

  // Handle mouse events for resizing
  const handleMouseDown = (type) => {
    setIsDragging(type);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;

    if (isDragging === 'left') {
      const newWidth = Math.max(200, Math.min(500, e.clientX));
      setLeftWidth(newWidth);
    } else if (isDragging === 'middle') {
      const newHeight = Math.max(200, Math.min(600, e.clientY - 60));
      setMiddleHeight(newHeight);
    }
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(null);
  }, []);

  // Add mouse event listeners
  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = isDragging === 'left' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Open file browser for slot selection
  const openFileBrowser = (slotType) => {
    if (!currentDirectoryHandle) {
      alert('Please connect a folder first');
      return;
    }
    setSelectedSlot(slotType);
    setSearchTerm('');
    setShowFileBrowser(true);
  };

  // Select file and close browser - just select without upload
  const selectFileAndClose = (fileItem) => {
    if (fileItem.kind === 'directory') {
      navigateToDirectory(fileItem.handle, fileItem.name);
      return;
    }

    // Check file type for preview logic
    const ext = fileItem.name.split('.').pop()?.toLowerCase() || '';
    const isPreviewableImage = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
    
    // Create local preview only for previewable images
    const localPreview = isPreviewableImage ? URL.createObjectURL(fileItem.file) : null;

    setUploadedFiles(prev => ({
      ...prev,
      [selectedSlot]: {
        ...fileItem.file,
        source: 'browser',
        fileHandle: fileItem.handle,
        path: fileItem.path,
        localPreview
      }
    }));
    setShowFileBrowser(false);
    setSelectedSlot(null);
    
    // Show success message for file selection
    toast.success('File selected successfully!');
  };

  // Handle direct file upload from input - Updated for R2
  const handleFileUpload = async (type, event) => {
    const file = event.target.files[0];
    if (!file) return;

    console.log('Frontend: Starting file upload for type:', type, 'File:', file.name);

    // Check file size - R2 limit
    const maxFileSize = 100 * 1024 * 1024; // 100MB limit for R2
    if (file.size > maxFileSize) {
      toast.error('File is too large. Maximum size is 100MB.');
      return;
    }

    // Check file type for preview logic
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const isPreviewableImage = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
    
    // Create local preview only for previewable images
    const localPreview = isPreviewableImage ? URL.createObjectURL(file) : null;

    // Show uploading state immediately
    setUploadedFiles(prev => ({
      ...prev,
      [type]: {
        ...file,
        source: 'upload',
        localPreview,
        uploading: true
      }
    }));

    const toastId = toast.loading('Uploading...');

    try {
      const formData = new FormData();
      formData.append('file', file); // FIXED: Changed from 'image' to 'file'
      
      console.log('Frontend: FormData created with field "file":', {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type
      });

      const response = await axiosInstance.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          toast.loading(`Uploading... ${percent}%`, { id: toastId });
        }
      });
      
      const fileUrl = response.data?.data?.url;

      setUploadedFiles(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          url: fileUrl,
          source: 'r2', // Changed from 'cloudinary' to 'r2'
          uploading: false
        }
      }));

      toast.success('Upload successful!', { id: toastId });
      console.log('Frontend: Upload successful to R2:', response.data);
    } catch (error) {
      let errorMessage = 'Upload failed!';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Upload timeout - file may be too large';
      } else if (error.response) {
        errorMessage = error.response.data?.message || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = 'Network error - check connection';
      }
      
      toast.error(errorMessage, { id: toastId });
      console.error('Frontend: Upload error:', error);
      
      // Reset uploading state on error
      setUploadedFiles(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          uploading: false
        }
      }));
    }
  };

  // Handle file upload from browser (separate function) - Updated for R2
  const handleBrowserFileUpload = async (file, progressCallback) => {
    console.log('Frontend: Browser file upload starting:', file.name);
    
    // Check file size
    const maxFileSize = 100 * 1024 * 1024; // 100MB limit
    if (file.size > maxFileSize) {
      throw new Error('File is too large. Maximum size is 100MB.');
    }

    const formData = new FormData();
    formData.append('file', file); // FIXED: Changed from 'image' to 'file'

    try {
      const response = await axiosInstance.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          if (progressCallback) progressCallback(percent);
        }
      });

      // After successful upload, update the file in uploadedFiles with R2 URL
      const fileUrl = response.data?.data?.url;
      
      // Check file type for preview logic
      const ext = file.name.split('.').pop()?.toLowerCase() || '';
      const isPreviewableImage = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
      const localPreview = isPreviewableImage ? URL.createObjectURL(file) : null;

      setUploadedFiles(prev => ({
        ...prev,
        [selectedSlot]: {
          ...file,
          source: 'r2',
          url: fileUrl,
          localPreview: prev[selectedSlot]?.localPreview || localPreview,
          fileHandle: prev[selectedSlot]?.fileHandle,
          path: prev[selectedSlot]?.path
        }
      }));

      console.log('Frontend: Browser upload successful:', response.data);
      return response.data;
    } catch (error) {
      console.error('Frontend: Browser upload error:', error);
      // Better error handling
      if (error.code === 'ECONNABORTED') {
        throw new Error('Upload timeout - file may be too large or connection is slow');
      } else if (error.response) {
        throw new Error(error.response.data?.message || `Server error: ${error.response.status} - Upload failed`);
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error(`Upload failed: ${error.message}`);
      }
    }
  };

  // Handle browser file upload from the assignment section - Updated for R2
  const handleBrowserUpload = async (type) => {
    const file = uploadedFiles[type];
    if (!file) return;

    console.log('Frontend: Browser upload from assignment section:', type, file.name);

    const toastId = toast.loading('Uploading...');

    try {
      // Set uploading state
      setUploadedFiles(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          uploading: true
        }
      }));

      const formData = new FormData();
      formData.append('file', file); // FIXED: Changed from 'image' to 'file'

      const response = await axiosInstance.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000,
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          toast.loading(`Uploading... ${percent}%`, { id: toastId });
        }
      });
      
      const fileUrl = response.data?.data?.url;

      setUploadedFiles(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          url: fileUrl,
          source: 'r2',
          uploading: false
        }
      }));

      toast.success('Upload successful!', { id: toastId });
      console.log('Frontend: Assignment upload successful:', response.data);
    } catch (error) {
      let errorMessage = 'Upload failed!';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Upload timeout - file may be too large';
      } else if (error.response) {
        errorMessage = error.response.data?.message || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = 'Network error - check connection';
      }
      
      toast.error(errorMessage, { id: toastId });
      console.error('Frontend: Assignment upload error:', error);
      
      // Reset uploading state on error
      setUploadedFiles(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          uploading: false
        }
      }));
    }
  };

  const removeFile = (type) => {
    // Clean up local preview URL if it exists and it's a blob URL
    if (uploadedFiles[type]?.localPreview && uploadedFiles[type]?.localPreview?.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedFiles[type].localPreview);
    }
    
    const newUploadedFiles = {
      ...uploadedFiles,
      [type]: null
    };
    
    setUploadedFiles(newUploadedFiles);
    
    // Also update localStorage immediately
    saveToStorage('uploadedFiles', newUploadedFiles);
  };

  const generateReport = () => {
    if (!selectedDate || !selectedTime) {
      alert('Please select date and time first');
      return;
    }

    const uploadedCount = Object.values(uploadedFiles).filter(file => file !== null).length;
    if (uploadedCount === 0) {
      alert('Please upload at least one image file');
      return;
    }

    const hasBrowser = Object.values(uploadedFiles).some(file => file && file.source === 'browser');
    const source = hasBrowser ? 'File Browser' : 'Manual Upload';

    const newReport = {
      id: Date.now(), // Use timestamp for unique ID
      name: `Analysis Report #${String(reportHistory.length + 1).padStart(3, '0')}`,
      date: selectedDate,
      time: selectedTime,
      status: 'Processing',
      createdAt: new Date().toLocaleString(),
      source: source
    };

    const newReportHistory = [newReport, ...reportHistory];
    setReportHistory(newReportHistory);

    setTimeout(() => {
      const updatedHistory = newReportHistory.map(report =>
        report.id === newReport.id
          ? { ...report, status: 'Completed' }
          : report
      );
      setReportHistory(updatedHistory);
    }, 3000);
  };

  const deleteReport = (id) => {
    const newReportHistory = reportHistory.filter(report => report.id !== id);
    setReportHistory(newReportHistory);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Clear all data function (optional - for debugging or reset)
  const clearAllData = () => {
    if (window.confirm('Are you sure you want to clear all data? This cannot be undone.')) {
      // Clear all localStorage
      localStorage.removeItem('isDarkMode');
      localStorage.removeItem('selectedDate');
      localStorage.removeItem('selectedTime');
      localStorage.removeItem('uploadedFiles');
      localStorage.removeItem('leftWidth');
      localStorage.removeItem('middleHeight');
      localStorage.removeItem('reportHistory');
      
      // Reset all state
      setIsDarkMode(true);
      setSelectedDate('');
      setSelectedTime('');
      setUploadedFiles({
        'Past Image Frame 4 (T-3)': null,
        'Past Image Frame 3 (T-2)': null,
        'Past Image Frame 2 (T-1)': null,
        'Current Image Frame 1 (T)': null
      });
      setLeftWidth(280);
      setMiddleHeight(300);
      setReportHistory([]);
      
      toast.success('All data cleared successfully');
    }
  };

  // Filter files for search and supported formats
  const filteredFiles = currentFiles.filter(item => {
    // First filter by search term
    const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.type.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (!matchesSearch) return false;
    
    // If it's a directory, always show it
    if (item.kind === 'directory') return true;
    
    // For files, show all supported formats including H5
    const ext = item.name.split('.').pop()?.toLowerCase() || '';
    const supportedFormats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'h5', 'hdf5', 'gif', 'bmp', 'webp'];
    
    return supportedFormats.includes(ext);
  });

  // Theme classes
  const themeClasses = {
    bg: isDarkMode ? 'bg-stone-900' : 'bg-neutral-50',
    bgSecondary: isDarkMode ? 'bg-stone-800' : 'bg-stone-100',
    bgTertiary: isDarkMode ? 'bg-stone-700' : 'bg-white',
    text: isDarkMode ? 'text-stone-100' : 'text-stone-900',
    textSecondary: isDarkMode ? 'text-stone-300' : 'text-stone-600',
    textMuted: isDarkMode ? 'text-stone-400' : 'text-stone-500',
    border: isDarkMode ? 'border-stone-700' : 'border-stone-200',
    borderDashed: isDarkMode ? 'border-stone-600' : 'border-stone-300',
    hover: isDarkMode ? 'hover:bg-stone-700' : 'hover:bg-stone-100',
    button: isDarkMode ? 'text-orange-100 hover:bg-stone-600' : 'text-white hover:opacity-90',
    buttonBg: isDarkMode ? 'bg-stone-600' : '',
    input: isDarkMode ? 'bg-stone-800 border-stone-600 text-stone-100' : 'bg-white border-stone-300 text-stone-900',
    card: isDarkMode ? 'bg-stone-800 border-stone-700' : 'bg-white border-stone-200',
    accent: '#C15F3C',
    accentSecondary: '#da7756',

    success: isDarkMode ? 'text-teal-400' : 'text-teal-600',
    successBg: isDarkMode ? 'bg-teal-900/20' : 'bg-teal-50',
    successBorder: isDarkMode ? 'border-teal-700' : 'border-teal-200',

    error: isDarkMode ? 'text-red-400' : 'text-red-600',
    errorHover: isDarkMode ? 'hover:text-red-300' : 'hover:text-red-700',
    errorBg: isDarkMode ? 'bg-red-900/20' : 'bg-red-50',
    errorBorder: isDarkMode ? 'border-red-700' : 'border-red-200',
    warning: isDarkMode ? 'text-yellow-400' : 'text-yellow-600',
    warningBg: isDarkMode ? 'bg-yellow-900/20' : 'bg-yellow-50'
  };

  return (
    <div className={`min-h-screen ${themeClasses.bg}`}>
      {/* Theme Toggle Button */}
      <button
        onClick={toggleTheme}
        className={`fixed top-6 right-6 z-50 p-3 rounded-lg ${themeClasses.buttonBg} ${themeClasses.button} transition-colors shadow-lg`}
        style={{ backgroundColor: isDarkMode ? themeClasses.buttonBg : themeClasses.accent }}
      >
        {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </button>

      {/* Debug: Clear Data Button (optional - can be removed in production) */}
      {process.env.NODE_ENV === 'development' && (
        <button
          onClick={clearAllData}
          className={`fixed top-6 right-20 z-50 p-2 rounded-lg bg-red-600 text-white text-xs transition-colors shadow-lg hover:bg-red-700`}
          title="Clear all saved data (Development only)"
        >
          Clear Data
        </button>
      )}

      {/* Data Persistence Indicator */}
      <div className={`fixed bottom-6 right-6 z-40 p-2 rounded-lg ${themeClasses.bgSecondary} ${themeClasses.border} border text-xs ${themeClasses.textMuted}`}>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Data Auto-Saved</span>
        </div>
      </div>

      {/* Enhanced File Browser Modal */}
      <EnhancedFileBrowser
        showModal={showFileBrowser}
        onClose={() => setShowFileBrowser(false)}
        selectedSlot={selectedSlot}
        imageTypes={imageTypes}
        currentPath={currentPath}
        pathHistory={pathHistory}
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        loading={loading}
        filteredFiles={filteredFiles}
        currentFiles={currentFiles}
        navigateBack={navigateBack}
        selectFileAndClose={selectFileAndClose}
        navigateToDirectory={navigateToDirectory}
        rootDirectoryHandle={rootDirectoryHandle}
        onFileUpload={handleBrowserFileUpload}
        themeClasses={themeClasses}
      />

      {/* Main Layout */}
      <div className="flex h-screen">
        {/* Left Sidebar */}
        <Sidebar
          leftWidth={leftWidth}
          selectedDate={selectedDate}
          setSelectedDate={setSelectedDate}
          selectedTime={selectedTime}
          setSelectedTime={setSelectedTime}
          isSupported={isSupported}
          error={error}
          loading={loading}
          currentPath={currentPath}
          currentFiles={currentFiles}
          selectRootDirectory={selectRootDirectory}
          disconnectFolder={disconnectFolder}
          uploadedFiles={uploadedFiles}
          generateReport={generateReport}
          themeClasses={themeClasses}
        />

        {/* Vertical Resizer */}
        <Resizer
          direction="vertical"
          onMouseDown={() => handleMouseDown('left')}
          themeClasses={themeClasses}
          isDragging={isDragging === 'left'}
        />

        {/* Right Content */}
        <div className="flex-1 flex flex-col">
          {/* File Assignment Section */}
          <FileAssignmentSection
            middleHeight={middleHeight}
            imageTypes={imageTypes}
            uploadedFiles={uploadedFiles}
            currentPath={currentPath}
            openFileBrowser={openFileBrowser}
            handleFileUpload={handleFileUpload}
            handleBrowserUpload={handleBrowserUpload}
            removeFile={removeFile}
            themeClasses={themeClasses}
          />

          {/* Horizontal Resizer */}
          <Resizer
            direction="horizontal"
            onMouseDown={() => handleMouseDown('middle')}
            themeClasses={themeClasses}
            isDragging={isDragging === 'middle'}
          />

          {/* Report History */}
          <ReportHistorySection
            reportHistory={reportHistory}
            deleteReport={deleteReport}
            themeClasses={themeClasses}
          />
        </div>
      </div>
    </div>
  );
};

export default ReportPage;