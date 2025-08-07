/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
// components/FileAssignmentSection.jsx
import React, { useState, useEffect } from 'react';
import axiosInstance from '../libs/axios'

import toast from 'react-hot-toast'
import {
  Image,
  FolderOpen,
  Upload,
  Trash2,
  Database,
  Play,
  TestTube,
  Folder,
  CheckCircle,
  XCircle,
  BarChart3,
  RefreshCw,
  Zap,
  Brain,
  Sun,
  Moon
} from 'lucide-react';

import ModelTestAndTerminalPreview from './ModelTestAndTerminalPreview';

const FileAssignmentSection = ({
  middleHeight,
  imageTypes,
  uploadedFiles,
  currentPath,
  openFileBrowser,
  handleFileUpload,
  handleBrowserUpload,
  removeFile,
  themeClasses,
  isModelTesting,
  setIsModelTesting,
  isPredicting,
  setIsPredicting,
  isDarkMode,
  toggleTheme,
  clearAllData,
  isProcessing
}) => {
  // Persistent state using localStorage
  const [activeTab, setActiveTab] = useState(() => {
    try {
      return localStorage.getItem('activeTab') || 'upload';
    } catch {
      return 'upload';
    }
  });
  const [testFolder, setTestFolder] = useState(() => {
    try {
      const saved = localStorage.getItem('testFolder') || '';
      return saved ? JSON.parse(saved) : '';
    } catch {
      return '';
    }
  });

  const [testResults, setTestResults] = useState(() => {
    try {
      const saved = localStorage.getItem('testResults');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [predictionResults, setPredictionResults] = useState(() => {
    try {
      const saved = localStorage.getItem('predictionResults');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Save to localStorage whenever state changes
  useEffect(() => {
    localStorage.setItem('activeTab', activeTab);
  }, [activeTab]);

  useEffect(() => {
    localStorage.setItem('testFolder', JSON.stringify(testFolder));
  }, [testFolder]);

  useEffect(() => {
    localStorage.setItem('testResults', JSON.stringify(testResults));
  }, [testResults]);

  useEffect(() => {
    localStorage.setItem('predictionResults', JSON.stringify(predictionResults));
  }, [predictionResults]);

  // Helper functions
  const isH5File = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['h5', 'hdf5'].includes(ext);
  };

  const isTiffFile = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['tif', 'tiff'].includes(ext);
  };

  const supportsImagePreview = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
  };

  const getFileIcon = (fileName) => {
    if (isH5File(fileName)) return Database;
    return Image;
  };

  const getFileTypeLabel = (fileName) => {
    if (isH5File(fileName)) return 'H5 Data';
    if (isTiffFile(fileName)) return 'TIFF Image';
    return 'Image';
  };

  const getFileTypeColor = (fileName) => {
    if (isH5File(fileName)) return 'text-purple-600 dark:text-purple-400';
    if (isTiffFile(fileName)) return 'text-green-600 dark:text-green-400';
    return themeClasses.success;
  };

  const getFileTypeBg = (fileName) => {
    if (isH5File(fileName)) return 'bg-purple-100 dark:bg-purple-900/20 border border-purple-300 dark:border-purple-700';
    if (isTiffFile(fileName)) return 'bg-green-100 dark:bg-green-900/20 border border-green-300 dark:border-green-700';
    return `${themeClasses.successBg} border ${themeClasses.successBorder}`;
  };

  // Check if all 4 frames are uploaded
  const allFramesUploaded = () => {
    return imageTypes.every(type => uploadedFiles[type.key] && !uploadedFiles[type.key].uploading);
  };

  // Handle predict future frames
  const handlePredictFutureFrames = async () => {
    if (!allFramesUploaded()) {
      alert('Please upload all 4 image frames first');
      return;
    }

    setIsPredicting(true);

  };

  // Handle test folder selection
  // const handleTestFolderSelection = async () => {
  
  //   try {
  //     await axiosInstance.post('/folder-path', {
  //       folderPath: testFolder
  //     });

  //     // const dirHandle = await window.showDirectoryPicker({
  //     //   mode: 'read'
  //     // });

  //     // // Scan directory for test files
  //     // const testFiles = [];
  //     // for await (const [name, handle] of dirHandle.entries()) {
  //     //   if (handle.kind === 'file') {
  //     //     const ext = name.split('.').pop()?.toLowerCase() || '';
  //     //     if (['h5', 'hdf5', 'tif', 'tiff', 'jpg', 'jpeg', 'png'].includes(ext)) {
  //     //       const file = await handle.getFile();
  //     //       testFiles.push({
  //     //         name,
  //     //         size: file.size,
  //     //         type: ext,
  //     //         handle
  //     //       });
  //     //     }
  //     //   }
  //     // }
  //     // console.log(dirHandle)
  //     // setTestFolder({
  //     //   name: dirHandle.name,
  //     //   handle: dirHandle,
  //     //   fileCount: testFiles.length,
  //     //   files: testFiles.slice(0, 10), // Store first 10 files for preview
  //     //   totalFiles: testFiles.length
  //     // });

  //   } catch (error) {
  //     if (error.name !== 'AbortError') {
  //       console.error('Test folder selection error:', error);
  //       alert('Failed to select test folder: ' + error.message);
  //     }
  //   }
  // };



  // Handle model testing
  // const handleModelTesting = async () => {


  //   setIsModelTesting(true);

  //   // try {
  //   //   // Use axios for API call to backend for prediction
  //   //   const response = await axiosInstance.post('/folder-path', {
  //   //     folderPath: 'C:\\Users\\Admin\\Documents\\Satellite\\August07',
  //   //   });
  //   //   console.log('✅ Node backend response:', response.data)
  //   //   const data = response.data;

  //   // } catch (error) {
  //   //   console.error('❌ Error sending folder path:', err.message);
  //   // }


  //   try {
  //     const ws = new WebSocket('ws://localhost:5001'); // match backend port

  //     ws.onopen = async () => {
  //       console.log('✅ WebSocket connected. Starting model test...');
  //       await axiosInstance.post('/folder-path', {
  //         folderPath: testFolder
  //       });
  //     };

  //     ws.onmessage = (event) => {
  //       const { type, content } = JSON.parse(event.data);

  //       setLogs(prev => [...prev, { type, content }]);
  //     };

  //     ws.onerror = (err) => console.error('WebSocket error:', err);
  //     ws.onclose = () => console.log('❌ WebSocket disconnected');
  //   } catch (err) {
  //     console.error('Axios Error:', err.message);
  //   }
  // };

  // Clear test results
  const clearTestResults = () => {
    setTestResults(null);
    setTestFolder(null);
  };

  const clearPredictionResults = () => {
    setPredictionResults(null);
  };

  return (
    <div
      className={`${themeClasses.bgTertiary} border-b ${themeClasses.border} p-4 relative z-10`}
      style={{ height: middleHeight }}
    >
      {/* Processing Overlay */}
      {isProcessing && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 rounded">
          <div className={`${themeClasses.card} border rounded-lg p-6 max-w-sm w-full mx-4`}>

            {/* <div className="text-center">
              <div className="relative">
                {isPredicting ? (
                  <Brain className={`w-12 h-12 mx-auto mb-4 ${themeClasses.accent} animate-pulse`} style={{ color: themeClasses.accent }} />
                ) : (
                  <TestTube className={`w-12 h-12 mx-auto mb-4 ${themeClasses.accent} animate-bounce`} style={{ color: themeClasses.accent }} />
                )}
                <RefreshCw className="w-6 h-6 absolute -top-1 -right-1 animate-spin text-blue-500" />
              </div>
              <h4 className={`text-lg font-medium ${themeClasses.text} mb-2`}>
                {isPredicting ? 'Predicting Future Frames...' : 'Testing Model...'}
              </h4>
              <p className={`text-sm ${themeClasses.textMuted} mb-4`}>
                {isPredicting ? 'AI is analyzing frames to predict future states' : 'Running model validation tests'}
              </p>
              <div className={`w-full ${themeClasses.bgSecondary} rounded-full h-2`}>
                <div
                  className="h-2 rounded-full transition-all duration-1000 animate-pulse"
                  style={{
                    width: '70%',
                    backgroundColor: themeClasses.accent
                  }}
                />
              </div>
              <p className={`text-xs ${themeClasses.textMuted} mt-2`}>
                Please wait, do not refresh the page...
              </p>
            </div> */}



          </div>
        </div>
      )}

      {/* Header with Tabs and Control Buttons */}
      <div className="flex items-center justify-between mb-4">
        <h2 className={`text-base font-semibold ${themeClasses.text}`}>AI Model Interface</h2>

        <div className="flex items-center space-x-3">
          {/* Tab Navigation */}
          <div className={`flex ${themeClasses.bgSecondary} rounded-lg p-1`}>
            <button
              onClick={() => setActiveTab('upload')}
              disabled={isProcessing}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === 'upload'
                ? `text-white shadow-sm`
                : `${themeClasses.textMuted} hover:${themeClasses.textSecondary}`
                } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
              style={{ backgroundColor: activeTab === 'upload' ? themeClasses.accent : 'transparent' }}
            >
              <Upload className="w-4 h-4 inline mr-1" />
              Frame Upload
            </button>
            <button
              onClick={() => setActiveTab('test')}
              disabled={isProcessing}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === 'test'
                ? `text-white shadow-sm`
                : `${themeClasses.textMuted} hover:${themeClasses.textSecondary}`
                } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
              style={{ backgroundColor: activeTab === 'test' ? themeClasses.accent : 'transparent' }}
            >
              <TestTube className="w-4 h-4 inline mr-1" />
              Model Testing
            </button>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center space-x-2">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              disabled={isProcessing}
              className={` cursor-pointer p-2.5 rounded-lg ${themeClasses.buttonBg} ${themeClasses.button} transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
                }`}
              style={{ backgroundColor: isDarkMode ? themeClasses.buttonBg : themeClasses.accent }}
              title="Toggle theme"
            >
              {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {/* Clear Cache Button - Development only */}
            {process.env.NODE_ENV === 'development' && (
              <button
                onClick={clearAllData}
                disabled={isProcessing}
                className={`cursor-pointer p-2.5 text-xs font-medium rounded-lg bg-stone-600 text-white transition-colors hover:bg-[#C15F3C] ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                title="Clear all saved data (Development only)"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="h-full overflow-y-auto">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-4">
            {/* Frame Upload Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {imageTypes.map((type) => (
                <div key={type.key} className={`${themeClasses.card} border rounded p-3 h-fit`}>
                  <div className="mb-2">
                    <h4 className={`text-sm font-medium ${themeClasses.text}`}>{type.label}</h4>
                    <p className={`text-xs ${themeClasses.textMuted} mt-0.5`}>{type.description}</p>
                  </div>

                  {uploadedFiles[type.key] ? (
                    <div className={`${getFileTypeBg(uploadedFiles[type.key].name)} rounded p-2 mb-2`}>
                      <div className="flex items-center space-x-1.5 mb-1">
                        {React.createElement(getFileIcon(uploadedFiles[type.key].name), {
                          className: `w-3 h-3 ${getFileTypeColor(uploadedFiles[type.key].name)}`
                        })}
                        <span className={`text-xs font-medium ${getFileTypeColor(uploadedFiles[type.key].name)}`}>
                          Selected
                        </span>
                      </div>

                      {/* File Preview/Details Section */}
                      {uploadedFiles[type.key] && (
                        <>
                          {/* H5 Files - Show details instead of preview */}
                          {isH5File(uploadedFiles[type.key].name) ? (
                            <div className="mb-1 p-3 bg-purple-50 dark:bg-purple-900/10 border border-purple-200 dark:border-purple-800 rounded">
                              <div className="flex items-center justify-center mb-2">
                                <Database className="w-8 h-8 text-purple-500" />
                              </div>
                              <div className="text-xs text-purple-700 dark:text-purple-300 text-center space-y-1">
                                <div className="font-medium">HDF5 Data File</div>
                                <div>Scientific Dataset</div>
                                <div>Multi-dimensional arrays</div>
                                {uploadedFiles[type.key].size && (
                                  <div className="text-purple-600 dark:text-purple-400 font-mono">
                                    {(uploadedFiles[type.key].size / (1024 * 1024)).toFixed(2)} MB
                                  </div>
                                )}
                              </div>
                            </div>
                          ) :
                            /* TIFF Files - Show info instead of preview */
                            isTiffFile(uploadedFiles[type.key].name) ? (
                              <div className="mb-1 p-3 bg-green-50 dark:bg-green-900/10 border border-green-200 dark:border-green-800 rounded">
                                <div className="flex items-center justify-center mb-2">
                                  <Image className="w-8 h-8 text-green-500" />
                                </div>
                                <div className="text-xs text-green-700 dark:text-green-300 text-center space-y-1">
                                  <div className="font-medium">TIFF Image File</div>
                                  <div>Raster Image Format</div>
                                  <div>Lossless compression</div>
                                  {uploadedFiles[type.key].size && (
                                    <div className="text-green-600 dark:text-green-400 font-mono">
                                      {(uploadedFiles[type.key].size / (1024 * 1024)).toFixed(2)} MB
                                    </div>
                                  )}
                                </div>
                              </div>
                            ) :
                              /* Uploading state */
                              uploadedFiles[type.key].uploading ? (
                                <div className="mb-1 p-4 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded flex items-center justify-center">
                                  <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
                                    <div className="animate-spin w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full mx-auto mb-1"></div>
                                    Uploading...
                                  </div>
                                </div>
                              ) :
                                /* Regular image preview */
                                supportsImagePreview(uploadedFiles[type.key].name) ? (
                                  <div className="mb-1">
                                    <img
                                      src={
                                        uploadedFiles[type.key].source === 'r2' || uploadedFiles[type.key].source === 'cloudinary'
                                          ? uploadedFiles[type.key].url
                                          : uploadedFiles[type.key].localPreview
                                      }
                                      alt="Preview"
                                      className="w-full max-h-40 object-cover rounded border"
                                      onError={(e) => {
                                        e.target.style.display = 'none';
                                        if (e.target.nextSibling) {
                                          e.target.nextSibling.style.display = 'block';
                                        }
                                      }}
                                    />
                                    <div
                                      className="text-xs text-gray-500 text-center p-2 border rounded hidden"
                                      style={{ display: 'none' }}
                                    >
                                      Preview unavailable
                                    </div>
                                  </div>
                                ) : (
                                  /* Non-previewable files */
                                  <div className="mb-1 p-3 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded">
                                    <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
                                      <Image className="w-8 h-8 mx-auto mb-2 opacity-50" />
                                      <div>Preview not available</div>
                                      <div className="font-mono mt-1">
                                        {uploadedFiles[type.key].name?.split('.').pop()?.toUpperCase()}
                                      </div>
                                    </div>
                                  </div>
                                )}
                        </>
                      )}

                      {/* Filename */}
                      <p className={`text-xs ${themeClasses.text} truncate mb-1 font-mono`}>
                        {uploadedFiles[type.key].name}
                      </p>

                      {/* Source and Remove Button */}
                      <div className="flex items-center justify-between">
                        <div className="flex flex-col">
                          <span className={`text-xs ${themeClasses.textMuted}`}>
                            {uploadedFiles[type.key].source === 'browser' ? 'Browser' :
                              uploadedFiles[type.key].source === 'r2' ? 'R2 Storage' :
                                uploadedFiles[type.key].source === 'cloudinary' ? 'Uploaded' : 'Local'}
                          </span>
                          <span className={`text-xs ${themeClasses.textMuted} font-medium`}>
                            {getFileTypeLabel(uploadedFiles[type.key].name)}
                          </span>
                        </div>
                        <div className="flex items-center space-x-1">
                          {/* Upload to cloud button for browser-selected files */}
                          {uploadedFiles[type.key].source === 'browser' && (
                            <button
                              onClick={() => handleBrowserUpload(type.key)}
                              disabled={isProcessing}
                              className={`px-2 py-1 text-xs rounded transition-colors text-white hover:opacity-90 ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                                }`}
                              style={{ backgroundColor: themeClasses.accent }}
                              title="Upload to cloud"
                            >
                              Upload
                            </button>
                          )}
                          <button
                            onClick={() => removeFile(type.key)}
                            disabled={isProcessing}
                            className={`${themeClasses.error} ${themeClasses.errorHover} p-0.5 transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                              }`}
                            title="Remove file"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-1.5">
                      {/* File Browser Button */}
                      {currentPath && (
                        <button
                          onClick={() => openFileBrowser(type.key)}
                          disabled={isProcessing}
                          className={`w-full px-2 py-2 text-xs rounded border ${themeClasses.border} ${themeClasses.hover} transition-colors flex items-center justify-center group ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                            }`}
                        >
                          <FolderOpen className={`w-3 h-3 mr-1 ${themeClasses.textMuted} group-hover:text-opacity-80`} />
                          <span className={`${themeClasses.textSecondary} group-hover:text-opacity-80`}>
                            Browse Files
                          </span>
                        </button>
                      )}

                      {/* Manual Upload */}
                      <label className={`cursor-pointer block ${isProcessing ? 'pointer-events-none' : ''}`}>
                        <input
                          type="file"
                          accept="image/*,.tif,.tiff,.h5,.hdf5"
                          onChange={(e) => handleFileUpload(type.key, e)}
                          disabled={isProcessing}
                          className="hidden"
                        />
                        <div className={`border border-dashed ${themeClasses.borderDashed} rounded p-3 text-center transition-colors hover:border-opacity-60 group ${isProcessing ? 'opacity-50' : ''
                          }`}>
                          <Upload className={`mx-auto w-3 h-3 ${themeClasses.textMuted} mb-1 group-hover:text-opacity-80`} />
                          <span className={`text-xs ${themeClasses.textMuted} group-hover:text-opacity-80`}>
                            Upload File
                          </span>
                          <div className={`text-xs ${themeClasses.textMuted} mt-0.5 opacity-75`}>
                            JPG, PNG, TIFF, H5
                          </div>
                        </div>
                      </label>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Predict Future Frames Button */}
            <div className={`${themeClasses.card} border rounded p-4 mt-4`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg`} style={{ backgroundColor: `${themeClasses.accent}20` }}>
                    <Zap className="w-5 h-5" style={{ color: themeClasses.accent }} />
                  </div>
                  <div>
                    <h3 className={`text-sm font-medium ${themeClasses.text}`}>AI Frame Prediction</h3>
                    <p className={`text-xs ${themeClasses.textMuted}`}>
                      Generate future frame predictions using uploaded sequence
                    </p>
                  </div>
                </div>
                <button
                  onClick={handlePredictFutureFrames}
                  disabled={!allFramesUploaded() || isProcessing}
                  className={`flex items-center space-x-2 px-4 py-2 text-sm rounded-lg font-medium transition-all ${allFramesUploaded() && !isProcessing
                    ? 'text-white hover:opacity-90 transform hover:scale-105'
                    : 'opacity-50 cursor-not-allowed'
                    }`}
                  style={{ backgroundColor: allFramesUploaded() && !isProcessing ? themeClasses.accent : '#666' }}
                >
                  <Play className="w-4 h-4" />
                  <span>Predict Frames</span>
                </button>
              </div>

              {/* Prediction Results */}
              {predictionResults && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className={`text-sm font-medium ${themeClasses.text}`}>Prediction Results</h4>
                    <button
                      onClick={clearPredictionResults}
                      className={`text-xs ${themeClasses.textMuted} hover:${themeClasses.error} transition-colors`}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className={`p-2 rounded ${themeClasses.bgSecondary}`}>
                        <span className={themeClasses.textMuted}>Processing Time:</span>
                        <div className={`font-mono ${themeClasses.text}`}>{predictionResults.processingTime}</div>
                      </div>
                      <div className={`p-2 rounded ${themeClasses.bgSecondary}`}>
                        <span className={themeClasses.textMuted}>Model Version:</span>
                        <div className={`font-mono ${themeClasses.text}`}>{predictionResults.modelVersion}</div>
                      </div>
                    </div>
                    <div className={`p-2 rounded ${themeClasses.successBg} border ${themeClasses.successBorder}`}>
                      <span className={`text-xs ${themeClasses.success}`}>
                        ✓ Future frames generated successfully with high confidence
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Test Tab */}
        {activeTab === 'test' && (
          <div className="space-y-4">
            {/* Test Folder Selection */}
            <div className={`${themeClasses.card} border rounded p-4`}>
              <div className="flex items-center justify-between mb-3">
                <h3 className={`text-sm font-medium ${themeClasses.text}`}>Test Data Folder</h3>
                {testFolder && (
                  <button
                    onClick={clearTestResults}
                    disabled={isProcessing}
                    className={`text-xs ${themeClasses.textMuted} hover:${themeClasses.error} transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                  >
                    Clear
                  </button>
                )}
              </div>

         
                <div className="text-center py-8">
                  <Folder className={`w-12 h-12 mx-auto mb-3 ${themeClasses.textMuted} opacity-50`} />
                  <p className={`text-sm ${themeClasses.textMuted} mb-4`}>
                    Select a folder containing test data files
                  </p>
                  <input
                    type="text"
                    placeholder="Paste full folder path"
                    value={testFolder}
                    onChange={(e) => setTestFolder(e.target.value)}
                    className=" text-amber-50 border-2 rounded-xl  border-orange-600/60 outline-none  p-2  w-full"
                  />
                  {/* <button
                    onClick={handleModelTesting}
                    disabled={isProcessing}
                    className={`w-full flex items-center space-x-2 px-4 py-2 text-center mt-2 text-sm rounded-lg font-medium transition-all ${isProcessing
                      ? 'opacity-50 cursor-not-allowed bg-gray-400'
                      : 'text-white hover:opacity-90 transform '
                      }`}
                    style={{ backgroundColor: isProcessing ? '#666' : themeClasses.accent }}
                  >
                    <span>Continue</span>
                  </button> */}


                  <ModelTestAndTerminalPreview testFolder={testFolder} />
                </div>
        
                {/* <div className={`p-3 rounded ${themeClasses.successBg} border ${themeClasses.successBorder}`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className={`w-5 h-5 ${themeClasses.success}`} />
                      <div>
                        <p className={`text-sm font-medium ${themeClasses.text}`}>{testFolder.name}</p>
                        <p className={`text-xs ${themeClasses.textMuted}`}>
                          {testFolder.totalFiles} test files found
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleTestFolderSelection}
                      disabled={isProcessing}
                      className={`text-xs px-2 py-1 rounded border ${themeClasses.border} ${themeClasses.textMuted} ${themeClasses.hover} transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                        }`}
                    >
                      Change
                    </button>
                  </div> */}

                  {/* Preview of test files */}
                  {/* {testFolder.files.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <p className={`text-xs ${themeClasses.textMuted} mb-2`}>Preview (first 10 files):</p>
                      <div className="grid grid-cols-2 gap-1 text-xs">
                        {testFolder.files.map((file, index) => (
                          <div key={index} className={`flex items-center space-x-1 p-1 rounded ${themeClasses.bgSecondary}`}>
                            <Database className="w-3 h-3 text-blue-500" />
                            <span className={`truncate ${themeClasses.text} font-mono`}>{file.name}</span>
                          </div>
                        ))}
                      </div>
                      {testFolder.totalFiles > 10 && (
                        <p className={`text-xs ${themeClasses.textMuted} mt-2 text-center`}>
                          ... and {testFolder.totalFiles - 10} more files
                        </p>
                      )}
                    </div>
                  )} */}
                {/* </div> */}
              
            </div>

            {/* Model Testing Button */}
            {testFolder && (
              <div className={`${themeClasses.card} border rounded p-4`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg`} style={{ backgroundColor: `${themeClasses.accent}20` }}>
                      <TestTube className="w-5 h-5" style={{ color: themeClasses.accent }} />
                    </div>
                    <div>
                      <h3 className={`text-sm font-medium ${themeClasses.text}`}>Model Validation</h3>
                      <p className={`text-xs ${themeClasses.textMuted}`}>
                        Run comprehensive testing on your AI model
                      </p>
                    </div>
                  </div>


                  {/* <button
                    onClick={handleModelTesting}
                    disabled={isProcessing}
                    className={`flex items-center space-x-2 px-4 py-2 text-sm rounded-lg font-medium transition-all ${!isProcessing
                      ? 'text-white hover:opacity-90 transform hover:scale-105'
                      : 'opacity-50 cursor-not-allowed'
                      }`}
                    style={{ backgroundColor: !isProcessing ? themeClasses.accent : '#666' }}
                  >
                    <Play className="w-4 h-4" />
                    <span>Run Tests</span>
                  </button> */}
                </div>
              </div>
            )}

            {/* Test Results */}
            {testResults && (
              <div className={`${themeClasses.card} border rounded p-4`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className={`text-sm font-medium ${themeClasses.text} flex items-center space-x-2`}>
                    <BarChart3 className="w-4 h-4" />
                    <span>Test Results</span>
                  </h3>
                  <div className={`flex items-center space-x-2 text-xs ${themeClasses.textMuted}`}>
                    <span>Model: {testResults.results.modelVersion}</span>
                    <span>•</span>
                    <span>{new Date(testResults.results.timestamp).toLocaleString()}</span>
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-center`}>
                    <div className="text-lg font-bold" style={{ color: themeClasses.accent }}>
                      {(testResults.results.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className={`text-xs ${themeClasses.textMuted}`}>Accuracy</div>
                  </div>
                  <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-center`}>
                    <div className="text-lg font-bold" style={{ color: themeClasses.accent }}>
                      {(testResults.results.precision * 100).toFixed(1)}%
                    </div>
                    <div className={`text-xs ${themeClasses.textMuted}`}>Precision</div>
                  </div>
                  <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-center`}>
                    <div className="text-lg font-bold" style={{ color: themeClasses.accent }}>
                      {(testResults.results.recall * 100).toFixed(1)}%
                    </div>
                    <div className={`text-xs ${themeClasses.textMuted}`}>Recall</div>
                  </div>
                  <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-center`}>
                    <div className="text-lg font-bold" style={{ color: themeClasses.accent }}>
                      {(testResults.results.f1Score * 100).toFixed(1)}%
                    </div>
                    <div className={`text-xs ${themeClasses.textMuted}`}>F1 Score</div>
                  </div>
                </div>

                {/* Test Summary */}
                <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} mb-4`}>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="text-center">
                      <div className={`font-bold ${themeClasses.text}`}>{testResults.results.testsRun}</div>
                      <div className={`text-xs ${themeClasses.textMuted}`}>Total Tests</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-green-600">{testResults.results.passedTests}</div>
                      <div className={`text-xs ${themeClasses.textMuted}`}>Passed</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-red-600">{testResults.results.failedTests}</div>
                      <div className={`text-xs ${themeClasses.textMuted}`}>Failed</div>
                    </div>
                    <div className="text-center">
                      <div className={`font-bold ${themeClasses.text}`}>{testResults.results.averageProcessingTime}</div>
                      <div className={`text-xs ${themeClasses.textMuted}`}>Avg Time</div>
                    </div>
                  </div>
                </div>

                {/* Status Badge */}
                <div className="flex items-center justify-center">
                  <div className={`inline-flex items-center space-x-2 px-3 py-2 rounded-full ${testResults.results.accuracy > 0.9
                    ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-300 dark:border-green-700'
                    : testResults.results.accuracy > 0.8
                      ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300 border border-yellow-300 dark:border-yellow-700'
                      : 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700'
                    }`}>
                    {testResults.results.accuracy > 0.9 ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : testResults.results.accuracy > 0.8 ? (
                      <RefreshCw className="w-4 h-4" />
                    ) : (
                      <XCircle className="w-4 h-4" />
                    )}
                    <span className="text-sm font-medium">
                      {testResults.results.accuracy > 0.9
                        ? 'Excellent Performance'
                        : testResults.results.accuracy > 0.8
                          ? 'Good Performance'
                          : 'Needs Improvement'
                      }
                    </span>
                  </div>
                </div>

             
              </div>
            )}


          </div>
        )}
      </div>
    </div>
  );
};

export default FileAssignmentSection;