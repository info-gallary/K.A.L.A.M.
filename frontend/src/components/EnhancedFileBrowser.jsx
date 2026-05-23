/* eslint-disable no-unused-vars */
// components/EnhancedFileBrowser.jsx
import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { 
  ArrowLeft, Search, Folder, Image, File, FileText, Archive, 
  RefreshCw, X, Eye, Download, Grid, List, ChevronRight,
  Home, FolderOpen, ZoomIn, ZoomOut, RotateCw, Upload, Cloud,
  Database, Info
} from 'lucide-react';

const EnhancedFileBrowser = ({
  showModal,
  onClose,
  selectedSlot,
  imageTypes,
  currentPath,
  pathHistory,
  searchTerm,
  setSearchTerm,
  loading,
  filteredFiles,
  currentFiles,
  navigateBack,
  selectFileAndClose,
  navigateToDirectory,
  rootDirectoryHandle,
  onFileUpload,
  themeClasses
}) => {
  const [viewMode, setViewMode] = useState('grid');
  const [previewFile, setPreviewFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imagePreviewScale, setImagePreviewScale] = useState(1);
  const [imagePreviewRotation, setImagePreviewRotation] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  // Clean up preview URL when modal closes or file changes
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Helper function to check if file supports image preview
  const supportsImagePreview = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext);
  };

  // Helper function to check if file is H5 data file
  const isH5File = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['h5', 'hdf5'].includes(ext);
  };

  // Helper function to check if file is TIFF
  const isTiffFile = (fileName) => {
    if (!fileName) return false;
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    return ['tif', 'tiff'].includes(ext);
  };

  // Handle file preview
  const handlePreview = async (fileItem) => {
    if (fileItem.kind === 'directory') return;
    
    try {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      
      // Check if file type supports preview
      const supportsPreview = supportsImagePreview(fileItem.name);
      
      if (supportsPreview) {
        const url = URL.createObjectURL(fileItem.file);
        setPreviewUrl(url);
      } else {
        setPreviewUrl(null); // No preview URL for non-previewable files
      }
      
      setPreviewFile(fileItem);
      setImagePreviewScale(1);
      setImagePreviewRotation(0);
    } catch (error) {
      console.error('Error creating preview:', error);
      toast.error('Failed to create preview');
    }
  };

  // Close preview
  const closePreview = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewFile(null);
    setPreviewUrl(null);
    setImagePreviewScale(1);
    setImagePreviewRotation(0);
  };

  // Handle file selection (not upload) - Auto preview for images
  const handleFileSelect = (fileItem) => {
    if (fileItem.kind === 'directory') {
      navigateToDirectory(fileItem.handle, fileItem.name);
      return;
    }
    
    // Just select the file, don't upload it yet
    setSelectedFile(fileItem);
    
    // Auto-preview for image files or show details for H5/TIFF files
    if (supportsImagePreview(fileItem.name) || isH5File(fileItem.name) || isTiffFile(fileItem.name)) {
      handlePreview(fileItem);
    } else {
      // Close preview for other file types
      closePreview();
    }
  };

  // Handle file upload from browser
  const handleFileUpload = async (fileItem) => {
    if (!onFileUpload || fileItem.kind !== 'file') return;
    
    const toastId = toast.loading('Uploading...');
    
    try {
      setUploading(true);
      
      const result = await onFileUpload(fileItem.file, (progress) => {
        toast.loading(`Uploading... ${progress}%`, { id: toastId });
      });
      
      toast.success('Upload successful!', { id: toastId });
      
      // Close the file browser after successful upload
      setTimeout(() => {
        setUploading(false);
        onClose(); // Close the modal
      }, 500);
      
    } catch (error) {
      console.error('Upload failed:', error);
      setUploading(false);
      toast.error('Upload failed: ' + error.message, { id: toastId });
    }
  };

  // Select file and close browser (for immediate selection without upload)
  const selectFile = () => {
    if (selectedFile) {
      selectFileAndClose(selectedFile);
      setSelectedFile(null);
    }
  };

  // Navigate to root
  const navigateToRoot = () => {
    if (rootDirectoryHandle && pathHistory.length > 0) {
      navigateToDirectory(rootDirectoryHandle, rootDirectoryHandle.name);
    }
  };

  // Generate breadcrumb path
  const getBreadcrumbPath = () => {
    if (!currentPath) return [];
    const parts = currentPath.split('/');
    return parts.map((part, index) => ({
      name: part,
      isLast: index === parts.length - 1,
      path: parts.slice(0, index + 1).join('/')
    }));
  };

  if (!showModal) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-40 p-4">
      <div className={`${themeClasses.card} border rounded-xl w-full max-w-7xl h-[90vh] flex flex-col shadow-2xl`}>
        {/* Header */}
        <div className={`px-4 py-3 border-b ${themeClasses.border} flex items-center justify-between`}>
          <div className="flex items-center space-x-3">
            <div>
              <h3 className={`text-lg font-semibold ${themeClasses.text}`}>
                File System Browser
              </h3>
              {selectedSlot && (
                <p className={`text-sm ${themeClasses.textMuted}`}>
                  Selecting for {selectedSlot}: {imageTypes?.find(t => t.key === selectedSlot)?.description}
                </p>
              )}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
              className={`p-2 rounded-lg ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
              title={`Switch to ${viewMode === 'grid' ? 'list' : 'grid'} view`}
            >
              {viewMode === 'grid' ? <List className="w-4 h-4" /> : <Grid className="w-4 h-4" />}
            </button>
            <button
              onClick={onClose}
              className={`p-2 rounded-lg ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Navigation Bar */}
        <div className={`px-4 py-3 border-b ${themeClasses.border} ${themeClasses.bgSecondary}`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <button
                onClick={navigateToRoot}
                disabled={pathHistory.length === 0}
                className={`p-1.5 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors ${
                  pathHistory.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                title="Go to root"
              >
                <Home className="w-4 h-4" />
              </button>
              
              <button
                onClick={navigateBack}
                disabled={pathHistory.length === 0}
                className={`p-1.5 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors ${
                  pathHistory.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                title="Go back"
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
            </div>

            <div className="relative">
              <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 ${themeClasses.textMuted}`} />
              <input
                type="text"
                placeholder="Search files and folders..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className={`pl-10 pr-4 py-2 w-80 text-sm ${themeClasses.input} ${themeClasses.border} border rounded-lg focus:ring-2`}
              />
            </div>
          </div>

          {/* Breadcrumb */}
          <div className="flex items-center space-x-1 text-sm">
            <FolderOpen className={`w-4 h-4 ${themeClasses.textMuted}`} />
            <div className="flex items-center space-x-1">
              {getBreadcrumbPath().map((crumb, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <ChevronRight className={`w-3 h-3 ${themeClasses.textMuted}`} />}
                  <span className={`${crumb.isLast ? themeClasses.text : themeClasses.textMuted} font-mono`}>
                    {crumb.name}
                  </span>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* File List */}
          <div className={`flex-1 overflow-y-auto ${previewFile ? 'w-2/3' : 'w-full'}`}>
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <RefreshCw className={`w-8 h-8 animate-spin ${themeClasses.textMuted}`} />
                <span className={`ml-3 text-sm ${themeClasses.textMuted}`}>Loading...</span>
              </div>
            ) : (
              <div className="p-4">
                {viewMode === 'grid' ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-3">
                    {filteredFiles.map((item, index) => {
                      let IconComponent = FileText;
                      
                      if (item.icon === 'Image') {
                        IconComponent = Image;
                      } else if (item.icon === 'Folder') {
                        IconComponent = Folder;
                      } else if (item.icon === 'Archive') {
                        IconComponent = Archive;
                      } else if (item.icon === 'File') {
                        IconComponent = File;
                      } else if (isH5File(item.name)) {
                        IconComponent = Database;
                      }
                      
                      const isSelected = selectedFile?.name === item.name;
                      
                      return (
                        <div
                          key={index}
                          className={`group relative aspect-square rounded-lg border-2 transition-all hover:shadow-lg cursor-pointer overflow-hidden ${
                            isSelected 
                              ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20' 
                              : `${themeClasses.border} ${themeClasses.hover}`
                          }`}
                          onClick={() => handleFileSelect(item)}
                        >
                          <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
                            <div className={`p-2 rounded-lg mb-2 ${
                              item.kind === 'directory' ? themeClasses.successBg : 
                              isH5File(item.name) ? 'bg-purple-100 dark:bg-purple-900/20' :
                              themeClasses.bgSecondary
                            }`}>
                              <IconComponent className={`w-8 h-8 ${
                                isH5File(item.name) ? 'text-purple-600 dark:text-purple-400' : item.color
                              }`} />
                            </div>
                            <span className={`text-xs ${themeClasses.text} text-center truncate w-full px-1`}>
                              {item.name}
                            </span>
                            {item.kind === 'file' && (
                              <span className={`text-xs ${themeClasses.textMuted} mt-1`}>
                                {item.sizeFormatted}
                              </span>
                            )}
                          </div>
                          
                          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex space-x-1">
                            {item.kind === 'file' && (supportsImagePreview(item.name) || isH5File(item.name) || isTiffFile(item.name)) && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handlePreview(item);
                                }}
                                className={`p-1 rounded ${themeClasses.bgTertiary} ${themeClasses.textMuted} hover:${themeClasses.text} transition-colors shadow-sm`}
                                title={supportsImagePreview(item.name) ? "Preview" : "Details"}
                              >
                                {supportsImagePreview(item.name) ? <Eye className="w-3 h-3" /> : <Info className="w-3 h-3" />}
                              </button>
                            )}
                          </div>
                          
                          {item.kind === 'directory' && (
                            <div className="absolute bottom-2 right-2">
                              <ChevronRight className={`w-4 h-4 ${themeClasses.textMuted}`} />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="space-y-1">
                    {filteredFiles.map((item, index) => {
                      let IconComponent = FileText;
                      
                      if (item.icon === 'Image') {
                        IconComponent = Image;
                      } else if (item.icon === 'Folder') {
                        IconComponent = Folder;
                      } else if (item.icon === 'Archive') {
                        IconComponent = Archive;
                      } else if (item.icon === 'File') {
                        IconComponent = File;
                      } else if (isH5File(item.name)) {
                        IconComponent = Database;
                      }
                      
                      const isSelected = selectedFile?.name === item.name;
                      
                      return (
                        <div
                          key={index}
                          onClick={() => handleFileSelect(item)}
                          className={`group flex items-center space-x-3 p-3 rounded-lg border cursor-pointer transition-all hover:shadow-sm ${
                            isSelected 
                              ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20' 
                              : `${themeClasses.border} ${themeClasses.hover}`
                          }`}
                        >
                          <div className={`p-2 rounded-lg ${
                            item.kind === 'directory' ? themeClasses.successBg : 
                            isH5File(item.name) ? 'bg-purple-100 dark:bg-purple-900/20' :
                            themeClasses.bgSecondary
                          }`}>
                            <IconComponent className={`w-5 h-5 ${
                              isH5File(item.name) ? 'text-purple-600 dark:text-purple-400' : item.color
                            }`} />
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <h4 className={`text-sm font-medium ${themeClasses.text} truncate`}>
                              {item.name}
                            </h4>
                            <div className={`flex items-center space-x-3 text-xs ${themeClasses.textMuted} mt-0.5`}>
                              <span className={`px-2 py-0.5 rounded-full ${
                                isH5File(item.name) ? 'bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400' :
                                themeClasses.bgTertiary
                              }`}>
                                {isH5File(item.name) ? 'H5 Data' : item.type}
                              </span>
                              {item.kind === 'file' && (
                                <>
                                  <span>{item.sizeFormatted}</span>
                                  <span>{new Date(item.file.lastModified).toLocaleDateString()}</span>
                                </>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {item.kind === 'file' && (supportsImagePreview(item.name) || isH5File(item.name) || isTiffFile(item.name)) && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handlePreview(item);
                                }}
                                className={`opacity-0 group-hover:opacity-100 p-1.5 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-all`}
                                title={supportsImagePreview(item.name) ? "Preview" : "Details"}
                              >
                                {supportsImagePreview(item.name) ? <Eye className="w-4 h-4" /> : <Info className="w-4 h-4" />}
                              </button>
                            )}
                            
                            {item.kind === 'directory' && (
                              <ChevronRight className={`w-4 h-4 ${themeClasses.textMuted}`} />
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                
                {filteredFiles.length === 0 && !loading && (
                  <div className={`py-16 text-center ${themeClasses.textMuted}`}>
                    <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium mb-2">No items found</p>
                    <p className="text-sm">
                      {searchTerm ? 'Try adjusting your search terms' : 'This folder appears to be empty'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Preview Panel */}
          {previewFile && (
            <div className={`w-1/3 border-l ${themeClasses.border} flex flex-col`}>
              <div className={`px-4 py-3 border-b ${themeClasses.border} flex items-center justify-between`}>
                <h4 className={`text-sm font-medium ${themeClasses.text}`}>
                  {supportsImagePreview(previewFile.name) ? 'Preview' : 'Details'}
                </h4>
                <div className="flex items-center space-x-1">
                  {supportsImagePreview(previewFile.name) && (
                    <>
                      <button
                        onClick={() => setImagePreviewScale(Math.max(0.1, imagePreviewScale - 0.1))}
                        className={`p-1 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
                        title="Zoom out"
                      >
                        <ZoomOut className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setImagePreviewScale(Math.min(3, imagePreviewScale + 0.1))}
                        className={`p-1 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
                        title="Zoom in"
                      >
                        <ZoomIn className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setImagePreviewRotation((prev) => (prev + 90) % 360)}
                        className={`p-1 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
                        title="Rotate"
                      >
                        <RotateCw className="w-4 h-4" />
                      </button>
                    </>
                  )}
                  <button
                    onClick={closePreview}
                    className={`p-1 rounded ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              <div className="flex-1 overflow-auto p-4">
                {supportsImagePreview(previewFile.name) && previewUrl ? (
                  <div className="flex items-center justify-center h-full">
                    <img
                      src={previewUrl}
                      alt={previewFile.name}
                      className="max-w-full max-h-full object-contain transition-transform duration-200"
                      style={{
                        transform: `scale(${imagePreviewScale}) rotate(${imagePreviewRotation}deg)`
                      }}
                      onError={(e) => {
                        console.error('Image preview error:', e);
                        toast.error('Failed to load image preview');
                      }}
                    />
                  </div>
                ) : (
                  <div className={`text-center ${themeClasses.textMuted} p-4`}>
                    {isH5File(previewFile.name) ? (
                      <>
                        <Database className="w-16 h-16 mx-auto mb-4 text-purple-500" />
                        <p className="mb-2 font-medium text-purple-600 dark:text-purple-400">HDF5 Data File</p>
                        <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-left text-xs space-y-2`}>
                          <div className="font-medium text-purple-600 dark:text-purple-400 mb-2">HDF5 File Information</div>
                          <div>Format: Hierarchical Data Format v5</div>
                          <div>Type: Scientific Dataset</div>
                          <div>Contains: Multi-dimensional arrays and metadata</div>
                          <div>Common use: Satellite imagery, climate data</div>
                        </div>
                      </>
                    ) : isTiffFile(previewFile.name) ? (
                      <>
                        <Image className="w-16 h-16 mx-auto mb-4 text-green-500" />
                        <p className="mb-2 font-medium text-green-600 dark:text-green-400">TIFF Image File</p>
                        <div className={`p-3 rounded-lg ${themeClasses.bgSecondary} text-left text-xs space-y-2`}>
                          <div className="font-medium text-green-600 dark:text-green-400 mb-2">TIFF File Information</div>
                          <div>Format: Tagged Image File Format</div>
                          <div>Type: Raster Image</div>
                          <div>Features: Lossless compression, multiple layers</div>
                          <div>Common use: GIS data, satellite imagery</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p className="mb-2">Preview not available for this file type</p>
                      </>
                    )}
                    <div className="mt-3 text-xs">
                      <div>File Format: {previewFile.name.split('.').pop()?.toUpperCase()}</div>
                      <div>Size: {previewFile.sizeFormatted}</div>
                    </div>
                  </div>
                )}
                
                <div className={`mt-4 p-3 ${themeClasses.bgSecondary} rounded-lg`}>
                  <h5 className={`text-sm font-medium ${themeClasses.text} mb-2`}>File Details</h5>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className={themeClasses.textMuted}>Name:</span>
                      <span className={`${themeClasses.text} font-mono truncate ml-2`}>{previewFile.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className={themeClasses.textMuted}>Type:</span>
                      <span className={themeClasses.text}>
                        {isH5File(previewFile.name) ? 'H5 Data' : 
                         isTiffFile(previewFile.name) ? 'TIFF Image' : 
                         previewFile.type}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className={themeClasses.textMuted}>Size:</span>
                      <span className={themeClasses.text}>{previewFile.sizeFormatted}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className={themeClasses.textMuted}>Modified:</span>
                      <span className={themeClasses.text}>{previewFile.lastModified}</span>
                    </div>
                    {isH5File(previewFile.name) && (
                      <div className="flex justify-between">
                        <span className={themeClasses.textMuted}>Format:</span>
                        <span className={themeClasses.text}>HDF5</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Upload Progress Overlay */}
        {uploadProgress !== null && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className={`${themeClasses.card} border rounded-lg p-6 max-w-sm w-full mx-4`}>
              <div className="text-center">
                <Cloud className={`w-12 h-12 mx-auto mb-4 ${themeClasses.textMuted} ${uploading ? 'animate-bounce' : ''}`} />
                <h4 className={`text-lg font-medium ${themeClasses.text} mb-2`}>
                  {uploadProgress === 100 ? 'Upload Complete!' : 'Uploading...'}
                </h4>
                <div className={`w-full ${themeClasses.bgSecondary} rounded-full h-2 mb-4`}>
                  <div 
                    className="h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${uploadProgress}%`,
                      backgroundColor: uploadProgress === 100 ? '#10b981' : themeClasses.accent
                    }}
                  />
                </div>
                <p className={`text-sm ${themeClasses.textMuted}`}>
                  {uploadProgress === 100 ? 'File uploaded successfully' : `${uploadProgress}% complete`}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className={`px-4 py-3 border-t ${themeClasses.border} flex justify-between items-center`}>
          <div className="flex items-center space-x-4">
            <span className={`text-sm ${themeClasses.textMuted}`}>
              {filteredFiles.length} of {currentFiles.length} items
            </span>
            {searchTerm && (
              <span className={`text-sm ${themeClasses.textMuted}`}>
                • Filtered by "{searchTerm}"
              </span>
            )}
            {selectedFile && (
              <span className={`text-sm ${themeClasses.text} font-medium`}>
                • Selected: {selectedFile.name}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {selectedFile && selectedFile.kind === 'file' && onFileUpload && (
              <button
                onClick={() => handleFileUpload(selectedFile)}
                disabled={uploading}
                className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                  uploading ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-90'
                }`}
                style={{ backgroundColor: themeClasses.accent, color: 'white' }}
              >
                {uploading ? 'Uploading...' : 'Upload to Cloud'}
              </button>
            )}
            {selectedFile && selectedFile.kind === 'file' && (
              <button
                onClick={selectFile}
                className={`px-4 py-2 text-sm rounded-lg border transition-colors`}
                style={{ borderColor: themeClasses.accent, color: themeClasses.accent }}
              >
                Select File
              </button>
            )}
            <button
              onClick={onClose}
              className={`px-4 py-2 text-sm rounded-lg border ${themeClasses.border} ${themeClasses.textMuted} ${themeClasses.hover} transition-colors`}
            >
              Close Browser
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedFileBrowser;