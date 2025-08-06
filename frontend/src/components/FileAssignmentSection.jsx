// components/FileAssignmentSection.jsx
import React from 'react';
import { Image, FolderOpen, Upload, Trash2, Database } from 'lucide-react';

const FileAssignmentSection = ({
  middleHeight,
  imageTypes,
  uploadedFiles,
  currentPath,
  openFileBrowser,
  handleFileUpload,
  handleBrowserUpload,
  removeFile,
  themeClasses
}) => {
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

  return (
    <div
      className={`${themeClasses.bgTertiary} border-b ${themeClasses.border} p-4`}
      style={{ height: middleHeight }}
    >
      <h2 className={`text-base font-semibold ${themeClasses.text} mb-3`}>Image File Assignment</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 h-full overflow-y-auto">
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
                        className={`px-2 py-1 text-xs rounded transition-colors text-white hover:opacity-90`}
                        style={{ backgroundColor: themeClasses.accent }}
                        title="Upload to cloud"
                      >
                        Upload
                      </button>
                    )}
                    <button
                      onClick={() => removeFile(type.key)}
                      className={`${themeClasses.error} ${themeClasses.errorHover} p-0.5 transition-colors`}
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
                    className={`w-full px-2 py-2 text-xs rounded border ${themeClasses.border} ${themeClasses.hover} transition-colors flex items-center justify-center group`}
                  >
                    <FolderOpen className={`w-3 h-3 mr-1 ${themeClasses.textMuted} group-hover:text-opacity-80`} />
                    <span className={`${themeClasses.textSecondary} group-hover:text-opacity-80`}>
                      Browse Files
                    </span>
                  </button>
                )}

                {/* Manual Upload */}
                <label className="cursor-pointer block">
                  <input
                    type="file"
                    accept="image/*,.tif,.tiff,.h5,.hdf5"
                    onChange={(e) => handleFileUpload(type.key, e)}
                    className="hidden"
                  />
                  <div className={`border border-dashed ${themeClasses.borderDashed} rounded p-3 text-center transition-colors hover:border-opacity-60 group`}>
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
    </div>
  );
};

export default FileAssignmentSection;