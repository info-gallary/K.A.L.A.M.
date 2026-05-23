// hooks/useFileSystem.jsx
import { useState, useCallback, useEffect } from 'react';

export const useFileSystem = () => {
  // Helper functions for localStorage
  const loadFromStorage = (key, defaultValue) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error);
      return defaultValue;
    }
  };

  const saveToStorage = (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error);
    }
  };

  // File System states with localStorage persistence
  const [rootDirectoryHandle, setRootDirectoryHandle] = useState(null);
  const [currentDirectoryHandle, setCurrentDirectoryHandle] = useState(null);
  const [currentPath, setCurrentPath] = useState(() => loadFromStorage('currentPath', ''));
  const [pathHistory, setPathHistory] = useState([]);
  const [currentFiles, setCurrentFiles] = useState(() => loadFromStorage('currentFiles', []));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isSupported] = useState(typeof window !== 'undefined' && 'showDirectoryPicker' in window);

  // Save to localStorage when these values change
  useEffect(() => {
    saveToStorage('currentPath', currentPath);
  }, [currentPath]);

  useEffect(() => {
    saveToStorage('currentFiles', currentFiles);
  }, [currentFiles]);

  // Get file type and icon
  const getFileInfo = useCallback((name, kind) => {
    if (kind === 'directory') {
      return { type: 'Folder', icon: 'Folder', color: 'text-blue-500' };
    }
    
    const ext = name.split('.').pop()?.toLowerCase() || '';
    const fileTypes = {
      // Images
      jpg: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      jpeg: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      png: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      gif: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      bmp: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      webp: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      tiff: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      tif: { type: 'Image', icon: 'Image', color: 'text-green-500' },
      
      // Data files
      h5: { type: 'Data', icon: 'Database', color: 'text-purple-500' },
      hdf5: { type: 'Data', icon: 'Database', color: 'text-purple-500' },
      nc: { type: 'Data', icon: 'FileText', color: 'text-purple-500' },
      csv: { type: 'Data', icon: 'FileText', color: 'text-green-600' },
      json: { type: 'Data', icon: 'FileText', color: 'text-green-600' },
      xml: { type: 'Data', icon: 'FileText', color: 'text-green-600' },
      
      // Documents
      pdf: { type: 'PDF', icon: 'FileText', color: 'text-red-500' },
      doc: { type: 'Document', icon: 'FileText', color: 'text-blue-600' },
      docx: { type: 'Document', icon: 'FileText', color: 'text-blue-600' },
      txt: { type: 'Text', icon: 'FileText', color: 'text-gray-500' },
      
      // Archives
      zip: { type: 'Archive', icon: 'Archive', color: 'text-yellow-600' },
      rar: { type: 'Archive', icon: 'Archive', color: 'text-yellow-600' },
      '7z': { type: 'Archive', icon: 'Archive', color: 'text-yellow-600' },
    };
    
    return fileTypes[ext] || { type: 'File', icon: 'File', color: 'text-gray-400' };
  }, []);

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Check if file is supported format
  const isSupportedFormat = useCallback((name, kind) => {
    if (kind === 'directory') return true;
    
    const ext = name.split('.').pop()?.toLowerCase() || '';
    const supportedFormats = [
      // Images
      'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tif', 'tiff',
      // Data files
      'h5', 'hdf5', 'nc', 'csv', 'json', 'xml',
      // Documents
      'pdf', 'txt', 'doc', 'docx'
    ];
    
    return supportedFormats.includes(ext);
  }, []);

  // Scan directory contents
  const scanDirectory = useCallback(async (directoryHandle, path = '') => {
    try {
      const items = [];
      for await (const [name, handle] of directoryHandle.entries()) {
        // Only include supported formats
        if (!isSupportedFormat(name, handle.kind)) {
          continue;
        }

        const fileInfo = getFileInfo(name, handle.kind);
        
        if (handle.kind === 'directory') {
          items.push({
            name,
            handle,
            kind: 'directory',
            type: fileInfo.type,
            icon: fileInfo.icon,
            color: fileInfo.color,
            path: path ? `${path}/${name}` : name
          });
        } else {
          try {
            const file = await handle.getFile();
            items.push({
              name,
              handle,
              file,
              kind: 'file',
              type: fileInfo.type,
              icon: fileInfo.icon,
              color: fileInfo.color,
              size: file.size,
              sizeFormatted: formatFileSize(file.size),
              lastModified: new Date(file.lastModified).toLocaleString(),
              path: path ? `${path}/${name}` : name
            });
          } catch (fileError) {
            console.warn(`Could not access file ${name}:`, fileError);
            // Skip files that can't be accessed
            continue;
          }
        }
      }
      
      // Sort: directories first, then files, both alphabetically
      items.sort((a, b) => {
        if (a.kind !== b.kind) {
          return a.kind === 'directory' ? -1 : 1;
        }
        return a.name.localeCompare(b.name);
      });
      
      return items;
    } catch (err) {
      throw new Error(`Failed to read directory: ${err.message}`);
    }
  }, [getFileInfo, isSupportedFormat]);

  // Select root directory
  const selectRootDirectory = useCallback(async () => {
    if (!isSupported) {
      setError('File System Access API not supported. Please use Chrome or Edge browser.');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const dirHandle = await window.showDirectoryPicker({
        mode: 'read'
      });

      setRootDirectoryHandle(dirHandle);
      setCurrentDirectoryHandle(dirHandle);
      setCurrentPath(dirHandle.name);
      setPathHistory([]);

      const items = await scanDirectory(dirHandle, dirHandle.name);
      setCurrentFiles(items);
      setLoading(false);

      console.log(`📁 Connected to folder: ${dirHandle.name}`);
      console.log(`📊 Found ${items.length} supported files`);

    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Folder selection was cancelled');
      } else {
        setError(`Failed to access folder: ${err.message}`);
      }
      setLoading(false);
    }
  }, [isSupported, scanDirectory]);

  // Disconnect folder
  const disconnectFolder = useCallback(() => {
    setRootDirectoryHandle(null);
    setCurrentDirectoryHandle(null);
    setCurrentPath('');
    setPathHistory([]);
    setCurrentFiles([]);
    setError(null);
    
    // Clear from localStorage
    localStorage.removeItem('currentPath');
    localStorage.removeItem('currentFiles');
    
    console.log('📁 Disconnected from folder');
  }, []);

  // Navigate to subdirectory
  const navigateToDirectory = useCallback(async (directoryHandle, dirName) => {
    try {
      setLoading(true);
      setError(null);

      // Add current directory to history
      setPathHistory(prev => [...prev, { handle: currentDirectoryHandle, name: currentPath }]);
      
      const newPath = currentPath + '/' + dirName;
      setCurrentDirectoryHandle(directoryHandle);
      setCurrentPath(newPath);

      const items = await scanDirectory(directoryHandle, newPath);
      setCurrentFiles(items);
      setLoading(false);

      console.log(`📁 Navigated to: ${newPath}`);

    } catch (err) {
      setError(`Failed to navigate to directory: ${err.message}`);
      setLoading(false);
    }
  }, [currentDirectoryHandle, currentPath, scanDirectory]);

  // Navigate back
  const navigateBack = useCallback(async () => {
    if (pathHistory.length === 0) return;

    try {
      setLoading(true);
      setError(null);

      const previous = pathHistory[pathHistory.length - 1];
      setPathHistory(prev => prev.slice(0, -1));
      setCurrentDirectoryHandle(previous.handle);
      setCurrentPath(previous.name);

      const items = await scanDirectory(previous.handle, previous.name);
      setCurrentFiles(items);
      setLoading(false);

      console.log(`📁 Navigated back to: ${previous.name}`);

    } catch (err) {
      setError(`Failed to navigate back: ${err.message}`);
      setLoading(false);
    }
  }, [pathHistory, scanDirectory]);

  return {
    // State
    rootDirectoryHandle,
    currentDirectoryHandle,
    currentPath,
    pathHistory,
    currentFiles,
    loading,
    error,
    isSupported,
    
    // Actions
    selectRootDirectory,
    disconnectFolder,
    navigateToDirectory,
    navigateBack,
    
    // Utilities
    getFileInfo,
    formatFileSize,
    scanDirectory,
    isSupportedFormat
  };
};