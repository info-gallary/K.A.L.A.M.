/* eslint-disable no-unused-vars */
// components/Sidebar.jsx
import React from 'react';
import { Calendar, Clock, Folder, FolderOpen, AlertCircle, RefreshCw, FileText, X } from 'lucide-react';

const Sidebar = ({
  leftWidth,
  selectedDate,
  setSelectedDate,
  selectedTime,
  setSelectedTime,
  isSupported,
  error,
  loading,
  currentPath,
  currentFiles,
  selectRootDirectory,
  disconnectFolder,
  generateReport,
  themeClasses,
  onMouseDown
}) => {
  return (
    <div 
      className={`${themeClasses.bgSecondary} ${themeClasses.border} border-r p-4`}
      style={{ width: leftWidth }}
    >
      <h2 className={`text-base font-semibold ${themeClasses.text} mb-4`}>Report Settings</h2>
      
      {/* Date and Time Selection */}
      {/* <div className="space-y-3 mb-4">
        <div>
          <label className={`block text-xs font-medium ${themeClasses.textSecondary} mb-1`}>
            <Calendar className="inline w-3 h-3 mr-1" />
            Date
          </label>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className={`w-full px-2 py-1.5 text-xs ${themeClasses.input} ${themeClasses.border} border rounded focus:ring-1`}
          />
        </div>
        
        <div>
          <label className={`block text-xs font-medium ${themeClasses.textSecondary} mb-1`}>
            <Clock className="inline w-3 h-3 mr-1" />
            Time
          </label>
          <input
            type="time"
            value={selectedTime}
            onChange={(e) => setSelectedTime(e.target.value)}
            className={`w-full px-2 py-1.5 text-xs ${themeClasses.input} ${themeClasses.border} border rounded focus:ring-1`}
          />
        </div>
      </div> */}

      {/* Browser Support Notice */}
      {!isSupported && (
        <div className={`mb-3 p-2 ${themeClasses.errorBg} border ${themeClasses.errorBorder} rounded`}>
          <div className={`flex items-start ${themeClasses.error}`}>
            <AlertCircle className="w-3 h-3 mr-1.5 mt-0.5 flex-shrink-0" />
            <div className="text-xs">
              <p className="font-medium">Browser Not Supported</p>
              <p>Use Chrome or Edge for file browser.</p>
            </div>
          </div>
        </div>
      )}

      {/* File Browser Access */}
      <div className={`mb-4 p-3 ${themeClasses.card} border rounded`}>
        <h3 className={`text-sm font-medium ${themeClasses.text} mb-2 flex items-center`}>
          <Folder className="w-3 h-3 mr-1" />
          File Browser
        </h3>
        
        {error && (
          <div className={`mb-2 p-2 ${themeClasses.errorBg} border ${themeClasses.errorBorder} rounded`}>
            <div className={`flex items-start ${themeClasses.error}`}>
              <AlertCircle className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" />
              <span className="text-xs">{error}</span>
            </div>
          </div>
        )}

        {!currentPath ? (
          <button
            onClick={selectRootDirectory}
            disabled={loading || !isSupported}
            className={`w-full px-3 py-2 text-xs rounded ${themeClasses.button} transition-colors flex items-center justify-center ${
              !isSupported ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            style={{ backgroundColor: isSupported ? themeClasses.accent : '#666' }}
          >
            {loading ? (
              <>
                <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                Scanning...
              </>
            ) : (
              <>
                <FolderOpen className="w-3 h-3 mr-1" />
                Connect Folder
              </>
            )}
          </button>
        ) : (
          <div>
            <div className={`mb-2 p-2 ${themeClasses.successBg} border ${themeClasses.successBorder} rounded`}>
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className={`text-xs font-medium ${themeClasses.success}`}>✓ Connected</p>
                  <p className={`text-xs ${themeClasses.textMuted} mt-0.5 font-mono truncate`}>{currentPath}</p>
                  <p className={`text-xs ${themeClasses.textMuted}`}>
                    {currentFiles.length} items
                  </p>
                </div>
                <button
                  onClick={disconnectFolder}
                  className={`${themeClasses.textMuted} hover:${themeClasses.error} p-0.5 transition-colors ml-1`}
                  title="Disconnect folder"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            </div>
            
            <div className="flex space-x-1">
              <button
                onClick={selectRootDirectory}
                disabled={loading}
                className={`flex-1 px-2 py-1.5 text-xs rounded border ${themeClasses.border} ${themeClasses.textMuted} ${themeClasses.hover} transition-colors`}
              >
                Change
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Generate Report */}
      <div className={`${themeClasses.card} border rounded p-3`}>
        <h3 className={`text-sm font-medium ${themeClasses.text} mb-2`}>Manual Report</h3>
        <div className={`text-xs ${themeClasses.textMuted} mb-2`}>
          Generate basic report manually
        </div>
        <button
          onClick={generateReport}
          className={`w-full px-3 py-2 text-xs rounded ${themeClasses.button} transition-colors flex items-center justify-center`}
          style={{ backgroundColor: themeClasses.accent }}
        >
          <FileText className="w-3 h-3 mr-1" />
          Generate
        </button>
      </div>
    </div>
  );
};

export default Sidebar;