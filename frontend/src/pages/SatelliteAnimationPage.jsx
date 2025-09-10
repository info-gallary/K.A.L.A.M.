/* eslint-disable no-unused-vars */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, ChevronLeft, ChevronRight, FolderOpen, Zap, Brain, RefreshCw, Database, ArrowLeft, ArrowRight, Sun, Moon, Home, Activity, Clock, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { usePersistentStorage } from '../hooks/usePersistentStorage';
import ImageAnimator from '../components/ImageAnimator';
import axiosInstance from '../libs/axios';
import toast from 'react-hot-toast';
import {Calendar} from 'lucide-react';

const SatelliteAnimationPage = () => {
  // Persistent theme state
  const [isDarkMode, setIsDarkMode] = usePersistentStorage('isDarkMode', true);
  
  // Core state
  const [selectedDirectory, setSelectedDirectory] = useState(null);
  const [bandData, setBandData] = useState(null);
  const [currentTimeWindow, setCurrentTimeWindow] = useState([0, 1, 2, 3]); // Window of 4 consecutive frames
  const [isAnimating, setIsAnimating] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResults, setPredictionResults] = useState(null);
  const [selectedBand, setSelectedBand] = useState('IMG_VIS');
  const [predictionMetrics, setPredictionMetrics] = useState(null);

  // Animation refs
  const animationRefs = useRef({});

  // Band configuration - mapping to API band numbers
  const BANDS = ['IMG_VIS', 'IMG_MIR', 'IMG_SWIR', 'IMG_TIR1', 'IMG_TIR2', 'IMG_WV'];
  const BAND_LABELS = {
    'IMG_VIS': 'Visible',
    'IMG_MIR': 'Mid-IR', 
    'IMG_SWIR': 'Short Wave IR',
    'IMG_TIR1': 'Thermal IR 1',
    'IMG_TIR2': 'Thermal IR 2',
    'IMG_WV': 'Water Vapor'
  };

  // Updated: Band names now match exactly with backend naming convention
  const BAND_NAME_MAP = {
    'IMG_VIS': 'IMG_VIS',
    'IMG_MIR': 'IMG_MIR',
    'IMG_SWIR': 'IMG_SWIR',
    'IMG_TIR1': 'IMG_TIR1',
    'IMG_TIR2': 'IMG_TIR2',
    'IMG_WV': 'IMG_WV'
  };

  // Theme classes
  const themeClasses = {
    bg: isDarkMode ? 'bg-stone-900' : 'bg-neutral-50',
    bgSecondary: isDarkMode ? 'bg-stone-800' : 'bg-stone-100',
    bgTertiary: isDarkMode ? 'bg-stone-700' : 'bg-white',
    text: isDarkMode ? 'text-stone-100' : 'text-stone-900',
    textSecondary: isDarkMode ? 'text-stone-300' : 'text-stone-600',
    textMuted: isDarkMode ? 'text-stone-400' : 'text-stone-500',
    border: isDarkMode ? 'border-stone-600' : 'border-stone-300',
    card: isDarkMode ? 'bg-stone-800 border-stone-600' : 'bg-white border-stone-300',
    button: isDarkMode ? 'bg-stone-600 text-stone-100 hover:bg-stone-500' : 'bg-stone-200 text-stone-900 hover:bg-stone-300',
    accent: '#C15F3C',
    hover: isDarkMode ? 'hover:bg-stone-700' : 'hover:bg-stone-50',
    input: isDarkMode ? 'bg-stone-700 border-stone-600 text-stone-100' : 'bg-white border-stone-300 text-stone-900'
  };

  // Updated: Calculate actual date/time based on serial index with 30-min intervals
  const calculateDateTime = (index) => {
    // Starting date: August 1, 2024, 12:15 AM
    const startDate = new Date(2024, 7, 1, 0, 15, 0); // Month is 0-indexed
    
    // Add 30 minutes for each frame
    const frameDate = new Date(startDate.getTime() + (index * 30 * 60 * 1000));
    
    return {
      date: frameDate.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: 'numeric'
      }),
      time: frameDate.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true
      }),
      fullDateTime: frameDate
    };
  };

  // Get date range for current window
  const getWindowDateRange = () => {
    if (!currentTimeWindow || currentTimeWindow.length === 0) return '';
    
    const startDateTime = calculateDateTime(currentTimeWindow[0]);
    const endDateTime = calculateDateTime(currentTimeWindow[3]);
    
    // If same date, show just one date
    if (startDateTime.date === endDateTime.date) {
      return `${startDateTime.date} (${startDateTime.time} - ${endDateTime.time})`;
    }
    
    // Different dates, show date range
    return `${startDateTime.date} ${startDateTime.time} - ${endDateTime.date} ${endDateTime.time}`;
  };

  // Updated: Handle directory selection with serial indexing
  const handleDirectorySelect = async () => {
    try {
      const dirHandle = await window.showDirectoryPicker({ mode: 'read' });
      const bandFolders = {};
      
      // Scan for band folders
      for await (const [name, handle] of dirHandle.entries()) {
        if (handle.kind === 'directory' && BANDS.includes(name)) {
          const images = [];
          for await (const [fileName, fileHandle] of handle.entries()) {
            if (fileName.endsWith('.png')) {
              const file = await fileHandle.getFile();
              
              // Extract index from filename (assuming sequential naming)
              const match = fileName.match(/sample_(\d+)/);
              if (match) {
                const index = parseInt(match[1]);
                const dateTime = calculateDateTime(index);
                
                images.push({
                  filename: fileName,
                  file,
                  url: URL.createObjectURL(file),
                  index: index,
                  ...dateTime
                });
              }
            }
          }
          // Sort by index
          images.sort((a, b) => a.index - b.index);
          bandFolders[name] = images;
        }
      }

      if (Object.keys(bandFolders).length === 6) {
        setSelectedDirectory(dirHandle);
        setBandData(bandFolders);
        setCurrentTimeWindow([0, 1, 2, 3]);
        setPredictionResults(null); // Clear previous predictions
        toast.success(`Loaded ${Object.values(bandFolders)[0].length} images per band`);
      } else {
        toast.error('Directory must contain exactly 6 band folders');
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        toast.error('Failed to load directory: ' + error.message);
      }
    }
  };

  // Get max frames available
  const getMaxFrames = () => {
    if (!bandData) return 0;
    return Math.min(...Object.values(bandData).map(band => band.length));
  };

  // Updated: Handle time window navigation with auto-scroll
  const navigateTimeWindow = useCallback((direction) => {
    const maxFrames = getMaxFrames();
    if (maxFrames < 4) return;

    setCurrentTimeWindow(prev => {
      const newWindow = [...prev];
      if (direction === 'right' && newWindow[3] < maxFrames - 1) {
        const updatedWindow = newWindow.map(i => i + 1);
        
        // Auto-scroll timeline to keep window visible
        setTimeout(() => {
          const timelineContainer = document.getElementById('timeline-container');
          const windowElement = document.querySelector(`[data-frame-index="${updatedWindow[1]}"]`);
          if (timelineContainer && windowElement) {
            windowElement.scrollIntoView({ 
              behavior: 'smooth', 
              inline: 'center',
              block: 'nearest'
            });
          }
        }, 100);
        
        return updatedWindow;
      } else if (direction === 'left' && newWindow[0] > 0) {
        const updatedWindow = newWindow.map(i => i - 1);
        
        // Auto-scroll timeline to keep window visible
        setTimeout(() => {
          const timelineContainer = document.getElementById('timeline-container');
          const windowElement = document.querySelector(`[data-frame-index="${updatedWindow[1]}"]`);
          if (timelineContainer && windowElement) {
            windowElement.scrollIntoView({ 
              behavior: 'smooth', 
              inline: 'center',
              block: 'nearest'
            });
          }
        }, 100);
        
        return updatedWindow;
      }
      return prev;
    });
    
    // Clear prediction results when time window changes
    setPredictionResults(null);
    setPredictionMetrics(null);
  }, [getMaxFrames]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'ArrowRight') {
        navigateTimeWindow('right');
      } else if (e.key === 'ArrowLeft') {
        navigateTimeWindow('left');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [navigateTimeWindow]);

  // Simple toggle animation function - starts animation that will auto-stop
  const handlePlayAnimation = () => {
    if (isAnimating) {
      // If currently animating, stop it
      setIsAnimating(false);
    } else {
      // Start animation - it will auto-stop after one cycle
      setIsAnimating(true);
    }
  };

  // Handle animation completion callback
  const handleAnimationComplete = () => {
    // Animation completed one cycle - auto pause
    setIsAnimating(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (bandData) {
        Object.values(bandData).forEach(band => {
          band.forEach(img => URL.revokeObjectURL(img.url));
        });
      }
    };
  }, []);

  // Handle prediction request
  const handlePredictFutureFrames = async () => {
    if (!bandData) return;
    
    setIsPredicting(true);
    const toastId = toast.loading('Predicting future frames...');

    try {
      // Prepare data for prediction - current window frames
      const predictionData = {
        timeWindow: currentTimeWindow,
        selectedDirectory: selectedDirectory.name,
        bands: Object.keys(BAND_NAME_MAP), // Send actual band names instead of numbers
        windowSize: 4
      };

      const response = await axiosInstance.post('/predict-frames', predictionData);
      
     if (response.data?.success) {
      setPredictionResults(response.data.data);
      setPredictionMetrics(response.data.data.performance);
      toast.success('Prediction completed!', { id: toastId });
    } else {
      // Handle edge case where response is success: false
      throw new Error(response.data.error || 'Prediction failed');
    }
  } catch (error) {
    console.error('Prediction error:', error);

    if (error.response?.status === 404) {
      // If folder not found, show null images and null metrics
      const nullData = {
        predictedFrames: [],
        performance: {
          raw: null,
          aggregates: {},
          overall: {
            avg_mse_loss: null,
            total_frames: 0,
            sequence_id: null
          }
        },
        summary: {
          totalPredictedFrames: 0,
          framesFound: 0,
          bandsPerFrame: 0,
          metricsAvailable: false
        }
      };

      setPredictionResults(nullData);
      setPredictionMetrics(nullData.performance);
      toast('Prediction folder not found, showing empty results.', { id: toastId, icon: '⚠️' });
    } else {
      // For other errors
      toast.error('Prediction failed: ' + error.message, { id: toastId });
    }
  } finally {
    setIsPredicting(false);
  }

  };

  // Get current frame image for a band (simplified)
  const getCurrentFrameImage = (bandName, windowIndex = 0) => {
    if (!bandData || !bandData[bandName]) return null;
    const frameIndex = currentTimeWindow[windowIndex];
    return bandData[bandName][frameIndex];
  };

  // Updated: Get predicted image URL from results - Fixed for new naming convention
  const getPredictedImageUrl = (timeStep, bandName) => {
    if (!predictionResults?.predictedFrames) return null;
    
    const frame = predictionResults.predictedFrames.find(f => f.timeStep === timeStep);
    
    if (frame?.bands?.[bandName]) {
      // Use the relativePath from response to construct the URL
      const relativePath = frame.bands[bandName].relativePath;
      // Backend serves static files at /api/prediction-images/
      return `http://localhost:3000/api/prediction-images/${relativePath}`;
    }
    return null;
  };

  // Get ground truth image for future frames
  const getGroundTruthImage = (offset, bandName) => {
    if (!bandData || !bandData[bandName]) return null;
    const frameIndex = currentTimeWindow[3] + offset; // offset is 1, 2, or 3 for T+1, T+2, T+3
    return bandData[bandName][frameIndex];
  };

  // Updated: Format time display using calculated date/time
  const formatTimeOnly = (index) => {
    if (typeof index !== 'number') return '';
    const dateTime = calculateDateTime(index);
    return dateTime.time;
  };

  // Updated: Format full timestamp display
  const formatTimestamp = (index) => {
    if (typeof index !== 'number') return '';
    const dateTime = calculateDateTime(index);
    return `${dateTime.date} ${dateTime.time}`;
  };

  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  // Render performance metrics card
  const renderPerformanceMetrics = () => {
    if (!predictionMetrics) return null;

    const { overall, aggregates, raw } = predictionMetrics;

    return (
      <div className={`${themeClasses.card} border rounded-lg p-6`}>
        <div className="flex items-center space-x-2 mb-4">
          <Activity className={`w-5 h-5 ${themeClasses.textMuted}`} />
          <h3 className={`text-lg font-semibold ${themeClasses.text}`}>
            Model Performance Metrics
          </h3>
        </div>
        
        <div className="space-y-6">
          {/* Overall Summary */}
          {overall && (
            <div className={`${themeClasses.bgSecondary} rounded-lg p-4`}>
              <h4 className={`font-medium ${themeClasses.text} mb-3 flex items-center`}>
                <TrendingUp className="w-4 h-4 mr-2" />
                Overall Performance
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className={`text-lg font-bold ${themeClasses.text}`}>
                    {overall.avg_mse_loss ? overall.avg_mse_loss.toFixed(6) : 'N/A'}
                  </div>
                  <div className={themeClasses.textMuted}>Average MSE Loss</div>
                </div>
                <div className="text-center">
                  <div className={`text-lg font-bold ${themeClasses.text}`}>
                    {overall.total_frames || 0}
                  </div>
                  <div className={themeClasses.textMuted}>Total Frames</div>
                </div>
                <div className="text-center">
                  <div className={`text-lg font-bold ${themeClasses.text}`}>
                    {overall.sequence_id !== undefined ? overall.sequence_id : 'N/A'}
                  </div>
                  <div className={themeClasses.textMuted}>Sequence ID</div>
                </div>
              </div>
            </div>
          )}

          {/* Time Step Performance */}
          {aggregates && (
            <div className={`${themeClasses.bgSecondary} rounded-lg p-4`}>
              <h4 className={`font-medium ${themeClasses.text} mb-3 flex items-center`}>
                <Clock className="w-4 h-4 mr-2" />
                Performance by Time Step (Overall Band Average)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {['t5', 't6', 't7'].map((timeStep, index) => {
                  const metrics = aggregates[timeStep];
                  if (!metrics) return null;
                  
                  return (
                    <div key={timeStep} className={`${themeClasses.bgTertiary} rounded p-3`}>
                      <div className={`text-sm font-medium ${themeClasses.text} mb-2 text-center`}>
                        T+{index + 1} ({timeStep.toUpperCase()})
                      </div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className={themeClasses.textMuted}>PSNR:</span>
                          <span className={`font-mono ${themeClasses.text}`}>
                            {metrics.avg_psnr.toFixed(2)} dB
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className={themeClasses.textMuted}>SSIM:</span>
                          <span className={`font-mono ${themeClasses.text}`}>
                            {metrics.avg_ssim.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className={themeClasses.textMuted}>MSE:</span>
                          <span className={`font-mono ${themeClasses.text}`}>
                            {metrics.avg_mse_loss.toFixed(6)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`min-h-screen ${themeClasses.bg} transition-colors duration-200`}>
      {/* Header */}
      <div className={`${themeClasses.bgTertiary} border-b ${themeClasses.border} p-4`}>
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center space-x-4">
            <Link 
              to="/" 
              className={`p-2 rounded-lg ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
              title="Home"
            >
              <Home className="w-5 h-5" />
            </Link>
            
            <div className="bg-[#C15F3C] p-3 pr-5 rounded-l-lg rounded-tr-4xl">
              <h1 className="text-base font-semibold text-white">
                Satellite Image Animation & Prediction
              </h1>
            </div>
            
            {!selectedDirectory && (
              <button
                onClick={handleDirectorySelect}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg border ${themeClasses.border} ${themeClasses.hover} ${themeClasses.textSecondary} transition-colors`}
              >
                <FolderOpen className="w-4 h-4" />
                <span>Select Directory</span>
              </button>
            )}

            {selectedDirectory && (
              <button
                onClick={handleDirectorySelect}
                className={`flex items-center space-x-2 px-3 py-2 text-xs rounded-lg border ${themeClasses.border} ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
              >
                <RefreshCw className="w-3 h-3" />
                <span>Change Directory</span>
              </button>
            )}
          </div>

          <button
            onClick={toggleTheme}
            className={`p-2 rounded-lg ${themeClasses.hover} ${themeClasses.textMuted} transition-colors`}
            title="Toggle theme"
          >
            {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {!selectedDirectory ? (
        // Directory selection screen
        <div className="flex items-center justify-center h-[80vh]">
          <div className={`${themeClasses.card} border rounded-xl p-8 max-w-md text-center`}>
            <Database className={`w-16 h-16 mx-auto mb-4 ${themeClasses.textMuted}`} />
            <h2 className={`text-xl font-semibold ${themeClasses.text} mb-2`}>
              Select Data Directory
            </h2>
            <p className={`${themeClasses.textMuted} mb-6`}>
              Choose a directory containing 6 band folders (IMG_VIS, IMG_MIR, IMG_SWIR, IMG_TIR1, IMG_TIR2, IMG_WV) with PNG images.
            </p>
            <button
              onClick={handleDirectorySelect}
              className="px-6 py-3 text-white rounded-lg transition-colors hover:opacity-90"
              style={{ backgroundColor: themeClasses.accent }}
            >
              <FolderOpen className="w-5 h-5 inline mr-2" />
              Browse Directory
            </button>
          </div>
        </div>
      ) : (
        // Main application
        <div className="max-w-7xl mx-auto p-4 space-y-6">
          {/* Time Window Display & Controls */}
          <div className={`${themeClasses.card} border rounded-lg p-4`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className={`text-lg font-semibold ${themeClasses.text}`}>
                Time Window Control
              </h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => navigateTimeWindow('left')}
                  disabled={currentTimeWindow[0] === 0}
                  className={`p-2 rounded ${themeClasses.button} transition-colors ${
                    currentTimeWindow[0] === 0 ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                  title="Previous window (Left Arrow)"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={handlePlayAnimation}
                  className="px-4 py-2 text-white rounded-lg transition-colors"
                  style={{ backgroundColor: themeClasses.accent }}
                >
                  {isAnimating ? (
                    <><Pause className="w-4 h-4 inline mr-2" />Stop</>
                  ) : (
                    <><Play className="w-4 h-4 inline mr-2" />Play Once</>
                  )}
                </button>
                <button
                  onClick={() => navigateTimeWindow('right')}
                  disabled={currentTimeWindow[3] >= getMaxFrames() - 1}
                  className={`p-2 rounded ${themeClasses.button} transition-colors ${
                    currentTimeWindow[3] >= getMaxFrames() - 1 ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                  title="Next window (Right Arrow)"
                >
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Enhanced Timeline with auto-scroll and date display */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className={themeClasses.textMuted}>Timeline ({getMaxFrames()} total frames)</span>
                <div className="flex flex-col items-end">
                  <span className={themeClasses.textMuted}>
                    Window: {currentTimeWindow[0] + 1}-{currentTimeWindow[3] + 1}
                  </span>
                  <span className={`text-xs flex gap-2 ${themeClasses.text} font-medium`}>
                    <Calendar className='size-4' /> {getWindowDateRange()}
                  </span>
                </div>
              </div>
              
              <div 
                id="timeline-container"
                className="flex items-center space-x-1 overflow-x-auto pb-2 scroll-smooth"
                style={{ scrollBehavior: 'smooth' }}
              >
                {bandData && Object.values(bandData)[0]?.map((image, index) => (
                  <div
                    key={index}
                    data-frame-index={index}
                    className={`flex-shrink-0 px-2 py-1 text-xs rounded transition-all cursor-pointer hover:scale-105 ${
                      currentTimeWindow.includes(index)
                        ? 'text-white shadow-md'
                        : `${themeClasses.bgSecondary} ${themeClasses.textMuted} hover:${themeClasses.hover}`
                    }`}
                    style={{
                      backgroundColor: currentTimeWindow.includes(index) ? themeClasses.accent : undefined
                    }}
                    onClick={() => {
                      // Allow clicking to center window on this frame
                      const newWindow = [
                        Math.max(0, index - 1),
                        Math.max(1, index),
                        Math.min(getMaxFrames() - 2, index + 1),
                        Math.min(getMaxFrames() - 1, index + 2)
                      ];
                      if (newWindow[3] - newWindow[0] === 3) {
                        setCurrentTimeWindow(newWindow);
                        // Auto-scroll to keep clicked frame visible
                        setTimeout(() => {
                          const clickedElement = document.querySelector(`[data-frame-index="${index}"]`);
                          if (clickedElement) {
                            clickedElement.scrollIntoView({ 
                              behavior: 'smooth', 
                              inline: 'center',
                              block: 'nearest'
                            });
                          }
                        }, 100);
                      }
                    }}
                    title={`Frame ${index + 1} - ${formatTimestamp(index)}`}
                  >
                    {index + 1}
                  </div>
                ))}
              </div>
            </div>

            {/* Updated: Current window detailed info with calculated dates */}
            <div className={`mt-3 p-3 ${themeClasses.bgSecondary} rounded-lg`}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                {currentTimeWindow.map((frameIndex, windowIndex) => {
                  const dateTime = calculateDateTime(frameIndex);
                  return (
                    <div key={windowIndex} className="text-center">
                      <div className={`font-semibold ${themeClasses.text}`}>
                        T{windowIndex === 3 ? '' : `-${3-windowIndex}`} (Frame {frameIndex + 1})
                      </div>
                      <div className={`text-xs ${themeClasses.textMuted} mt-1`}>
                        {dateTime.time}
                      </div>
                      <div className={`text-xs ${themeClasses.textMuted}`}>
                        {dateTime.date}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* FIXED: Band Image Previews with proper animation */}
          <div className={`${themeClasses.card} border rounded-lg p-4`}>
            <h3 className={`text-lg font-semibold ${themeClasses.text} mb-4`}>
              Band Previews - Current Time Window (Frames {currentTimeWindow[0] + 1}-{currentTimeWindow[3] + 1})
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {BANDS.map((bandName) => {
                // FIXED: Get all 4 images from current time window for this band in proper format for ImageAnimator
                const windowImages = currentTimeWindow
                  .map(frameIndex => {
                    const imageData = bandData[bandName]?.[frameIndex];
                    if (!imageData) return null;
                    
                    // Return in the format expected by ImageAnimator
                    return {
                      url: imageData.url,
                      filename: imageData.filename
                    };
                  })
                  .filter(Boolean); // Remove null values

                console.log(`Band ${bandName} - Window images:`, windowImages.length, 'images loaded');

                return (
                  <div key={bandName} className={`${themeClasses.bgSecondary} rounded-lg p-3 relative`}>
                    <div className={`text-xs font-medium ${themeClasses.text} mb-2 text-center`}>
                      {BAND_LABELS[bandName]}
                    </div>
                    <ImageAnimator
                      images={windowImages}
                      isPlaying={isAnimating}
                      frameRate={500} // 0.5 seconds per frame
                      onAnimationComplete={handleAnimationComplete}
                      altText={`${bandName} animation`}
                      className="relative"
                    />
                    {/* Band info */}
                    <div className={`mt-2 text-xs ${themeClasses.textMuted} text-center`}>
                      {windowImages.length} frames loaded
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Animation status */}
            <div className={`mt-4 text-sm ${themeClasses.textMuted} text-center`}>
              {isAnimating ? (
                <span className="flex items-center justify-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Playing animation through frames {currentTimeWindow[0] + 1}-{currentTimeWindow[3] + 1} (0.5s per frame)</span>
                </span>
              ) : (
                <span>Animation ready - click "Play Once" to animate through current window once</span>
              )}
            </div>
          </div>
          
          {/* Predict Button */}
          <div className="text-center">
            <button
              onClick={handlePredictFutureFrames}
              disabled={isPredicting || !bandData}
              className={`px-8 py-3 text-white rounded-lg font-medium transition-all transform hover:scale-105 ${
                isPredicting ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              style={{ backgroundColor: themeClasses.accent }}
            >
              {isPredicting ? (
                <>
                  <RefreshCw className="w-5 h-5 inline mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5 inline mr-2" />
                  Predict Future Frames
                </>
              )}
            </button>
          </div>

          {/* Performance Metrics */}
          {renderPerformanceMetrics()}

          {/* Prediction Results - ENHANCED VERSION */}
          {predictionResults && (
            <div className={`${themeClasses.card} border rounded-lg p-6`}>
              <div className="flex items-center justify-between mb-6">
                <h3 className={`text-xl font-semibold ${themeClasses.text}`}>
                  🔮 Prediction Results - Future Frames
                </h3>
                <select
                  value={selectedBand}
                  onChange={(e) => setSelectedBand(e.target.value)}
                  className={`px-4 py-2 rounded-lg ${themeClasses.input} border ${themeClasses.border}`}
                >
                  {BANDS.map(band => (
                    <option key={band} value={band}>{BAND_LABELS[band]}</option>
                  ))}
                </select>
              </div>

              {/* Prediction Summary */}
              <div className={`mb-6 p-4 ${themeClasses.bgSecondary} rounded-lg`}>
                <h4 className={`text-sm font-medium ${themeClasses.text} mb-3`}>Prediction Overview</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center">
                    <div className={`font-semibold ${themeClasses.text}`}>Input Window</div>
                    <div className={themeClasses.textMuted}>
                      Frames {predictionResults.inputWindow?.[0] + 1}-{predictionResults.inputWindow?.[3] + 1}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`font-semibold ${themeClasses.text}`}>Sequence</div>
                    <div className={themeClasses.textMuted}>
                      {predictionResults.sequenceFolder || 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`font-semibold ${themeClasses.text}`}>Predicted Frames</div>
                    <div className={themeClasses.textMuted}>
                      {predictionResults.summary?.totalPredictedFrames || 0}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`font-semibold ${themeClasses.text}`}>Generated</div>
                    <div className={themeClasses.textMuted}>
                      {predictionResults.metadata?.timestamp ? 
                        new Date(predictionResults.metadata.timestamp).toLocaleTimeString() : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* MAIN COMPARISON SECTION - Predicted vs Ground Truth */}
              <div className="space-y-8">
                {/* Section Headers */}
                <div className="text-center">
                  <h4 className={`text-lg font-medium ${themeClasses.text} mb-2`}>
                    Predicted vs Ground Truth Comparison - {BAND_LABELS[selectedBand]}
                  </h4>
                  <p className={`text-sm ${themeClasses.textMuted}`}>
                    Comparing AI predictions with actual future frames
                  </p>
                </div>

                {/* Time Step Labels */}
                <div className="grid grid-cols-3 gap-6">
                  <div className={`text-center font-medium ${themeClasses.text} p-2 ${themeClasses.bgSecondary} rounded`}>
                    T+1 (Next Frame)
                  </div>
                  <div className={`text-center font-medium ${themeClasses.text} p-2 ${themeClasses.bgSecondary} rounded`}>
                    T+2 (+2 Frames)
                  </div>
                  <div className={`text-center font-medium ${themeClasses.text} p-2 ${themeClasses.bgSecondary} rounded`}>
                    T+3 (+3 Frames)
                  </div>
                </div>

                {/* Predicted Frames Row */}
                <div>
                  <h4 className={`text-md font-medium ${themeClasses.text} mb-4 flex items-center`}>
                    <Brain className="w-5 h-5 mr-2 text-blue-500" />
                     AI Predicted Frames
                  </h4>
                  <div className="grid grid-cols-3 gap-6">
                    {['t5', 't6', 't7'].map((timeStep, index) => {
                      const predictedImageUrl = getPredictedImageUrl(timeStep, selectedBand);
                      
                      // Updated: Get metrics for this specific frame and band with new naming
                      const frameData = predictionResults?.predictedFrames?.find(f => f.timeStep === timeStep);
                      const bandMetrics = frameData?.bands?.[selectedBand]?.metrics;
                      
                      return (
                        <div key={timeStep} className={`${themeClasses.bgSecondary} rounded-lg p-4`}>
                          <div className={`text-sm ${themeClasses.textMuted} mb-3 text-center font-medium`}>
                            Predicted T+{index + 1}
                          </div>
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border-2 border-blue-500/50 mb-3 relative">
                            {predictedImageUrl ? (
                              <>
                                <img
                                  src={predictedImageUrl}
                                  alt={`Predicted ${timeStep}`}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    console.error('Failed to load predicted image:', predictedImageUrl);
                                    e.target.style.display = 'none';
                                    e.target.nextSibling.style.display = 'flex';
                                  }}
                                />
                                <div className="absolute top-2 left-2 bg-blue-500 text-white text-xs px-2 py-1 rounded">
                                  AI
                                </div>
                              </>
                            ) : null}
                            <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm" 
                                 style={{ display: predictedImageUrl ? 'none' : 'flex' }}>
                              <div className="text-center">
                                <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                <div>Loading Prediction...</div>
                                <div className="text-xs mt-1">T+{index + 1}</div>
                              </div>
                            </div>
                          </div>
                          
                          {/* Individual Frame Metrics */}
                          {bandMetrics && (
                            <div className={`text-xs ${themeClasses.bgTertiary} rounded-lg p-3 space-y-2`}>
                              <div className="text-center font-medium text-blue-600 dark:text-blue-400 mb-2">
                                Quality Metrics
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>PSNR:</span>
                                <span className={`font-mono ${themeClasses.text}`}>
                                  {bandMetrics.psnr.toFixed(2)} dB
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>SSIM:</span>
                                <span className={`font-mono ${themeClasses.text}`}>
                                  {bandMetrics.ssim.toFixed(4)}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>MSE:</span>
                                <span className={`font-mono ${themeClasses.text}`}>
                                  {bandMetrics.mse_loss.toFixed(6)}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Ground Truth Row */}
                <div>
                  <h4 className={`text-md font-medium ${themeClasses.text} mb-4 flex items-center`}>
                    <Database className="w-5 h-5 mr-2 text-green-500" />
                    Ground Truth Frames
                  </h4>
                  <div className="grid grid-cols-3 gap-6">
                    {[1, 2, 3].map(offset => {
                      const groundTruthImage = getGroundTruthImage(offset, selectedBand);
                      const frameIndex = currentTimeWindow[3] + offset;
                      
                      return (
                        <div key={offset} className={`${themeClasses.bgSecondary} rounded-lg p-4`}>
                          <div className={`text-sm ${themeClasses.textMuted} mb-3 text-center font-medium`}>
                             Ground Truth T+{offset}
                          </div>
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border-2 border-green-500/50 mb-3 relative">
                            {groundTruthImage ? (
                              <>
                                <img
                                  src={groundTruthImage.url}
                                  alt={`Ground truth T+${offset}`}
                                  className="w-full h-full object-cover"
                                />
                                <div className="absolute top-2 left-2 bg-green-500 text-white text-xs px-2 py-1 rounded">
                                  GT
                                </div>
                              </>
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
                                <div className="text-center">
                                  <Database className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                  <div>No Ground Truth</div>
                                  <div className="text-xs mt-1">Frame {frameIndex + 1}</div>
                                  <div className="text-xs mt-1">Beyond dataset</div>
                                </div>
                              </div>
                            )}
                          </div>
                          
                          {/* Ground Truth Info */}
                          {groundTruthImage && (
                            <div className={`text-xs ${themeClasses.bgTertiary} rounded-lg p-3 space-y-2`}>
                              <div className="text-center font-medium text-green-600 dark:text-green-400 mb-2">
                                Frame Details
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>Frame:</span>
                                <span className={`font-mono ${themeClasses.text}`}>
                                  #{frameIndex + 1}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>Time:</span>
                                <span className={`font-mono ${themeClasses.text}`}>
                                  {formatTimeOnly(frameIndex)}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>Date:</span>
                                <span className={`font-mono ${themeClasses.text} text-xs`}>
                                  {calculateDateTime(frameIndex).date}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className={themeClasses.textMuted}>File:</span>
                                <span className={`font-mono ${themeClasses.text} truncate`} title={groundTruthImage.filename}>
                                  {groundTruthImage.filename.slice(0, 12)}...
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

               
              </div>

              {/* Additional Information */}
              {predictionResults.metadata && (
                <div className={`mt-6 p-4 ${themeClasses.bgSecondary} rounded-lg`}>
                  <h4 className={`text-sm font-medium ${themeClasses.text} mb-3 flex items-center`}>
                    <Activity className="w-4 h-4 mr-2" />
                    Technical Details
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                    <div>
                      <span className={themeClasses.textMuted}>Total Bands:</span>
                      <span className={`ml-2 ${themeClasses.text} font-mono`}>
                        {predictionResults.metadata.totalBands}
                      </span>
                    </div>
                    <div>
                      <span className={themeClasses.textMuted}>Window Size:</span>
                      <span className={`ml-2 ${themeClasses.text} font-mono`}>
                        {predictionResults.metadata.windowSize}
                      </span>
                    </div>
                    <div>
                      <span className={themeClasses.textMuted}>Frames Found:</span>
                      <span className={`ml-2 ${themeClasses.text} font-mono`}>
                        {predictionResults.summary?.framesFound || 0}/3
                      </span>
                    </div>
                    <div>
                      <span className={themeClasses.textMuted}>Bands per Frame:</span>
                      <span className={`ml-2 ${themeClasses.text} font-mono`}>
                        {predictionResults.summary?.bandsPerFrame || 0}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SatelliteAnimationPage;