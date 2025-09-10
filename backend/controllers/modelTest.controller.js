import { spawn } from 'child_process';
import { getClient } from '../utils/websocket.js';
import path from 'path';
import fs from 'fs';

// ... (keep the existing streamModelLogs function as is)

export const streamModelLogs = async (req, res) => {
  const { folderPath } = req.body;
  const client = getClient();

  if (!folderPath || typeof folderPath !== 'string' || !folderPath.trim()) {
    console.warn('❌ Invalid folderPath:', folderPath);
    return res.status(400).json({ error: 'folderPath is required and must be a non-empty string' });
  }

  console.log('📁 Folder path received:', folderPath);

  // Path to your Python script
  const pythonScriptPath = "d:\\Hackathon\\ISRO\\pre_final\\test1.py";
  
  // Check if Python script exists
  if (!fs.existsSync(pythonScriptPath)) {
    const errorMsg = `❌ Python script not found: ${pythonScriptPath}`;
    console.error(errorMsg);
    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'error',
        content: errorMsg
      }));
    }
    return res.status(404).json({ error: 'Python script not found' });
  }

  // Send start notification
  if (client && client.readyState === 1) {
    client.send(JSON.stringify({
      type: 'info',
      content: '🚀 Starting Python model validation...\n'
    }));
  }

  // Enhanced spawn options
  const spawnOptions = {
    env: { 
      ...process.env, 
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',  // Ensure real-time output
      PYTHONDONTWRITEBYTECODE: '1'  // Avoid .pyc files
    },
    cwd: path.dirname(pythonScriptPath),  // Set working directory
    stdio: ['pipe', 'pipe', 'pipe']
  };

  const py = spawn('python', ['-u', pythonScriptPath], spawnOptions);

  // Track process state
  let processStarted = false;
  let processCompleted = false;

  // Handle process startup
  py.on('spawn', () => {
    processStarted = true;
    console.log('🐍 Python process started successfully');
    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'info',
        content: '🐍 Python process started successfully\n'
      }));
    }
  });

  // Handle stdout (normal output)
  py.stdout.on('data', (data) => {
    const message = data.toString();
    console.log('📤 Python stdout:', message);
    
    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'stdout',
        content: message
      }));
    }
  });

  // Handle stderr (errors and warnings)
  py.stderr.on('data', (data) => {
    const message = data.toString();
    console.error('⚠️ Python stderr:', message);

    // Determine if this is a critical error or just a warning
    const isCriticalError = message.includes('Traceback') || 
                           message.includes('ImportError') || 
                           message.includes('ModuleNotFoundError') ||
                           message.includes('FileNotFoundError');

    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: isCriticalError ? 'error' : 'warning',
        content: message
      }));
    }

    // If it's a critical error, suggest solutions
    if (isCriticalError) {
      let suggestion = '';
      
      if (message.includes('safe_globals')) {
        suggestion = '💡 PyTorch version compatibility issue detected. Solutions:\n' +
                    '   1. Update PyTorch: pip install torch --upgrade\n' +
                    '   2. Or fix imports in your Python files\n' +
                    '   3. Check the corrected code provided\n';
      } else if (message.includes('ImportError') || message.includes('ModuleNotFoundError')) {
        const moduleName = message.match(/No module named '(\w+)'/)?.[1] || 'unknown';
        suggestion = `💡 Missing Python module detected: ${moduleName}\n` +
                    `   Try: pip install ${moduleName}\n` +
                    '   Or check your Python environment\n';
      } else if (message.includes('FileNotFoundError')) {
        suggestion = '💡 File not found error detected.\n' +
                    '   Check if all required files exist in the correct paths\n';
      }

      if (suggestion && client && client.readyState === 1) {
        client.send(JSON.stringify({
          type: 'suggestion',
          content: suggestion
        }));
      }
    }
  });

  // Handle process completion
  py.on('close', (code) => {
    processCompleted = true;
    const message = `🏁 Python script completed with exit code: ${code}\n`;
    console.log(message);

    if (client && client.readyState === 1) {
      let resultType = 'success';
      let resultMessage = message;

      if (code === 0) {
        resultMessage += '✅ Process completed successfully!';
      } else {
        resultType = 'error';
        resultMessage += `❌ Process failed (exit code: ${code})`;
      }

      client.send(JSON.stringify({
        type: resultType,
        content: resultMessage
      }));

      // Send final summary
      client.send(JSON.stringify({
        type: 'completion',
        content: {
          exitCode: code,
          success: code === 0,
          processStarted: processStarted,
          message: code === 0 ? 'Validation completed successfully!' : 'Validation failed - check errors above'
        }
      }));
    }
  });

  // Handle process errors (spawn failures)
  py.on('error', (error) => {
    const errorMessage = `💥 Failed to start Python process: ${error.message}\n`;
    console.error(errorMessage);

    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'error',
        content: errorMessage
      }));

      // Send troubleshooting suggestions
      const suggestions = '🔧 Troubleshooting suggestions:\n' +
                         '   1. Check if Python is installed and in PATH\n' +
                         '   2. Verify the Python script path is correct\n' +
                         '   3. Check file permissions\n' +
                         '   4. Try running the script manually first\n';
      
      client.send(JSON.stringify({
        type: 'suggestion',
        content: suggestions
      }));
    }
  });

  // Timeout handling (optional - 30 minutes timeout)
  const timeout = setTimeout(() => {
    if (!processCompleted) {
      console.warn('⏰ Python process timeout - killing process');
      py.kill('SIGTERM');
      
      if (client && client.readyState === 1) {
        client.send(JSON.stringify({
          type: 'warning',
          content: '⏰ Process timed out after 30 minutes and was terminated\n'
        }));
      }
    }
  }, 30 * 60 * 1000); // 30 minutes

  // Clean up timeout when process completes
  py.on('close', () => {
    clearTimeout(timeout);
  });

  // Send immediate response to client
  res.status(200).json({ 
    message: '✅ Python model validation started', 
    data: { 
      folderPath,
      scriptPath: pythonScriptPath,
      processStarted: true
    } 
  });
};

export const predictFutureFrames = async (req, res) => {
  const { timeWindow, selectedDirectory, bands, windowSize } = req.body;

  
  // Validate input parameters
  if (!selectedDirectory || !bands || !Array.isArray(bands)) {
    return res.status(400).json({ 
      error: 'Missing required parameters: selectedDirectory and bands are required' 
    });
  }

  if (!timeWindow || !Array.isArray(timeWindow) || timeWindow.length < 4) {
    return res.status(400).json({ 
      error: 'timeWindow must be an array with at least 4 frame indices' 
    });
  }

  console.log('🔮 Prediction request received:', {
    timeWindow,
    selectedDirectory,
    bands: bands.length,
    windowSize
  });

  try {
    // Calculate sequence number based on the first frame index
    // For timeWindow [0,1,2,3] -> sequence_0000
    // For timeWindow [1,2,3,4] -> sequence_0001
    const sequenceNumber = timeWindow[0];
    const sequenceFolderName = `sequence_${sequenceNumber.toString().padStart(4, '0')}`;
    
    // Path to the test output directory
    const testOutputPath = "D:\\Hackathon\\ISRO\\pre_final\\inference_outputs\\test"; // Adjust this path as needed
    const sequencePath = path.join(testOutputPath, sequenceFolderName);

    console.log('📁 Looking for sequence folder:', sequencePath);

    // Check if sequence folder exists
    if (!fs.existsSync(sequencePath)) {
      return res.status(404).json({ 
        error: `Predicted sequence folder not found: ${sequenceFolderName}`,
        path: sequencePath
      });
    }

    // Read performance.json
    const performanceJsonPath = path.join(sequencePath, 'performance.json');
    let performanceData = null;
    
    if (fs.existsSync(performanceJsonPath)) {
      try {
        const performanceContent = fs.readFileSync(performanceJsonPath, 'utf8');
        performanceData = JSON.parse(performanceContent);
      } catch (error) {
        console.warn('⚠️ Failed to parse performance.json:', error.message);
      }
    }

    // Get all predicted image files
    const files = fs.readdirSync(sequencePath);
    const imageFiles = files.filter(file => file.endsWith('.png'));

    // NEW: Updated band name mapping to match your new naming convention
    const BAND_NAME_MAP = {
      'IMG_VIS': 'IMG_VIS',
      'IMG_MIR': 'IMG_MIR', 
      'IMG_SWIR': 'IMG_SWIR',
      'IMG_TIR1': 'IMG_TIR1',
      'IMG_TIR2': 'IMG_TIR2',
      'IMG_WV': 'IMG_WV'
    };

    // Organize files by time step and band with new naming convention
    const organizedFrames = {
      t5: {},
      t6: {},
      t7: {}
    };

    // Process each image file with updated regex pattern
    imageFiles.forEach(filename => {
      // Updated regex to match: predicted_t5_IMG_VIS.png, predicted_t6_IMG_MIR.png, etc.
      const match = filename.match(/predicted_(t\d+)_(IMG_\w+)\.png/);
      if (match) {
        const timeStep = match[1];  // t5, t6, t7
        const bandName = match[2];  // IMG_VIS, IMG_MIR, etc.
        
        if (!organizedFrames[timeStep]) {
          organizedFrames[timeStep] = {};
        }
        
        organizedFrames[timeStep][bandName] = {
          filename: filename,
          path: path.join(sequencePath, filename),
          relativePath: `${sequenceFolderName}/${filename}`
        };
      }
    });

    // Prepare response with next 3 predicted frames (t5, t6, t7)
    const predictedFrames = [];
    
    ['t5', 't6', 't7'].forEach((timeStep, index) => {
      const frameData = {
        timeStep: timeStep,
        frameIndex: timeWindow[0] + 4 + index, // Next 3 frames after input window
        sequenceFolder: sequenceFolderName,
        bands: {}
      };

      // Add band data for requested bands using new naming convention
      Object.keys(BAND_NAME_MAP).forEach(bandName => {
        if (organizedFrames[timeStep] && organizedFrames[timeStep][bandName]) {
          const bandInfo = organizedFrames[timeStep][bandName];
          
          // Find performance metrics for this specific frame from performance.json
          let frameMetrics = null;
          if (performanceData && performanceData.frames) {
            // Updated frame key format: T5_IMG_VIS, T6_IMG_MIR, etc.
            const frameKey = `${timeStep.toUpperCase()}_${bandName}`;
            frameMetrics = performanceData.frames.find(f => f.frame === frameKey);
          }

          frameData.bands[bandName] = {
            filename: bandInfo.filename,
            relativePath: bandInfo.relativePath,
            absolutePath: bandInfo.path,
            metrics: frameMetrics ? {
              psnr: frameMetrics.psnr,
              ssim: frameMetrics.ssim,
              mse_loss: frameMetrics.mse_loss
            } : null
          };
        }
      });

      predictedFrames.push(frameData);
    });

    // Calculate aggregate metrics by time step and band with new naming
    const aggregateMetrics = {};
    if (performanceData && performanceData.frames) {
      // Group by time step
      ['T5', 'T6', 'T7'].forEach(timeStep => {
        const timeStepFrames = performanceData.frames.filter(f => f.frame.startsWith(timeStep));
        if (timeStepFrames.length > 0) {
          aggregateMetrics[timeStep.toLowerCase()] = {
            avg_psnr: timeStepFrames.reduce((sum, f) => sum + f.psnr, 0) / timeStepFrames.length,
            avg_ssim: timeStepFrames.reduce((sum, f) => sum + f.ssim, 0) / timeStepFrames.length,
            avg_mse_loss: timeStepFrames.reduce((sum, f) => sum + f.mse_loss, 0) / timeStepFrames.length,
            frame_count: timeStepFrames.length
          };
        }
      });

      // Group by band with new naming convention
      Object.keys(BAND_NAME_MAP).forEach(bandName => {
        const bandFrames = performanceData.frames.filter(f => f.frame.endsWith(`_${bandName}`));
        if (bandFrames.length > 0) {
          aggregateMetrics[bandName] = {
            avg_psnr: bandFrames.reduce((sum, f) => sum + f.psnr, 0) / bandFrames.length,
            avg_ssim: bandFrames.reduce((sum, f) => sum + f.ssim, 0) / bandFrames.length,
            avg_mse_loss: bandFrames.reduce((sum, f) => sum + f.mse_loss, 0) / bandFrames.length,
            frame_count: bandFrames.length
          };
        }
      });
    }

    // Prepare final response
    const responseData = {
      inputWindow: timeWindow,
      sequenceNumber: sequenceNumber,
      sequenceFolder: sequenceFolderName,
      predictedFrames: predictedFrames,
      performance: {
        raw: performanceData, // Original performance.json data
        aggregates: aggregateMetrics, // Calculated aggregates
        overall: {
          avg_mse_loss: performanceData?.avg_mse_loss || null,
          total_frames: performanceData?.frames?.length || 0,
          sequence_id: performanceData?.sequence_id || sequenceNumber
        }
      },
      metadata: {
        totalBands: Object.keys(BAND_NAME_MAP).length,
        requestedBands: Object.keys(BAND_NAME_MAP),
        windowSize: windowSize,
        testOutputPath: testOutputPath,
        timestamp: new Date().toISOString()
      },
      summary: {
        totalPredictedFrames: 3,
        framesFound: predictedFrames.length,
        bandsPerFrame: Object.keys(BAND_NAME_MAP).length,
        metricsAvailable: performanceData !== null
      }
    };

    console.log('✅ Prediction data prepared successfully');
    
    res.status(200).json({ 
      success: true,
      message: 'Predicted frames data retrieved successfully', 
      data: responseData
    });

  } catch (error) {
    console.error('❌ Error retrieving prediction data:', error);
    
    res.status(500).json({ 
      success: false,
      error: 'Failed to retrieve predicted frames data',
      details: error.message,
      timeWindow: timeWindow
    });
  }
};

// Helper function to get all available sequences
export const getAvailableSequences = async (req, res) => {
  try {
    const testOutputPath = "D:\\Hackathon\\ISRO\\pre_final\\inference_outputs\\test";
    
    if (!fs.existsSync(testOutputPath)) {
      return res.status(404).json({ 
        error: 'Test output directory not found',
        path: testOutputPath
      });
    }

    const folders = fs.readdirSync(testOutputPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory() && dirent.name.startsWith('sequence_'))
      .map(dirent => dirent.name)
      .sort();

    res.status(200).json({
      success: true,
      message: 'Available sequences retrieved successfully',
      data: {
        sequences: folders,
        count: folders.length,
        testOutputPath: testOutputPath
      }
    });

  } catch (error) {
    console.error('❌ Error getting available sequences:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to get available sequences',
      details: error.message
    });
  }
};