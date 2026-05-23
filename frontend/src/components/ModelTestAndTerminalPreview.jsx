import { useState, useEffect, useRef } from 'react';
import { Play, Terminal } from 'lucide-react';
import axiosInstance from '../libs/axios';

export default function ModelTestAndTerminalPreview({ testFolder, themeClasses = {} }) {
  const [logs, setLogs] = useState([]);
  const terminalRef = useRef(null);

  // Default theme classes if not provided
  const defaultTheme = {
    accent: '#C15F3C',
    card: 'bg-white dark:bg-stone-800',
    border: 'border-stone-200 dark:border-stone-700',
    bgSecondary: 'bg-stone-100 dark:bg-stone-800',
    bgTertiary: 'bg-stone-50 dark:bg-stone-700',
    text: 'text-stone-900 dark:text-stone-100',
    textMuted: 'text-stone-500 dark:text-stone-400'
  };

  const theme = { ...defaultTheme, ...themeClasses };

  const handleModelTesting = async () => {
    console.log(testFolder, 'Test folder path');
    try {
      const ws = new WebSocket('ws://localhost:3001'); // match backend port

      ws.onopen = async () => {
        console.log('✅ WebSocket connected. Starting model test...');
        await axiosInstance.post('/folder-path', {
          folderPath: testFolder,
        });
      };

      ws.onmessage = (event) => {
        const { type, content } = JSON.parse(event.data);
        setLogs(prev => [...prev, { type, content }]);
      };

      ws.onerror = (err) => console.error('WebSocket error:', err);
      ws.onclose = () => console.log('❌ WebSocket disconnected');
    } catch (err) {
      console.error('Axios Error:', err.message);
    }
  };

  // ⏬ Scroll to bottom when logs update
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-full space-y-4 h-[235px]">
      {/* Test Button with consistent styling */}
      <button
        onClick={handleModelTesting}
        disabled={!testFolder}
        className={`flex items-center space-x-2 px-4 py-2 text-sm rounded-lg font-medium transition-all ${
          testFolder
            ? 'text-white hover:opacity-90 transform hover:scale-105'
            : ' text-white opacity-80 cursor-not-allowed'
        }`}
        style={{ backgroundColor: testFolder ? theme.accent : '#666' }}
      >
        <Play className="w-4 h-4" />
        <span>Test Model</span>
      </button>

      {/* Terminal Container with consistent styling */}
      <div className={`${theme.card} border rounded overflow-hidden`}>
        {/* Terminal Header */}
        <div className={`flex items-center justify-between px-4 py-2 ${theme.bgSecondary} border-b ${theme.border}`}>
          <div className="flex items-center space-x-2">
            <Terminal className={`w-4 h-4 ${theme.textMuted}`} />
            <span className={`text-sm font-medium ${theme.text}`}>Terminal Output</span>
          </div>
          <div className="flex space-x-1">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
        </div>

        {/* Terminal Content */}
        <div
          ref={terminalRef}
          className={`${theme.bgTertiary} font-mono text-sm h-[180px] overflow-y-auto p-4`}
          style={{ 
            backgroundColor: theme.bgTertiary,
            color: theme.text
          }}
        >
          {logs.length === 0 ? (
            <div className={`${theme.textMuted} text-center py-8`}>
              <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Terminal output will appear here</p>
              <p className="text-xs mt-1">Click "Test Model" to start</p>
            </div>
          ) : (
            <pre className="whitespace-pre-wrap text-left">
              {logs.map((log, idx) => (
                <div
                  key={idx}
                  className={log.type === 'stderr' ? 'text-red-500' : 'text-white'}
                >
                  {log.content}
                </div>
              ))}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}