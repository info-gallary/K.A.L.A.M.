import { useState, useEffect, useRef } from 'react';
import axiosInstance from '../libs/axios';

export default function ModelTestAndTerminalPreview({testFolder}) {
  const [logs, setLogs] = useState([]);
  const terminalRef = useRef(null);

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
    <div className="w-full max-w-3xl mx-auto mt-8">
      <button
        onClick={handleModelTesting}
        className="mb-4 px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded shadow"
      >
        ▶️ Test Model
      </button>
      <div
        ref={terminalRef}
        className="bg-black text-green-400 font-mono rounded p-4 h-[400px] overflow-y-auto shadow-inner border border-gray-700"
      >
        <pre className="whitespace-pre-wrap">
          {logs.map((log, idx) => (
            <div
              key={idx}
              className={log.type === 'stderr' ? 'text-red-500' : 'text-green-400'}
            >
              {log.content}
            </div>
          ))}
        </pre>
      </div>

    </div>
  );
}
