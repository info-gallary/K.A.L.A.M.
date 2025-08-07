import { spawn } from 'child_process';
import { getClient } from '../utils/websocket.js';

export const streamModelLogs = async (req, res) => {
  const { folderPath } = req.body;

  const client = getClient();


  if (!folderPath || typeof folderPath !== 'string' || !folderPath.trim()) {
    console.warn('❌ Invalid folderPath:', folderPath);
    return res.status(400).json({ error: 'folderPath is required and must be a non-empty string' });
  }

  console.log('📁 Folder path received:', folderPath);

  const py = spawn('python', ['-u', 'run_model.py'], {  //   const py = spawn('python', ['-u', 'run_model.py', folderPath], {

    env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
  });

  py.stdout.on('data', (data) => {
    const message = data.toString();

    console.log('📤 Python:', message);
    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'stdout',
        content: message
      }));
    }
  });

  py.stderr.on('data', (data) => {
    const message = data.toString();

    console.error('❌ Python Error:', message);

    // Send structured message with type
    if (client && client.readyState === 1) {
      client.send(JSON.stringify({
        type: 'stderr',
        content: message
      }));
    }
  });

  py.on('close', (code) => {
    console.log(`✅ Python script exited with code ${code}`);
    if (client) client.send(`✅ Model testing complete (exit code ${code})`);
  });


  res.status(200).json({ message: '✅ Python model started', data: { folderPath } });
}