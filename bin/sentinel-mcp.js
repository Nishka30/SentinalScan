#!/usr/bin/env node

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const isWindows = process.platform === 'win32';
const mcpCmd = isWindows ? 'sentinel-mcp.exe' : 'sentinel-mcp';

// 1. Try to spawn from the global/user PATH first
function runFromPath() {
  const child = spawn(mcpCmd, process.argv.slice(2), {
    stdio: 'inherit',
    shell: isWindows // Windows needs shell to resolve cmd scripts in path
  });

  child.on('error', (err) => {
    if (err.code === 'ENOENT') {
      // Not found on PATH, try to auto-install or guide the user
      autoInstallOrExit();
    } else {
      console.error(`Failed to start Sentinel MCP server: ${err.message}`);
      process.exit(1);
    }
  });

  child.on('exit', (code) => {
    process.exit(code ?? 0);
  });
}

// 2. Try to auto-install sentinel-risk if not available
function autoInstallOrExit() {
  console.log('Sentinel CLI/MCP server was not found on your system PATH.');
  console.log('Attempting to install "sentinel-risk" via pip...');

  let pythonCmd = 'python';
  try {
    execSync('python3 --version', { stdio: 'ignore' });
    pythonCmd = 'python3';
  } catch (e) {
    try {
      execSync('python --version', { stdio: 'ignore' });
      pythonCmd = 'python';
    } catch (err) {
      console.error('\nError: Python is not installed or not in your PATH.');
      console.error('Please install Python 3.11+ and try again.');
      process.exit(1);
    }
  }

  try {
    console.log(`Running: ${pythonCmd} -m pip install --user --upgrade sentinel-risk`);
    execSync(`${pythonCmd} -m pip install --user --upgrade sentinel-risk`, { stdio: 'inherit' });
    console.log('\nInstallation completed successfully!');
    console.log('Restarting Sentinel MCP server...\n');

    // Run again since it is now installed
    runFromPath();
  } catch (err) {
    console.error('\nAuto-install failed. Please install it manually:');
    console.error('  pip install sentinel-risk');
    process.exit(1);
  }
}

runFromPath();
