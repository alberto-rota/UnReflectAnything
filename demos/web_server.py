import os
import subprocess
import time
import socket
import signal
import sys
from flask import Flask, render_template_string
import glob
import threading

app = Flask(__name__)

# Global storage for running processes and their URLs
running_demos = {}
demo_processes = []

def find_free_port(start_port=60001):
    """Find a free port starting from start_port"""
    port = start_port
    while port <= 60200:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("No free ports available in range 60001-60200")

def start_rerun_viewer(demo_file, ws_port, web_port):
    """Start rerun viewer for a specific demo file"""
    cmd = [
        'python', '-m', 'rerun', '--web-viewer', f'/demos/{demo_file}',
        '--port', str(ws_port), '--web-viewer-port', str(web_port), '--bind', '0.0.0.0'
    ]
    
    print(f"Starting rerun for {demo_file} on ws_port={ws_port}, web_port={web_port}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def wait_for_port(port, timeout=30):
    """Wait for a port to be available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except:
            pass
        time.sleep(0.5)
    return False

def initialize_demos():
    """Initialize all demo viewers at startup"""
    global running_demos, demo_processes
    
    demos_path = '/demos'
    if not os.path.exists(demos_path):
        print("Demos directory not found!")
        return
    
    demo_files = [os.path.basename(f) for f in glob.glob(os.path.join(demos_path, '*.rrd'))]
    demo_files.sort()
    
    if not demo_files:
        print("No .rrd files found in /demos directory")
        return
    
    print(f"Found {len(demo_files)} demo files: {demo_files}")
    
    current_port = 60001
    
    for demo_file in demo_files:
        try:
            # Use consecutive ports for each demo (ws_port, web_port)
            ws_port = current_port
            web_port = current_port + 1
            current_port += 2
            
            if web_port > 60200:
                print(f"Reached port limit, skipping {demo_file}")
                break
            
            # Start the rerun process
            process = start_rerun_viewer(demo_file, ws_port, web_port)
            demo_processes.append(process)
            
            # Wait for the service to be ready
            print(f"Waiting for {demo_file} to start on port {web_port}...")
            if wait_for_port(web_port, timeout=15):
                # Generate the direct URL to the web viewer
                demo_name = demo_file.replace('.rrd', '')
                url = f"http://localhost:{web_port}/?url=rerun%2Bws://localhost:{ws_port}"
                running_demos[demo_name] = {
                    'file': demo_file,
                    'url': url,
                    'web_port': web_port,
                    'ws_port': ws_port,
                    'process': process
                }
                print(f"✅ {demo_file} started successfully on {url}")
            else:
                print(f"❌ Failed to start {demo_file} - timeout waiting for port {web_port}")
                process.terminate()
                demo_processes.remove(process)
                
        except Exception as e:
            print(f"❌ Error starting {demo_file}: {e}")
    
    print(f"Successfully started {len(running_demos)} demos")

def cleanup_processes():
    """Clean up all running processes"""
    print("Cleaning up processes...")
    for process in demo_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, shutting down...")
    cleanup_processes()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Rerun Demo Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .demo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
        .demo-item { 
            background: #007bff; color: white; padding: 15px; text-align: center; 
            border-radius: 5px; text-decoration: none; transition: background-color 0.3s;
            display: block;
        }
        .demo-item:hover { background: #0056b3; text-decoration: none; color: white; }
        .status { margin-top: 20px; padding: 15px; border-radius: 5px; background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .no-demos { text-align: center; color: #666; margin-top: 30px; }
        .demo-info { font-size: 12px; margin-top: 5px; opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Rerun Demo Viewer</h1>
        
        {% if demos %}
        <div class="status">
            ✅ {{ demos|length }} demo(s) ready to view
        </div>
        
        <div class="demo-grid">
            {% for demo_name, demo_info in demos.items() %}
            <a href="{{ demo_info.url }}" target="_blank" class="demo-item">
                {{ demo_name }}
                <div class="demo-info">Port: {{ demo_info.web_port }}</div>
            </a>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-demos">
            <h3>No demos available</h3>
            <p>Place .rrd files in the /demos directory and restart the container.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, demos=running_demos)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'demos_count': len(running_demos),
        'demos': list(running_demos.keys())
    }

if __name__ == '__main__':
    print("Starting Rerun Demo Server...")
    
    # Initialize demos in a separate thread to not block Flask startup
    def init_demos_async():
        time.sleep(1)  # Give Flask a moment to start
        initialize_demos()
    
    init_thread = threading.Thread(target=init_demos_async, daemon=True)
    init_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    finally:
        cleanup_processes()