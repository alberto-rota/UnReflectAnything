import os
from flask import Flask, render_template_string, send_from_directory, request
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "HEAD", "OPTIONS"],
        "allow_headers": "*"
    }
})

# Configuration
DEMOS_DIR = "/demos"
MAIN_PORT = 60000
HOSTNAME = os.getenv('RERUN_HOSTNAME', 'localhost')

def discover_demo_files():
    """Discover all .rrd files and create URLs for them"""
    demos_path = Path(DEMOS_DIR)
    
    print(f"Looking for .rrd files in: {DEMOS_DIR}")
    print(f"Directory exists: {demos_path.exists()}")
    print(f"Using hostname: {HOSTNAME}")
    
    if demos_path.exists():
        print(f"Directory contents: {list(demos_path.iterdir())}")
    
    rrd_files = list(demos_path.glob("*.rrd"))
    
    print(f"Found {len(rrd_files)} demo files: {[f.name for f in rrd_files]}")
    
    demo_urls = {}
    for rrd_file in rrd_files:
        demo_name = rrd_file.name
        file_size = rrd_file.stat().st_size if rrd_file.exists() else 0
        
        # Determine protocol and port
        if HOSTNAME in ['localhost', '127.0.0.1']:
            protocol = 'http'
            port_suffix = f':{MAIN_PORT}'
        else:
            # Assume public hostname uses HTTPS (like Tailscale funnel)
            protocol = 'https'
            port_suffix = ''  # Most public services use standard ports
        
        rrd_url = f"{protocol}://{HOSTNAME}{port_suffix}/rrd/{demo_name}"
        
        # Check if there's a corresponding .rbl file
        rbl_file = rrd_file.with_suffix('.rbl')
        has_blueprint = rbl_file.exists()
        rbl_url = None
        if has_blueprint:
            rbl_url = f"{protocol}://{HOSTNAME}{port_suffix}/rbl/{rbl_file.name}"
        
        demo_urls[demo_name] = {
            'rrd_url': rrd_url,
            'rbl_url': rbl_url,
            'file_path': str(rrd_file),
            'file_size': file_size,
            'file_type': 'rrd',
            'has_blueprint': has_blueprint
        }
        print(f"Demo: {demo_name} -> {rrd_url} (size: {file_size} bytes, blueprint: {has_blueprint})")
    
    return demo_urls

def discover_blueprint_files():
    """Discover all .rbl files and create URLs for them"""
    demos_path = Path(DEMOS_DIR)
    
    print(f"Looking for .rbl files in: {DEMOS_DIR}")
    
    rbl_files = list(demos_path.glob("*.rbl"))
    
    print(f"Found {len(rbl_files)} blueprint files: {[f.name for f in rbl_files]}")
    
    blueprint_urls = {}
    for rbl_file in rbl_files:
        blueprint_name = rbl_file.name
        file_size = rbl_file.stat().st_size if rbl_file.exists() else 0
        
        # Determine protocol and port
        if HOSTNAME in ['localhost', '127.0.0.1']:
            protocol = 'http'
            port_suffix = f':{MAIN_PORT}'
        else:
            # Assume public hostname uses HTTPS (like Tailscale funnel)
            protocol = 'https'
            port_suffix = ''  # Most public services use standard ports
        
        rbl_url = f"{protocol}://{HOSTNAME}{port_suffix}/rbl/{blueprint_name}"
        
        # Check if there's a corresponding .rrd file
        rrd_file = rbl_file.with_suffix('.rrd')
        has_recording = rrd_file.exists()
        rrd_url = None
        if has_recording:
            rrd_url = f"{protocol}://{HOSTNAME}{port_suffix}/rrd/{rrd_file.name}"
        
        blueprint_urls[blueprint_name] = {
            'rbl_url': rbl_url,
            'rrd_url': rrd_url,
            'file_path': str(rbl_file),
            'file_size': file_size,
            'file_type': 'rbl',
            'has_recording': has_recording
        }
        print(f"Blueprint: {blueprint_name} -> {rbl_url} (size: {file_size} bytes, recording: {has_recording})")
    
    return blueprint_urls

# HTML template with modern design
EMBEDDED_VIEWERS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>UnReflectAnything Demos</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            display: flex;
            min-height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            overflow-y: auto;
            position: fixed;
            height: 100vh;
            z-index: 1000;
        }
        
        .sidebar h2 {
            color: #2c3e50;
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        
        .file-list {
            list-style: none;
        }
        
        .file-item {
            background: white;
            margin-bottom: 8px;
            border-radius: 25px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid #e1e8ed;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .file-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #3498db;
        }
        
        .file-item.active {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border-color: #3498db;
        }
        
        .file-name {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .file-size {
            font-size: 12px;
            opacity: 0.7;
        }
        
        .blueprint-indicator {
            display: inline-block;
            margin-right: 8px;
            font-size: 14px;
        }
        
        .blueprint-available {
            color: #27ae60;
        }
        
        .blueprint-unavailable {
            color: #e74c3c;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            margin-left: 280px;
            padding: 40px;
            overflow-y: auto;
        }
        
        .page-title {
            text-align: center;
            color: white;
            font-size: 36px;
            font-weight: 300;
            margin-bottom: 40px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .viewers-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .viewer-card {
            background: white;
            border-radius: 25px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }
        

        
        .viewer-header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 20px;
            position: relative;
        }
        
        .viewer-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .viewer-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .file-details {
            font-size: 13px;
            opacity: 0.9;
        }
        
        .blueprint-status {
            display: flex;
            align-items: center;
            font-size: 12px;
            padding: 4px 12px;
            border-radius: 25px;
            font-weight: 500;
        }
        
        .blueprint-available-status {
            background: rgba(39, 174, 96, 0.2);
            color: #27ae60;
            border: 1px solid rgba(39, 174, 96, 0.3);
        }
        
        .blueprint-unavailable-status {
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
            border: 1px solid rgba(231, 76, 60, 0.3);
        }
        
        .blueprint-instruction {
            font-size: 11px;
            opacity: 0.8;
            line-height: 1.4;
            margin-top: 8px;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 25px;
        }
        
        .viewer-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #27ae60, #229954);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
        }
        
        .btn-purple {
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            color: white;
        }
        
        .viewer-frame {
            width: 100%;
            height: 400px;
            border: none;
            background: #f8f9fa;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 400px;
            background: #f8f9fa;
            color: #666;
            font-style: italic;
        }
        
        .loading::before {
            content: "⏳";
            margin-right: 8px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .no-files {
            text-align: center;
            color: white;
            padding: 60px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 25px;
            backdrop-filter: blur(10px);
        }
        
        .no-files h2 {
            font-size: 24px;
            margin-bottom: 16px;
        }
        
        .no-files p {
            font-size: 16px;
            opacity: 0.8;
        }
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .viewers-grid {
                grid-template-columns: 1fr;
                max-width: 800px;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                height: auto;
                position: relative;
                margin-bottom: 20px;
            }
            
            .main-content {
                margin-left: 0;
                padding: 20px;
            }
            
            .container {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar -->
        <div class="sidebar">
            <h2>📁 Available Demos</h2>
            <ul class="file-list">
                {% for demo_name, demo_info in demos.items() %}
                <li class="file-item" onclick="openFullscreenWithBlueprint('{{ demo_info.rrd_url | urlencode }}', '{{ demo_info.rbl_url | urlencode if demo_info.has_blueprint else '' }}')">
                    <div class="file-name">
                        <span class="blueprint-indicator {% if demo_info.has_blueprint %}blueprint-available{% else %}blueprint-unavailable{% endif %}">
                            {% if demo_info.has_blueprint %}🎨{% else %}📊{% endif %}
                        </span>
                        {{ demo_name.replace('.rrd', '').upper() }}
                    </div>
                    <div class="file-size">{{ "%.1f"|format(demo_info.file_size / 1024 / 1024) }} MB</div>
                </li>
                {% endfor %}
            </ul>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <h1 class="page-title">UnReflectAnything Demos</h1>
            
            {% if demos %}
            <div class="viewers-grid">
                {% for demo_name, demo_info in demos.items() %}
                <div class="viewer-card" id="viewer-{{ demo_name }}">
                    <div class="viewer-header">
                        <div class="viewer-title">{{ demo_name.replace('.rrd', '').upper() }}</div>
                        <div class="viewer-info">
                            <div class="file-details">{{ "%.1f"|format(demo_info.file_size / 1024 / 1024) }} MB</div>
                            <div class="blueprint-status {% if demo_info.has_blueprint %}blueprint-available-status{% else %}blueprint-unavailable-status{% endif %}">
                                {% if demo_info.has_blueprint %}
                                🎨 Blueprint Available
                                {% else %}
                                ⚠️ No Blueprint
                                {% endif %}
                            </div>
                        </div>
                        
                        {% if demo_info.has_blueprint %}
                        <div class="blueprint-instruction">
                            💡 For optimal layout: Download the blueprint file and drag & drop it onto the viewer
                        </div>
                        {% endif %}
                        
                        <div class="viewer-actions">
                            <button class="btn btn-primary" onclick="openFullscreenWithBlueprint('{{ demo_info.rrd_url | urlencode }}', '{{ demo_info.rbl_url | urlencode if demo_info.has_blueprint else '' }}')">
                                🔍 Fullscreen
                            </button>
                            <button class="btn btn-success" onclick="downloadFile('{{ demo_info.rrd_url }}', '{{ demo_name }}')">
                                📥 Download RRD
                            </button>
                            {% if demo_info.has_blueprint %}
                            <button class="btn btn-purple" onclick="downloadFile('{{ demo_info.rbl_url }}', '{{ demo_name.replace('.rrd', '.rbl') }}')">
                                🎨 Download Blueprint
                            </button>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div class="loading">Loading Rerun viewer...</div>
                    <iframe 
                        class="viewer-frame" 
                        src="https://app.rerun.io/version/0.23.1/?url={{ demo_info.rrd_url | urlencode }}{% if demo_info.has_blueprint %}&blueprint-url={{ demo_info.rbl_url | urlencode }}{% endif %}"
                        title="{{ demo_name }} Rerun Viewer"
                        loading="lazy"
                        onload="this.previousElementSibling.style.display='none'"
                        onerror="this.previousElementSibling.textContent='Failed to load viewer - check if URLs are publicly accessible'"
                        allow="clipboard-write; web-share"
                        sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-downloads">
                    </iframe>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="no-files">
                <h2>📂 No Demo Files Found</h2>
                <p>No .rrd files found in the /demos directory.</p>
            </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function openFullscreenWithBlueprint(rrdUrl, rblUrl) {
            let fullUrl = 'https://app.rerun.io/version/0.23.1/?';
            if (rrdUrl && rrdUrl !== '') {
                fullUrl += `url=${rrdUrl}`;
                if (rblUrl && rblUrl !== '') {
                    fullUrl += `&blueprint-url=${rblUrl}`;
                }
            } else if (rblUrl && rblUrl !== '') {
                fullUrl += `blueprint-url=${rblUrl}`;
            }
            window.open(fullUrl, '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes');
        }
        
        function downloadFile(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        // Add smooth scrolling to the page
        document.addEventListener('DOMContentLoaded', function() {
            // Enable smooth scrolling for the main content
            const mainContent = document.querySelector('.main-content');
            mainContent.style.scrollBehavior = 'smooth';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def embedded_viewers():
    """Serve the page with all embedded viewers"""
    demos = discover_demo_files()
    blueprints = discover_blueprint_files()
    return render_template_string(EMBEDDED_VIEWERS_TEMPLATE, demos=demos, blueprints=blueprints, hostname=HOSTNAME)

@app.route('/rrd/<filename>')
def serve_rrd_file(filename):
    """Serve .rrd files directly with explicit CORS headers"""
    print(f"Serving RRD file: {filename}")
    print(f"Request from: {request.remote_addr}")
    print(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
    print(f"Origin: {request.headers.get('Origin', 'No origin')}")
    
    file_path = Path(DEMOS_DIR) / filename
    print(f"File exists: {file_path.exists()}")
    
    if file_path.exists():
        print(f"File size: {file_path.stat().st_size} bytes")
    
    try:
        response = send_from_directory(DEMOS_DIR, filename, as_attachment=False, 
                                     mimetype='application/octet-stream')
        
        # Explicitly set CORS headers (belt and suspenders approach)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        
        print(f"Response headers: {dict(response.headers)}")
        return response
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        return f"RRD file '{filename}' not found", 404

@app.route('/rrd/<filename>', methods=['OPTIONS'])
def handle_rrd_options(filename):
    """Handle preflight OPTIONS requests for RRD files"""
    print(f"OPTIONS request for: {filename}")
    print(f"Origin: {request.headers.get('Origin', 'No origin')}")
    
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Max-Age'] = '86400'
    
    return response

@app.route('/rbl/<filename>')
def serve_rbl_file(filename):
    """Serve .rbl files directly with explicit CORS headers"""
    print(f"Serving RBL file: {filename}")
    print(f"Request from: {request.remote_addr}")
    print(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
    print(f"Origin: {request.headers.get('Origin', 'No origin')}")
    
    file_path = Path(DEMOS_DIR) / filename
    print(f"File exists: {file_path.exists()}")
    
    if file_path.exists():
        print(f"File size: {file_path.stat().st_size} bytes")
    
    try:
        response = send_from_directory(DEMOS_DIR, filename, as_attachment=False, 
                                     mimetype='application/octet-stream')
        
        # Explicitly set CORS headers (belt and suspenders approach)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        
        print(f"Response headers: {dict(response.headers)}")
        return response
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        return f"RBL file '{filename}' not found", 404

@app.route('/rbl/<filename>', methods=['OPTIONS'])
def handle_rbl_options(filename):
    """Handle preflight OPTIONS requests for RBL files"""
    print(f"OPTIONS request for: {filename}")
    print(f"Origin: {request.headers.get('Origin', 'No origin')}")
    
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Max-Age'] = '86400'
    
    return response

@app.route('/health')
def health_check():
    """Health check endpoint"""
    demos = discover_demo_files()
    blueprints = discover_blueprint_files()
    return {
        "status": "ok", 
        "demos": len(demos), 
        "blueprints": len(blueprints),
        "hostname": HOSTNAME
    }

if __name__ == '__main__':
    print(f"Starting demo server on port {MAIN_PORT}")
    print(f"Using hostname: {HOSTNAME}")
    print("CORS enabled for all origins")
    print("Supporting both .rrd (recording) and .rbl (blueprint) files")
    
    app.run(host='0.0.0.0', port=MAIN_PORT, debug=False)