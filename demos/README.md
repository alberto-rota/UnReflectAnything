# UnReflectAnything Demos

This directory contains the demo web server and demo data for UnReflectAnything.

## Structure

```
demos/
├── Dockerfile.demos          # Docker configuration for the demo server
├── web_server.py            # Flask web server for serving demos
├── demo_data/               # Demo files (mounted at runtime)
│   ├── *.rrd               # Rerun recording files
│   └── *.rbl               # Rerun blueprint files
└── README.md               # This file
```

## Demo Data

The `demo_data/` directory contains:
- **RRD files (.rrd)**: Rerun recording files containing the actual data
- **RBL files (.rbl)**: Rerun blueprint files defining viewer layouts

## Running the Demo Server

### Using Docker Compose (Recommended)

```bash
# Start the demo server
docker compose up demos

# Access the web interface at http://localhost:60000
```

### Manual Setup

```bash
# Install dependencies
pip install flask flask-cors

# Run the server
cd demos
python web_server.py
```

## Adding New Demos

1. Place your `.rrd` and `.rbl` files in the `demo_data/` directory
2. Restart the container: `docker compose restart demos`
3. The new demos will automatically appear in the web interface

## Configuration

- **Port**: 60000 (configurable in docker-compose.yaml)
- **Hostname**: Set via `RERUN_HOSTNAME` environment variable
- **Data Directory**: `/demos` (mounted from `./demos/demo_data`)

## Features

- **Modern Web Interface**: Beautiful, responsive design
- **Sidebar Navigation**: Quick access to all available demos
- **Fullscreen Viewing**: Click any demo to open in fullscreen
- **Blueprint Support**: Automatic detection and loading of layout files
- **File Downloads**: Download RRD and RBL files directly
- **Responsive Design**: Works on desktop and mobile devices

## Troubleshooting

- **Files not appearing**: Ensure files are in `demo_data/` directory
- **CORS issues**: Set `RERUN_HOSTNAME` to your public domain
- **Port conflicts**: Change port in docker-compose.yaml
