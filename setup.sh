#!/bin/bash

# Update system
apt-get update && apt-get upgrade -y

# Install Python dependencies
pip install torch transformers sentence-transformers fastapi uvicorn gunicorn faiss-gpu

# Install your package
pip install -e .

# Create a systemd service file for the application
cat > /etc/systemd/system/bertalign.service << EOL
[Unit]
Description=Bertalign FastAPI Application
After=network.target

[Service]
User=root
WorkingDirectory=/workspace
Environment="PATH=/usr/local/bin"
ExecStart=uvicorn app:app --host 0.0.0.0 --port 5000 --workers 1
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Start and enable the service
systemctl daemon-reload
systemctl start bertalign
systemctl enable bertalign