#!/bin/bash

# Update system
apt-get update && apt-get upgrade -y

# Install Python dependencies
pip install torch transformers sentence-transformers flask gunicorn faiss-gpu

# Install your package
pip install -e .

# Create a systemd service file for the application
cat > /etc/systemd/system/bertalign.service << EOL
[Unit]
Description=Bertalign Flask Application
After=network.target

[Service]
User=root
WorkingDirectory=/workspace
Environment="PATH=/usr/local/bin"
ExecStart=gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 300 app:app

[Install]
WantedBy=multi-user.target
EOL

# Start and enable the service
systemctl daemon-reload
systemctl start bertalign
systemctl enable bertalign