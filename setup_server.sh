#!/bin/bash
# ==============================================================================
# Universal Linux Setup Script for Tutor LMS AI Video Studio Worker
# Compatible with Ubuntu, Debian, and major Linux distributions.
# This script installs Python 3, FFmpeg, configures a virtual environment,
# and installs a systemd background service to keep the FastAPI engine running.
# ==============================================================================

# Ensure the script is run with sudo/root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this script with sudo or as root."
  exit 1
fi

echo "======================================================================"
echo " Starting Tutor LMS AI Video Studio Background Server Setup"
echo "======================================================================"

# 1. Update system package repository and install Python + FFmpeg
echo "--> 1. Installing System Dependencies (Python 3, FFmpeg, Pip)..."
if [ -f /etc/debian_version ]; then
    # Debian/Ubuntu systems
    apt-get update
    apt-get install -y python3 python3-pip python3-venv ffmpeg build-essential
elif [ -f /etc/redhat-release ]; then
    # RedHat/CentOS/Rocky systems
    dnf check-update
    dnf install -y python3 python3-pip ffmpeg-free gcc python3-devel
else
    echo "Warning: Unknown Linux distribution. Attempting standard package install..."
    apt-get install -y python3 python3-pip python3-venv ffmpeg || yum install -y python3 python3-pip ffmpeg
fi

# 2. Setup the application directory
APP_DIR="/opt/tutor-lms-video-worker"
echo "--> 2. Configuring Application Directory in $APP_DIR..."
mkdir -p "$APP_DIR"
cp -R . "$APP_DIR"
cd "$APP_DIR"

# 3. Setup Python Virtual Environment
echo "--> 3. Configuring Python 3 Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install requirements
echo "--> 4. Installing Python Dependencies (FastAPI, Edge-TTS, Pillow, boto3, etc.)..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create Systemd Service File
echo "--> 5. Configuring systemd Service for background auto-run..."
SERVICE_FILE="/etc/systemd/system/tutor-lms-video-worker.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Tutor LMS AI Video Studio Background Rendering Service
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=NOTEBOOKLM_REST_API_KEY=""

[Install]
WantedBy=multi-user.target
EOF

# 6. Enable and Start Service
echo "--> 6. Enabling and Starting tutor-lms-video-worker Service..."
systemctl daemon-reload
systemctl enable tutor-lms-video-worker.service
systemctl restart tutor-lms-video-worker.service

# Get Public/Local IP Address
IP_ADDR=$(hostname -I | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    IP_ADDR="YOUR_SERVER_IP"
fi

echo "======================================================================"
echo " Setup Completed Successfully!"
echo "======================================================================"
echo " Your Tutor LMS AI Video Studio Worker is now running in the background."
echo " "
echo " --> Server Endpoint URL: http://$IP_ADDR:8000"
echo " "
echo " Copy this URL and paste it into your WordPress Settings under:"
echo " Tutor LMS > Settings > Advanced > AI Video Settings"
echo "======================================================================"
