apt-get update

apt-get install -y wget git unzip
DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 libglib2.0-0 libx264-dev ffmpeg
pip install -r requirements.txt