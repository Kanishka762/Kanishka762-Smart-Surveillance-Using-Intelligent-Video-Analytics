#!/bin/bash

if command -v conda &> /dev/null; then
    echo "Conda is installed. Continuing with the process..."
    # Add your package installation commands here
else
    echo "Conda is not installed. Please install Conda and run the script again."
    exit 1  # Exit the script with a non-zero status code
fi

# conda create -n deepstream_v1 python=3.8 
# conda activate deepstream_v1


sudo apt install \
libssl1.1 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev \
libjsoncpp-dev \
protobuf-compiler \
gcc \
make \
git 

sudo apt-get install gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
sudo apt-get install libgstreamer1.0-0 gstreamer1.0-dev gstreamer1.0-tools gstreamer1.0-doc


sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-toolkit-11-8

sudo apt-get install libnvinfer8=8.5.2-1+cuda11.8 libnvinfer-plugin8=8.5.2-1+cuda11.8 libnvparsers8=8.5.2-1+cuda11.8 \
libnvonnxparsers8=8.5.2-1+cuda11.8 libnvinfer-bin=8.5.2-1+cuda11.8 libnvinfer-dev=8.5.2-1+cuda11.8 \
libnvinfer-plugin-dev=8.5.2-1+cuda11.8 libnvparsers-dev=8.5.2-1+cuda11.8 libnvonnxparsers-dev=8.5.2-1+cuda11.8 \
libnvinfer-samples=8.5.2-1+cuda11.8 libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 \
python3-libnvinfer=8.5.2-1+cuda11.8 python3-libnvinfer-dev=8.5.2-1+cuda11.8

if [ ! -f deepstream_sdk_v6.2.0_jetson.tbz2 ]; then
    echo "deepstream_sdk_v6.2.0_jetson.tbz2 not found in the current directory."
    echo "Please download deepstream_sdk_v6.2.0_jetson.tbz2 in the current directory and run the script again."
    exit 1  # Exit the script with a non-zero status code
fi 

sudo tar -xvf deepstream_sdk_v6.2.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.2/
sudo ./install.sh
sudo ldconfig

sudo apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

cd /opt/nvidia/deepstream/deepstream-6.2/sources
sudo git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
sudo git submodule update --init
sudo apt-get install -y apt-transport-https ca-certificates -y
sudo update-ca-certificates
cd 3rdparty/gst-python/
./autogen.sh
make
sudo make install

cd deepstream_python_apps/bindings
sudo mkdir build
cd build
sudo cmake ..
make

cd /opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps/bindings/build

pip3 install ./pyds-1.1.6-py3-none*.whl


sudo apt-get -y install duplicity 
sudo apt-get install cmake
sudo apt install build-essential libdbus-glib-1-dev libgirepository1.0-dev
sudo apt-get install libudev-dev libsystemd-dev
sudo apt-get install libcups2-dev
sudo apt-get install libxml2-dev libxmlsec1-dev
sudo apt install libcairo2-dev pkg-config python3-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install libcairo2-dev gir1.2-gstreamer-1.0
sudo apt install libpq-dev python3-dev
sudo apt install postgresql-server-dev-all


pip3 install numpy==1.22.0
pip3 install Cython
pip3 install imageio
pip3 install nats-py
pip3 install pytz
pip3 install pycairo
pip3 install PyGObject
pip3 install python-dotenv
pip3 install opencv-python
pip3 install psycopg2
pip3 install torch==1.8.0
pip3 install torchvision==0.9.0
pip3 install natsort
