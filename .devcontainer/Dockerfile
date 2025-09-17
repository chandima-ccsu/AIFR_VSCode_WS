# Get a stable docker image with ros2 installed
FROM osrf/ros:kilted-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Install common tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    net-tools \
    # Add other ROS 2 dependencies as needed
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Append ROS 2 setup to /etc/bash.bashrc for all users
RUN echo "source /opt/ros/kilted/setup.bash" >> /etc/bash.bashrc

CMD ["bash"]
