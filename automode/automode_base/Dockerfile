# Use Ubuntu as the base image
FROM ubuntu:20.04

# Set environment variables to avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive


# --- AutoMoDE_INSTALLATION.rtf: 1. install prerequisites

# Install tzdata and dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    wget \
    git \
    cmake \
    build-essential \
    libfreeimage-dev \
    libfreeimageplus-dev \
    qt5-default \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    liblua5.2-dev \
    lua5.2 \
    doxygen \
    graphviz \
    graphviz-dev \
    asciidoc \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Set the timezone to UTC non-interactively
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Set environment variables for ARGoS3 as found throughout instructions
ENV ARGOS_INSTALL_PATH=/root
ENV PKG_CONFIG_PATH=$ARGOS_INSTALL_PATH/argos3-dist/lib/pkgconfig
ENV ARGOS_PLUGIN_PATH=$ARGOS_INSTALL_PATH/argos3-dist/lib/argos3
ENV LD_LIBRARY_PATH=$ARGOS_PLUGIN_PATH:$LD_LIBRARY_PATH
ENV PATH=$ARGOS_INSTALL_PATH/argos3-dist/bin/:$PATH

WORKDIR $ARGOS_INSTALL_PATH

# --- AutoMoDE_INSTALLATION.rtf: 2. install argos

# Install ARGoS3
RUN git clone https://github.com/ilpincy/argos3.git argos3 \
    && cd argos3 \
    && git checkout 3.0.0-beta48 \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=$ARGOS_INSTALL_PATH/argos3-dist -DCMAKE_BUILD_TYPE=Release -DARGOS_INSTALL_LDSOCONF=OFF -DARGOS_DOCUMENTATION=OFF ../src \
    && make \
    && make install

# --- AutoMoDE_INSTALLATION.rtf: 3. install AutoMode

# ------- AutoMoDE_INSTALLATION.rtf: 3.1 e-puck

# Remove default e-puck plugin
RUN rm -rf $ARGOS_INSTALL_PATH/argos3-dist/include/argos3/plugins/robots/e-puck \
    && rm -rf $ARGOS_INSTALL_PATH/argos3-dist/lib/argos3/lib*epuck*.so

# Install e-puck plugin
RUN git clone https://github.com/demiurge-project/argos3-epuck.git argos3-epuck \
    && cd argos3-epuck \
    && git checkout v48 \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=$ARGOS_INSTALL_PATH/argos3-dist -DCMAKE_BUILD_TYPE=Release ../src \
    && make \
    && make install 

# ------- AutoMoDE_INSTALLATION.rtf: 3.2 loop-functions
#NOTE: comment/uncomment subdirectories in loop-functions/CMakeLists.txt depending on requirements
# Install AutoMoDe loopfunctions
RUN git clone https://github.com/demiurge-project/experiments-loop-functions.git AutoMoDe-loopfunctions \
    && cd AutoMoDe-loopfunctions \
    && git checkout dev \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=$ARGOS_INSTALL_PATH/argos3-dist -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install

# ------- AutoMoDE_INSTALLATION.rtf: 3.3 e-puck-DAO
# Install e-puck DAO
RUN git clone https://github.com/demiurge-project/demiurge-epuck-dao.git AutoMoDe-DAO \
    && cd AutoMoDe-DAO \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=$ARGOS_INSTALL_PATH/argos3-dist -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install 

# ------- AutoMoDE_INSTALLATION.rtf: 3.4 AutoMode-BehaviorTree
# Install AutoMoDe
RUN git clone https://github.com/demiurge-project/ARGoS3-AutoMoDe.git AutoMoDe \
    && cd AutoMoDe \
    && mkdir build && cd build \
    && git checkout BehaviorTree \
    && cmake .. \
    && make

# Download and install irace
RUN mkdir -p ~/R/x86_64-redhat-linux-gnu-library/3.5/ \
    && wget -O /root/irace_2.2.tar.gz https://nextcloud.ananas.space/s/WhQ2yqhmqoaBRdC/download/irace_2.2.tar.gz \
    && R CMD INSTALL -l ~/R/x86_64-redhat-linux-gnu-library/3.5/ /root/irace_2.2.tar.gz \
    && rm /root/irace_2.2.tar.gz

# Add irace paths
ENV R_LIBS_USER=~/R/x86_64-redhat-linux-gnu-library/3.5
ENV IRACE_HOME=${R_LIBS_USER}/irace
ENV PATH=${IRACE_HOME}/bin/:${PATH}
ENV R_LIBS=${R_LIBS_USER}:${R_LIBS}
