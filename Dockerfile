# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  gcc \
  g++ \
  bison \
  flex \
  perl \
  tcl-dev \
  tk-dev \
  libxml2-dev \
  zlib1g-dev \
  default-jre \
  doxygen \
  graphviz \
  libwebkit2gtk-4.0-dev \
  qt5-qmake \
  qtbase5-dev \
  qtchooser \
  qt5-qmake \
  qtbase5-dev-tools \
  git \
  wget \
  libgsl-dev \
  cmake \
  sudo \
  software-properties-common

RUN wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-6.1.0/omnetpp-6.1.0-linux-x86_64.tgz
RUN git clone https://github.com/michele-segata/veins.git
RUN git clone https://github.com/michele-segata/plexe.git
RUN git clone https://github.com/michele-segata/cooperis.git

RUN apt-get install -y \
  python3.8-venv \
  python3-dev \
  libopenscenegraph-dev

# Install SUMO
RUN add-apt-repository ppa:sumo/stable && \
  apt-get update && \
  apt-get install -y sumo

# Install OMNeT++
RUN tar -xzf omnetpp-6.1.0-linux-x86_64.tgz && \
  cd omnetpp-6.1 && \
  echo "WITH_OSG=no" > configure.user && \
  python3 -m venv .venv && \
  . .venv/bin/activate && \
  python3 -m pip install -r python/requirements.txt && \
  /bin/bash -c "source setenv && ./configure && make"

# Set OMNeT++ environment variables
ENV PATH="/omnetpp-6.1/bin:$PATH"
ENV OMNETPP_ROOT="/omnetpp-6.1"

# Install Veins
RUN d veins && \
  git checkout cooperis && \
  ./configure && \
  make

# Install Plexe
RUN  cd plexe && \
  git checkout -b plexe-3.1.3-work plexe-3.1.3 && \
  ./configure && \
  make

# Install CoopeRIS
RUN cd cooperis && \
  ./configure --with-gsl-include=/usr/include --with-gsl-lib=/usr/lib && \
  make

# Build Plexe example
RUN cd plexe/subprojects/plexe_cooperis && \
  /bin/bash -c "source setenv && ./configure && make"

# Set working directory
WORKDIR /plexe/subprojects/plexe_cooperis/examples/plexe_cooperis

# Command to run the example
CMD ["plexe_cooperis_run", "-u", "Cmdenv", "-c", "TrackingTIntersection", "-r", "0"]