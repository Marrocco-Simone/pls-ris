Sure, here are the detailed instructions to install and build CoopeRIS on a Unix environment (Ubuntu or macOS):

### Prerequisites

1. **Install SUMO (1.18.0)**
2. **Install OMNeT++ (6.0.1)**
3. **Install GNU Scientific Library (GSL)**

### Step-by-Step Instructions

#### 1. Install SUMO

**Ubuntu:**

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo
```

**macOS:**

```bash
brew install sumo
```

#### 2. Install OMNeT++

**Ubuntu & macOS:**

1. Download OMNeT++ 6.0.1 from the [official website](https://omnetpp.org/download/).
2. Extract the downloaded file:
   ```bash
   tar -xzf omnetpp-6.0.1-src.tgz
   cd omnetpp-6.0.1
   ```
3. Install dependencies:
   ```bash
   sudo apt-get install build-essential gcc g++ bison flex perl tcl-dev tk-dev libxml2-dev zlib1g-dev default-jre doxygen graphviz libwebkit2gtk-4.0-dev qt5-qmake qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
   ```
   For macOS, use:
   ```bash
   brew install bison flex perl tcl-tk libxml2 zlib openjdk doxygen graphviz qt
   ```
4. Configure and build OMNeT++:
   ```bash
   ./configure
   make
   ```

#### 3. Install GNU Scientific Library (GSL)

**Ubuntu:**

```bash
sudo apt install libgsl-dev
```

**macOS:**

```bash
brew install gsl
```

#### 4. Install Veins

1. Clone the customized version of Veins:
   ```bash
   git clone https://github.com/michele-segata/veins.git
   cd veins
   git checkout cooperis
   ```
2. Compile Veins:
   ```bash
   ./configure
   make
   ```

#### 5. Install Plexe

1. Clone Plexe version 3.1.3:
   ```bash
   git clone https://github.com/michele-segata/plexe.git
   cd plexe
   git checkout -b plexe-3.1.3-work plexe-3.1.3
   ```
2. Compile Plexe:
   ```bash
   ./configure
   make
   ```

#### 6. Install CoopeRIS

1. Clone the CoopeRIS repository:
   ```bash
   git clone https://github.com/michele-segata/cooperis.git
   cd cooperis
   ```
2. Configure and build CoopeRIS with multithread support:
   ```bash
   ./configure --with-gsl-include=/opt/local/include --with-gsl-lib=/opt/local/lib
   make
   ```

#### 7. GPU Support (Optional)

**Cuda Support:**

1. Install the Cuda toolkit from the [official NVIDIA website](https://developer.nvidia.com/cuda-downloads).
2. Configure and build CoopeRIS with Cuda support:
   ```bash
   ./configure --with-gsl-include=/opt/local/include --with-gsl-lib=/opt/local/lib --with-cuda --with-cuda-include=/opt/local/include --with-cuda-lib=/opt/local/lib
   make
   ```

**OpenCL Support:**

1. Install the OpenCL framework from the [official Khronos website](https://www.khronos.org/opencl/).
2. Configure and build CoopeRIS with OpenCL support:
   ```bash
   ./configure --with-gsl-include=/opt/local/include --with-gsl-lib=/opt/local/lib --with-opencl --with-opencl-include=/opt/local/include --with-opencl-lib=/opt/local/lib
   make
   ```

#### 8. Run Plexe Example

1. Build the `plexe_cooperis` subproject:
   ```bash
   cd plexe/subprojects/plexe_cooperis
   source setenv
   ./configure
   make
   ```
2. Run the example:
   ```bash
   cd examples/plexe_cooperis
   plexe_cooperis_run -u Cmdenv -c TrackingTIntersection -r 0
   ```

### Notes

- You can specify the number of compute threads for RIS gain computation using the `*.**.nicRis.phyRis.maxWorkerThreads` parameter in the

omnetpp.ini

file.

- For GPU support, you can specify the Cuda or OpenCL device and platform using the `*.**.nicRis.phyRis.cudaDeviceId`, `*.**.nicRis.phyRis.openclDeviceId`, and `*.**.nicRis.phyRis.openclPlatformId` parameters in the `omnetpp.ini` file.

By following these steps, you should be able to install and build CoopeRIS on your Unix environment.
