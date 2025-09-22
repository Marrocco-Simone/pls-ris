# Multi-Receiver Physical Layer Security Using Reconfigurable Intelligent Surfaces

This repository contains the simulation code for investigating multi-receiver physical layer security using Reconfigurable Intelligent Surfaces (RIS).

## Overview

The project implements a comprehensive simulation environment for studying physical layer security in multi-receiver scenarios enhanced by RIS technology. The framework includes:

- **RIS Channel Modeling**: Advanced channel matrix calculations and RIS reflection optimization
- **Physical Layer Security**: Implementation of secrecy capacity calculations and secure transmission schemes
- **Multi-Receiver Support**: Simulation of scenarios with multiple legitimate receivers and eavesdroppers
- **Performance Analysis**: Bit Error Rate (BER) analysis and security metrics evaluation
- **Visualization Tools**: Heatmap generation and performance plotting capabilities

## Key Features

- **Space Shift Keying (SSK) Modulation**: Implementation of SSK transmission schemes for secure communication
- **RIS Optimization**: Diagonalization-based algorithms for optimal RIS reflection matrix design
- **Multi-threaded Simulation**: Parallel processing support for large-scale simulations
- **Noise Modeling**: Comprehensive noise power calculations with SNR and noise floor options
- **Security Metrics**: Secrecy rate calculations and eavesdropper analysis
- **Visualization**: Interactive heatmap generation with customizable parameters

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)

### Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

The main simulation script can be run directly:

```bash
python heatmap.py
```

This will generate performance heatmaps showing the security and communication performance across different spatial configurations.

> [!NOTE]
> The `heatmap_v2.py` file is currently work in progress and represents a rewrite of the main heatmap functionality. Use `heatmap.py` for stable simulations.

### Configuration

Key simulation parameters can be configured in the main scripts:

- **K**: Number of transmit antennas for SSK modulation
- **N**: Number of RIS elements
- **Pt_dbm**: Transmission power in dBm
- **eta**: RIS efficiency parameter
- **snr_db**: Signal-to-noise ratio in dB

### Parallel Processing

The framework supports multi-threaded execution for faster simulations. The number of CPU cores used is automatically detected and limited to 64 for optimal performance.

## Project Structure

- `diagonalization.py`: RIS reflection matrix optimization algorithms
- `ber.py`: Bit Error Rate simulation and analysis
- `heatmap.py`: Main simulation script with visualization capabilities
- `heatmap_v2.py`: Work-in-progress rewrite of the main heatmap functionality
- `heatmap_utils.py`: Utility functions for signal processing and channel calculations
- `noise_power_utils.py`: Noise modeling and power calculation utilities
- `requirements.txt`: Python dependencies

## Scientific Documentation

For detailed mathematical descriptions and theoretical background, please refer to the following publications:

### Journal Articles

```bibtex
@inproceedings{marrocco2025exploiting,
   author = {Marrocco, Simone and Casari, Paolo and Segata, Michele},
   title = {{Exploiting Reconfigurable Intelligent Surfaces to Achieve
Multi-Receiver Physical Layer Security}},
   publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
   address = {Ischia, Italy},
   booktitle = {31st IEEE International Symposium on On-Line Testing and
Robust System Design (IOLTS 2025)},
   month = {6},
   year = {2025}
}
```

### Conference Papers

```bibtex
@inproceedings{marrocco2025multireceiver,
   author = {Marrocco, Simone and Casari, Paolo and Segata, Michele},
   title = {{Multi-Receiver Physical Layer Security Using Reconfigurable
Intelligent Surfaces}},
   publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
   address = {Porto, Portugal},
   booktitle = {16th IEEE Vehicular Networking Conference (VNC 2025),
Poster Session},
   month = {6},
   year = {2025}
}
```

## Results and Data

Simulation results are automatically saved in the `results_data_v2/` directory:

- `results_data_v2/pdf/`: Generated plots and visualizations in PDF format
- `results_data_v2/data/`: Raw simulation data and numerical results

## License

This project is released for research and academic purposes. Please cite the relevant publications when using this code in your research.

## Contributing

This research code is maintained by Simone Marrocco. For questions or collaborations, please refer to the publications listed above.

## Acknowledgments

This work was supported by research funding and developed in collaboration with the Department of Information Engineering and Computer Science, University of Trento, Italy.
