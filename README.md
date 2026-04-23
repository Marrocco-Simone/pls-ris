# Multi-Receiver Physical Layer Security Using Reconfigurable Intelligent Surfaces

This repository contains simulation code for investigating physical layer security (PLS) in wireless communications using Reconfigurable Intelligent Surfaces (RIS).

## Overview

Physical Layer Security (PLS) introduces security and privacy mechanisms directly at the signal level, as an additional measure on top of what is provided by higher layers of the communication stack. By properly tuning the configuration of RIS, it is possible to enable communication with a set of intended receivers while making the signal unintelligible in nearby areas.

This codebase extends existing approaches to work with **multiple receivers** and **multiple RIS**, discussing secrecy characteristics in the spatial domain. The framework supports both **stochastic channel modeling** (fast simulations with Rice fading) and **realistic ray-tracing** (using Sionna RT with 3D scenarios and multipath propagation).

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run main spatial simulation
python heatmap.py
```

## Available Simulations

### 1. Spatial BER Heatmap (`heatmap.py`)

**What it does:** Generates spatial heatmaps showing Bit Error Rate (BER) and SNR across a grid of positions. Simulates how the signal quality varies in space for legitimate receivers vs. eavesdroppers.

**How to run:**
```bash
# Default simulation
python heatmap.py

# With custom parameters
python heatmap.py --K 4 --N 36 --num_symbols 1000 --Pt_dbm 20
```

**Parameters:**
- `--K`: Number of antennas (power of 2: 2, 4, 8...)
- `--N`: Number of RIS elements
- `--num_symbols`: Symbols to simulate per grid point
- `--Pt_dbm`: Transmission power in dBm
- `--use_noise_floor`: Use realistic noise floor instead of fixed SNR

**Output:** PDF heatmaps saved to `heatmap/K{N}_N{N}_.../pdf/`

---

### 2. BER vs SNR Curves (`ber.py`)

**What it does:** Simulates BER performance across different SNR values, comparing legitimate receivers and eavesdroppers. Includes confidence intervals.

**How to run:**
```bash
python ber.py
```

**Output:** PDF plots in `ber/pdf/`

---

### 3. Secrecy Rate Analysis (`secrecy_rate.py`)

**What it does:** Computes information-theoretic secrecy rates using Monte Carlo sampling. Compares achievable rates for legitimate receivers vs. eavesdroppers.

**How to run:**
```bash
python secrecy_rate.py
```

**Output:** PDF plots in `secrecy_results/`

---

### 4. LOS/NLOS Scenarios (`ber_los_scenarios.py`)

**What it does:** Analyzes scenarios with receivers having direct line-of-sight (LOS) vs. non-line-of-sight (NLOS). Compares passive vs. active RIS performance.

**How to run:**
```bash
python ber_los_scenarios.py
```

**Output:** PDF plots in `ber_los/pdf/`

---

### 5. Diagonalization Verification (`diagonalization.py`)

**What it does:** Standalone script to verify that the RIS reflection matrix produces diagonal effective channels for legitimate receivers while randomizing for eavesdroppers.

**How to run:**
```bash
python diagonalization.py
```

**Output:** Console output showing diagonalization success/failure

---

### 6. Sionna Ray-Tracing (`compute_sionna_channels.py`)

**What it does:** Computes realistic channel matrices using Sionna RT ray-tracing for all predefined scenarios. Replaces stochastic Rice fading with physically accurate propagation.

**Prerequisites:** Requires TensorFlow and Sionna (see `requirements.txt`)

**How to run:**
```bash
# First generate scene XML files
python generate_scene_xmls.py

# Then compute ray-traced channels
python compute_sionna_channels.py
```

**Output:** `.npz` files in `heatmap/channel_matrices/`

**Note:** Forces CPU mode to avoid GPU memory issues.

---

### 7. Real-World Scenes (`scripts/osm_to_sionna/`)

**What it does:** Converts OpenStreetMap building data to Sionna-compatible 3D scenes using Blender.

**Prerequisites:** Blender 3.6+ with blosm and mitsuba-blender addons

**How to run:**
```bash
# Interactive mode (opens map selector)
python scripts/osm_to_sionna/osm_to_sionna.py --name "my_scene"

# With coordinates
python scripts/osm_to_sionna/osm_to_sionna.py \
    --coords "11.12006,46.06603,11.12419,46.06868" \
    --name "trento_test"
```

See `scripts/osm_to_sionna/README.md` for detailed setup instructions.

---

## Project Structure

```
.
├── Core Simulations
│   ├── heatmap.py              # Spatial BER/SNR heatmaps
│   ├── ber.py                  # BER vs SNR curves
│   ├── secrecy_rate.py         # Secrecy rate analysis
│   └── ber_los_scenarios.py    # LOS/NLOS comparisons
│
├── RIS Optimization
│   ├── diagonalization.py      # Reflection matrix calculation
│   ├── estimation_ambiguity.py   # Channel estimation tools
│   └── estimation_ambiguity_series.py  # Multi-RIS estimation
│
├── Channel Modeling
│   ├── heatmap_utils.py        # Channel calculations
│   ├── heatmap_situations.py   # Scenario definitions
│   └── noise_power_utils.py    # Noise generation
│
├── Sionna Integration
│   ├── compute_sionna_channels.py    # Ray-traced channels
│   ├── sionna_utils.py               # Actor utilities
│   ├── generate_scene_xmls.py          # Scene generation
│   └── single_ris_channel_calc.py    # Single scenario verification
│
└── Tools
    └── scripts/osm_to_sionna/  # OpenStreetMap converter
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **K** | Transmit antennas (power of 2) | 2, 4, 8 |
| **N** | RIS elements | 16, 36, 200 |
| **J** | Legitimate receivers | 1, 2, 5 |
| **M** | RIS surfaces | 1, 2, 3 |
| **eta** | Reflection efficiency | 0.9 |
| **Pt_dBm** | Transmit power | 0, 20, 40 |

## Publications

This codebase has been developed as part of the following research publications. All papers share the affiliation: **Department of Information Engineering and Computer Science, University of Trento, Italy & CNIT, Italy**.

### Published

**S. Marrocco, P. Casari, M. Segata**, "Exploiting Reconfigurable Intelligent Surfaces to Achieve Multi-Receiver Physical Layer Security," in *31st IEEE International Symposium on On-Line Testing and Robust System Design (IOLTS 2025)*, Ischia, Italy, June 2025.

> Primary paper introducing the null-space diagonalization approach for multi-receiver physical layer security using RIS, with support for cascaded RIS configurations (in series and in parallel).

**S. Marrocco, P. Casari, M. Segata**, "Multi-Receiver Physical Layer Security Using Reconfigurable Intelligent Surfaces," in *16th IEEE Vehicular Networking Conference (VNC 2025), Poster Session*, Porto, Portugal, June 2025.

> Companion poster paper presenting the multi-receiver PLS framework at the vehicular networking venue.

### Accepted

**S. Marrocco, P. Casari, M. Segata**, "RIS-Based Physical Layer Security: Realistic Evaluation and Challenges Ahead," in *18th Wireless On-demand Network systems and Services Conference (WONS 2026)*, 2026.

> Extends the IOLTS 2025 work by integrating Sionna ray-tracing for realistic channel estimation, replacing the stochastic Rice fading model. Studies the spatial coverage of the PLS secured area using 3D ray-traced propagation.

## Citation

If you use this code in your research, please cite:

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

```bibtex
@inproceedings{marrocco2025sionna,
   author = {Marrocco, Simone and Casari, Paolo and Segata, Michele},
   title = {{RIS-Based Physical Layer Security: Realistic Evaluation
and Challenges Ahead}},
   booktitle = {18th Wireless On-demand Network systems and Services
Conference (WONS 2026)},
   year = {2026}
}
```

## License

This project is released for research and academic purposes. Please cite the relevant publications when using this code in your research.

## Contributing

This research code is maintained by **Simone Marrocco**. For questions or collaborations, please refer to the publications listed above or contact the Department of Information Engineering and Computer Science, University of Trento, Italy.
