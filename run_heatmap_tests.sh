#!/bin/bash

# Run heatmap.py with 4 combinations of parameters:
# - noise floor on/off
# - power 30.0 / 0.0
# - num_symbols 100 (fixed)

echo "Starting heatmap test runs..."
echo "================================"

# 1. Noise floor ON, Power 30.0, 100 symbols
echo "Run 1: noise_floor=ON, Pt_dbm=30.0, num_symbols=100"
python heatmap.py --use_noise_floor --Pt_dbm 30.0 --num_symbols 100

# 2. Noise floor ON, Power 0.0, 100 symbols
echo ""
echo "Run 2: noise_floor=ON, Pt_dbm=0.0, num_symbols=100"
python heatmap.py --use_noise_floor --Pt_dbm 0.0 --num_symbols 100

# 3. Noise floor OFF, Power 30.0, 100 symbols
echo ""
echo "Run 3: noise_floor=OFF, Pt_dbm=30.0, num_symbols=100"
python heatmap.py --no_use_noise_floor --Pt_dbm 30.0 --num_symbols 100

# 4. Noise floor OFF, Power 0.0, 100 symbols
echo ""
echo "Run 4: noise_floor=OFF, Pt_dbm=0.0, num_symbols=100"
python heatmap.py --no_use_noise_floor --Pt_dbm 0.0 --num_symbols 100

echo ""
echo "================================"
echo "All test runs completed!"
