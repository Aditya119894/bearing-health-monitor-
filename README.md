# Predictive Bearing Health Monitor

**Team Spectre** | UVCE, Bengaluru
Aditya Desai, Sudharshan M. Thejasvi

Physical AI Hackathon, Track 6: Robot-Assisted Manufacturing (Predictive Maintenance)

## Problem
Motor bearings wear gradually, then fail suddenly, causing breakdowns and lost production. Industrial monitoring is too costly for small factories.

## Solution: Perceive, Decide, Act
- Perceive: MPU6050 vibration sensor (simulated here)
- Decide: Isolation Forest (scikit-learn) trained only on healthy vibration. Features per 0.1 s window: RMS, kurtosis, peak, crest factor
- Act: after 4 consecutive abnormal windows, the motor slows to 40% and a maintenance alert is raised

## Simulation results
- Fault begins at 12 s, AI alarm at about 16 s (detection delay 4.1 s)
- Motor slowed to 40%
- No false alarm before the fault


## Run
pip install numpy scipy scikit-learn matplotlib imageio imageio-ffmpeg pillow
python simulation.py

Outputs: motor_pro_fixed.mp4 (dashboard video) and results.png (graph).

## Planned hardware
ESP32 + MPU6050 + motor driver, estimated under Rs 1,000.

## Limitation
Results use synthetic vibration data. Real-motor validation is the next step.

Built with Python, NumPy, SciPy, scikit-learn, Matplotlib, ImageIO, Pillow.