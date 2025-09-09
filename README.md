# SALM: Streak-Aware Localization Microscopy Enables High-Throughput Brain Imaging Across Platforms

Streak-Aware Localization Microscopy (SALM) is a novel localization microscopy framework that leverages motion-blurred streaks as intrinsic spatiotemporal barcodes to achieve super-resolved structural and functional brain imaging. SALM removes the need for explicit localization and tracking, reduces detector frame-rate requirements by more than an order of magnitude, and enables seamless integration across benchtop, miniaturized, and second near-infrared imaging platforms. SALM is powered by three illumination encoding strategies, a realistic spatiotemporal trajectory simulation engine, and a localization-free deep learning network (LfNet).

This repository provides the code and data for the simulation engine and the localization-free deep learning model.

## 📁 Project Structure

##### SALM/ 

  ├── Simulation_Engine/  - MATLAB codes for the spatiotemporal trajectory simulation engine  
  ├── Train/              - Training codes for LfNet under 3 SALM configurations    
  ├── Test/               - Testing codes for LfNet under 3 SALM configurations       
  ├── environment.yml     - Python dependencies

## 🚀 Quick Start (Testing Pretrained Models)
### 1. Install Dependencies
Make sure you have Python ≥ 3.8 + Pytorch ≥ 1.11.0 and install required packages:  
conda env create -f environment.yml
### 2. Download Pretrained Models
Pretrained models for SALM1/2/3 are available on [Google Drive](https://drive.google.com/drive/folders/1wE27752z1f0NMGmYSweA0jrt6v82pZIp?usp=sharing).

A backup link [Baidu Netdisk](https://pan.baidu.com/s/1jpxL8lyzBCj-qJaXkhvsTw) (Passwork: 0000).

Place move the downloaded model files into the 'Test/' folder.


## 🧠 Train a New Model on Your Data
### 1. Generate Training Dataset with Simulation Engine
Use the MATLAB scripts in '/Simulation_Engine' to create synthetic datasets:
- SALM1_Simulation_Main.m – SALM1 configuration (1× continuous-wave laser). 
- SALM2_Simulation_Main.m – SALM2 configuration (1× pulsed laser).
- SALM3_Simulation_Main.m – SALM3 configuration (1× CW laser + 1× pulsed laser).

Running these scripts will generate both:
- SALM1/2/3_Dataset.H5 (dataset file)
- A Train/ folder containing visualizations for quick inspection
  
The MATLAB scripts allow customization of several parameters, such as Gaussian kernel size for PSF simulation, Noise levels, Trajectory length ranges and Number of emitters. Please tune these parameters to ensure that the synthetic dataset closely resembles your experimental data.

### 2. Train LfNet
Move the generated H5 dataset into the '/Train' directory, then run:
- Train_StructuralMap.py   # Train structural map predictor
- Train_VelocityMap.py     # Train velocity map predictor
- Train_DirectionMap.py    # Train direction map predictor
  
The same training scripts can be used for SALM1/2/3 datasets.


## 🖼 Sample Results
Sample results for SALM1/2/3 can be found in: '/Test/SALM1/2/3_Results'.

## 📬 Contact
For questions, please contact:
xuyang_chang@163.com
Or open an issue on this GitHub repository.
