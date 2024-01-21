# Interpretable Machine Learning to Identify Three-dimensional Patterns in Radiation Associated with Tropical Cyclogenesis

Welcome to the GitHub repository for the paper "Identifying Three-Dimensional Radiative Patterns Associated with Early Tropical Cyclone Intensification". 

Here, you will find the code used to create the three main figures (Fig. 2-4) in the study. Below is a brief overview of the organization of this repository:

## util
The utility folder stores python scripts containing functions used to read in the trained models, benchmark performance, derive variables, and simple plotting. It also contains the pytorch script detailing the architecture of the VED and the baseline two-layer linear regression model. 

- **benchmark.py:** This file contains codes to read in and process the data into Pytorch tensors, codes to calculate different performance metrics for model evaluation, and codes to find the contributions of different radiation to intensification and the extracted mean and standard deviation structures based on the equations in the SI

- **plotting.py:** This file contains codes to create the spread-error diagrams in the SI, and quantify and SSREL metric.

- **read_and_proc.py:** Various codes to read in and clean up the processed time series. Also contains code to convert polar coordinate variables to cartesian one.

- **ts_models.py:** This code defines the baseline model, and the routines to train it.

- **vae.py:** This code defines the VED model, and the routines to train it.

## analysis
This folder stores jupyter notebooks to reproduce the three main figures reported in our study

- **fig2_4.ipynb:** This Jupyter Notebook allows you to reproduce fig2 for Haiyan, and fig4.

- **fig2_maria.ipynb:** This Jupyter Notebook allows you to reproduce fig2 for Maria.

- **fig3.ipynb:** This Jupyter Notebook allows you to reproduce fig3.

## Figures and Tables

You can find the main figures generated from the analysis in the `/figures` folder.

Please cite our preprint if you use this data or code for your research.

Thank you for your interest!
