import numpy as np

# Constants for OKS and mAP computation
SIGMAS = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079]) 
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)