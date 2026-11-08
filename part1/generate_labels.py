"""Generate N labels with different eulers & locations"""
import os
import numpy as np
import json

json_dir = 'labels'
N = 50

os.makedirs(json_dir, exist_ok=True)

for i in range(N):
    thetaX = np.random.uniform(np.pi/2, np.pi)  # Random rotation around X-axis
    thetaY = np.random.uniform(np.pi/2, np.pi)  # Random rotation around Y-axis
    thetaZ = np.random.uniform(0, np.pi/2)  # Random rotation around Z-axis

    Xpos = np.random.uniform(-3,3)  # Random X position
    Ypos = np.random.uniform(-3,3)  # Random Y position
    Zpos = np.random.uniform(0,2)  # Random Z position
    data = {
        'window_A': {
            'location': [np.random.uniform(-3,3),
                         np.random.uniform(-3,3),
                         np.random.uniform(0,2)],
            'euler': [np.pi/2,
                      np.random.uniform(np.pi/2, np.pi),
                      np.random.uniform(0, np.pi/2)],
            'scale': np.random.uniform(0.5, 2, size=None)
        },
        'window_B': {
            'location': [np.random.uniform(-3,3),
                         np.random.uniform(-3,3),
                         np.random.uniform(0,2)],
            'euler': [np.pi/2,
                      np.random.uniform(np.pi/2, np.pi),
                      np.random.uniform(0, np.pi/2)],
            'scale': np.random.uniform(0.5, 2, size=None)
        }
    }
    with open(os.path.join(json_dir, f'{i:04d}.json'), 'w') as outfile:
	    json.dump(data, outfile)