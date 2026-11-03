"""Generate N labels with different eulers & locations"""
import os
import numpy as np
import json

json_dir = 'dataset'
N = 10

os.makedirs(json_dir, exist_ok=True)

for i in range(N):
	data = {
		'location': np.random.uniform(0, 1, size=3).tolist(),
		'euler': np.random.uniform(0, np.pi/2, size=3).tolist(),
		'scale': np.random.uniform(0.5, 3, size=None)
	}
	with open(os.path.join(json_dir, f'{i:04d}.json'), 'w') as outfile:
		json.dump(data, outfile)