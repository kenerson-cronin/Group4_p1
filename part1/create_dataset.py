import os

import blendersynth as bsyn
json_dir = 'labels'

bsyn.execute_jobs(
	script='part1/blender_script.py',
	json_src=json_dir,
	output_directory='run_report',
	num_threads=10,
)

# for file in os.listdir('example_dataset'):
#     print(filename)