import blendersynth as bsyn
json_dir = 'dataset'

bsyn.execute_jobs(
	script='part1/blender_script.py',
	json_src=json_dir,
	output_directory='example_dataset',
	num_threads=5,
)
