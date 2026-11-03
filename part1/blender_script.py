"""Here, we show a script that creates a dataset of monkeys with different poses - rendering RGB & Normals.
To run this script, we first generate the dataset of labels using examples/dataset_creation/generate_labels.py
Then, we run this script using create_dataset.py
"""

import blendersynth as bsyn
import numpy as np

# When debugging, you can use the following two lines instead of inputs = bsyn.Inputs()
# bsyn.run_this_script()
# inputs = bsyn.DebugInputs(<path to test json file>)

inputs = bsyn.Inputs()  # This is an iterable of the jsons passed in via run.py. Also manages progress bar.

window = bsyn.Mesh.from_primitive('plane')  # Create window object

# Configure the window texture to be transparent
texture = bsyn.Material.from_image('assets/window_transparent.png')
nodes = texture.node_tree.nodes
links = texture.node_tree.links
image_node = next(n for n in nodes if n.type == 'TEX_IMAGE')
principled_node = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
links.new(image_node.outputs['Alpha'], principled_node.inputs['Alpha'])

# Assign the texture to the window object
window.material = texture  

# Create light source
light = bsyn.Light.create('POINT', location=(1, 0, 0), intensity=100.) 

cam = bsyn.Camera.create()
cam.look_at_object(window,up=(0,0,1))  # Make the camera look at the window

N = 10 # Number of objects
objects = []
for i in range(N):
	object = bsyn.Mesh.from_obj('assets/window.obj')
	object.set_minimum_to('Z', 0)
	objects.append(object)

bsyn.render.set_cycles_samples(10)
bsyn.render.set_resolution(640, 360)
bsyn.render.render_with_gpu(True)

# create compositor to output RGB, Normals AOV & Depth
comp = bsyn.Compositor()
comp.define_output('Image')  # render RGB layer

# Now iterate through and generate dataset
for i, (fname, input_data) in enumerate(inputs):
	# Set the pose of the window
    window.rotation_euler = input_data['euler']
    window.location = input_data['location']
    window.scale = input_data['scale']
    render_result = comp.render()
    render_result.save_all(f"example_dataset/{fname}")

	# Save the pose and lighting as an output json
    output = {**input_data}  # items to save to output label
    bsyn.file.save_label(output, f'example_dataset/label/{fname}.json')