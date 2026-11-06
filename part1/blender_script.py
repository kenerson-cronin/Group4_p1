import os

import blendersynth as bsyn
import numpy as np

# When debugging, you can use the following two lines instead of inputs = bsyn.Inputs()
# bsyn.run_this_script()
# inputs = bsyn.DebugInputs(<path to test json file>)

inputs = bsyn.Inputs()  # This is an iterable of the jsons passed in via run.py. Also manages progress bar.

window_A = bsyn.Mesh.from_primitive('plane')  # Create window object
window_B = bsyn.Mesh.from_primitive('plane')  # Create window object

bg_plane = bsyn.Mesh.from_primitive('plane', scale=50, location=(-30, 20, -10))
bg_plane.rotation_euler = (np.pi/2,0, 0)  # Rotate the background plane to be vertical

# Configure the window texture to be transparent
texture = bsyn.Material.from_image('assets/window_transparent.png')
nodes = texture.node_tree.nodes
links = texture.node_tree.links
image_node = next(n for n in nodes if n.type == 'TEX_IMAGE')
principled_node = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
links.new(image_node.outputs['Alpha'], principled_node.inputs['Alpha'])

# Assign the texture to the window object
window_A.material = texture  
window_B.material = texture  

# Create light source
light = bsyn.Light.create('SUN') 

cam_normals_aov = bsyn.aov.NormalsAOV(name='cam_normals', ref_frame='CAMERA')
window_A.assign_aov(cam_normals_aov)
window_B.assign_aov(cam_normals_aov)

cam = bsyn.Camera.create()
# cam.look_at_object(window,up=(0,0,1))  # Make the camera look at the window

bsyn.render.set_cycles_samples(4)
bsyn.render.set_resolution(640, 360)
bsyn.render.render_with_gpu(True)

# create compositor to output RGB, Normals AOV & Depth
comp = bsyn.Compositor()
comp.define_output('Image', color_mode='RGB')  # render RGB layer
comp.define_output(cam_normals_aov, name='normal', color_mode='BW')  # render normals layer

for i, (fname, input_data) in enumerate(inputs):
	window_A.render_visibility(True)
	window_B.render_visibility(True)

	# Select a random background
	random_bg = np.random.choice(os.listdir("part1/environment"))
	random_bg = os.path.join("part1/environment", random_bg)
	bg_material = bsyn.Material.from_image(random_bg)
	bg_plane.material = bg_material  # Assign the background to the plane

	# Set the pose of the window
	window_A.rotation_euler = input_data['window_A']['euler']
	window_A.location = input_data['window_A']['location']
	window_A.scale = input_data['window_A']['scale']

	window_B.rotation_euler = input_data['window_B']['euler']
	window_B.location = input_data['window_B']['location']
	window_B.scale = input_data['window_B']['scale']
	
	if np.random.rand() < 0.5:
		window_B.render_visibility(False)  # Randomly hide window B with 70% probability

	light = bsyn.Light.create('POINT',location=window_A.location,intensity=2000)  # Random light position and intensity

	render_result = comp.render()
	render_result.save_all(f"example_dataset/{fname}")

	# Save the pose and lighting as an output json
	output = {**input_data}  # items to save to output label
	annotation = bsyn.annotations.bounding_boxes([window_A, window_B], return_fmt='xywh').get_annotation_by_camera('Camera')
	output['bbox'] = annotation.bbox  # save the bounding box annotations
	bsyn.file.save_label(output, f'example_dataset/label/{fname}.json')