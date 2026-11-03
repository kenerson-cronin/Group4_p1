import blendersynth as bsyn
import numpy as np

# When debugging, you can use the following two lines instead of inputs = bsyn.Inputs()
# bsyn.run_this_script()
# inputs = bsyn.DebugInputs(<path to test json file>)

inputs = bsyn.Inputs()  # This is an iterable of the jsons passed in via run.py. Also manages progress bar.

window = bsyn.Mesh.from_primitive('plane')  # Create window object

bg_plane = bsyn.Mesh.from_primitive('plane', scale=50, location=(-30, 20, -10))
bg_plane.rotation_euler = (np.pi/2,0, 0)  # Rotate the background plane to be vertical
bg_material = bsyn.Material.from_image('part1/environment/PXL_20231015_173110155.jpg')  # Load the background image as a material
bg_plane.material = bg_material  # Assign the background to the plane

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
light = bsyn.Light.create('POINT', location=(1,0,0),intensity=1000.) 
light = bsyn.Light.create('SUN') 

cam_normals_aov = bsyn.aov.NormalsAOV(name='cam_normals', ref_frame='CAMERA')
window.assign_aov(cam_normals_aov)

cam = bsyn.Camera.create()
# cam.look_at_object(window,up=(0,0,1))  # Make the camera look at the window

bsyn.render.set_cycles_samples(10)
bsyn.render.set_resolution(640, 360)
bsyn.render.render_with_gpu(True)

# create compositor to output RGB, Normals AOV & Depth
comp = bsyn.Compositor()
comp.define_output('Image', color_mode='RGB')  # render RGB layer
comp.define_output(cam_normals_aov, name='normal', color_mode='BW')  # render normals layer

for i, (fname, input_data) in enumerate(inputs):
	# Set the pose of the window
	window.rotation_euler = input_data['euler']
	window.location = input_data['location']

	render_result = comp.render()
	render_result.save_all(f"example_dataset/{fname}")

	# Save the pose and lighting as an output json
	output = {**input_data}  # items to save to output label
	annotation = bsyn.annotations.bounding_boxes([window], return_fmt='xywh').get_annotation_by_camera('Camera')
	output['bbox'] = annotation.bbox  # save the bounding box annotations
	bsyn.file.save_label(output, f'example_dataset/label/{fname}.json')