# Imports
import numpy as np
import copy as cp

# Mujoco
import mujoco
import mujoco_viewer

# Controller functions
from control.controllers.mppi_locomotion import MPPI

from utils.tasks import get_task
from utils.transforms import batch_world_to_local_velocity

# Visualization
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib.pyplot as plt

#plt interactive mode on
plt.ion()
# Task
task = 'walk_straight'
task_data = get_task(task)
model_path = task_data['sim_path']
# model_path = 'models/go2w/go2w.xml'
# from dm_control import mjcf

# mjcf_model = mjcf.from_path(model_path)

# def print_body_tree(body, indent=0):
#     print("  " * indent + f"- {body.name}")
#     for child_body in body.body:  # <-- children of this body
#         print_body_tree(child_body, indent + 1)

# # Go through all top-level bodies in the world
# for top_body in mjcf_model.worldbody.body:
#     print_body_tree(top_body)

# Model visualizer
model_sim = mujoco.MjModel.from_xml_path(model_path)
dt_sim = 0.01
model_sim.opt.timestep = dt_sim
data_sim = mujoco.MjData(model_sim)
viewer = mujoco_viewer.MujocoViewer(model_sim, data_sim, 'offscreen')
# Reset robot (keyframes are defined in the xml)
mujoco.mj_resetDataKeyframe(model_sim, data_sim, 0) # stand position
mujoco.mj_forward(model_sim, data_sim)
q_init = cp.deepcopy(data_sim.qpos) # save reference pose
v_init = cp.deepcopy(data_sim.qvel) # save reference pose
print("Configuration: {}".format(q_init)) # save reference pose
img = viewer.read_pixels()
plt.imshow(img)

# Initialize controller
controller = MPPI(task=task)
controller.internal_ref = True
controller.reset_planner()
q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
x = np.concatenate([q_curr, v_curr])
# Set simulation time
tfinal = 8 # 14 for stairs, 30 for walk_octagon
tvec = np.linspace(0,tfinal,int(np.ceil(tfinal/dt_sim))+1)
mujoco.mj_resetDataKeyframe(model_sim, data_sim, 1)
mujoco.mj_forward(model_sim, data_sim)
viewer.cam.distance = 4.2
viewer.cam.lookat[:] = [2.2, 0, 0.27]
img = viewer.read_pixels()
plt.imshow(img)
# Run simulation
anim_imgs = []
sim_inputs = []
x_states = []

for ticks, ti in enumerate(tvec):
    q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
    v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
    x = np.concatenate([q_curr, v_curr])
    
    if ticks%1 == 0:
        u_joints = controller.update(x)  
        
    data_sim.ctrl[:] = u_joints
    mujoco.mj_step(model_sim, data_sim)
    mujoco.mj_forward(model_sim, data_sim)

    error = np.linalg.norm(np.array(controller.body_ref[:3]) - np.array(data_sim.qpos[:3]))

    viewer.add_marker(
        pos=controller.body_ref[:3]*1,         # Position of the marker
        size=[0.15, 0.15, 0.15],     # Size of the sphere
        rgba=[1, 0, 1, 1],           # Color of the sphere (red)
        type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
        label=""
    )

    if error < controller.goal_thresh[controller.goal_index]:
        controller.next_goal()
    
    img = viewer.read_pixels()
    if ticks % 2 == 0:
        anim_imgs.append(img)
    sim_inputs.append(u_joints)
    x_states.append(x)

# Get the resolution of the images
image_height, image_width = anim_imgs[0].shape[:2]

# Set the figure size to match the image resolution
fig, ax = plt.subplots(figsize=(image_width / 500, image_height / 500), dpi=100)
skip_frames = 5
interval = dt_sim*1000*skip_frames
# Remove the white border by setting margins and padding to zero
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_position([0, 0, 1, 1])

def animate(i):
    ax.clear()
    ax.imshow(anim_imgs[i * skip_frames])  # Display the image, skipping frames
    ax.axis('off')

# Create animation, considering the reduced frame rate due to skipped frames
ani = FuncAnimation(fig, animate, frames=len(anim_imgs) // skip_frames, interval=interval)  # 50 ms for 20 Hz

# Display the animation
HTML(ani.to_jshtml())


# Save the animation
# ani.save('{}.mov'.format(task), writer='ffmpeg', fps=20, codec='prores', bitrate=-1)
# ani.save('{}.mp4'.format(task), writer='ffmpeg', fps=20, codec='prores', bitrate=-1)
ani.save('{}.gif'.format(task), writer='pillow', fps=20)


plt.imshow(anim_imgs[-1])
input()