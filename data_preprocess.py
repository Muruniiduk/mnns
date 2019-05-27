from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

import mujoco_py
import os
import cv2
import numpy as np
import subprocess
import pandas as pd

#<camera name="main1" mode="targetbody" target="floor" euler="0 0 0" fovy="100" pos="0 0 2.5"/>
MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="robot" pos="0 0 0.2">
            <joint axis="1 0 0" damping="-0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="-0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="10" pos="0 0 0" rgba="1 0 0 1" size="0.15" type="sphere" friction="0 0" solref="-1000 -1"/>
        </body>

        <body name="box" pos="-0.98 0 0.2">
            <geom mass="0.1" size="0.02 1 0.15" rgba="0 1 0 1" type="box" friction="0 0"/>
        </body>
        <body name="box2" pos="0.98 0 0.2">
            <geom mass="0.1" size="0.02 1 0.15" rgba="0 1 0 1" type="box" friction="0 0"/>
        </body>
        <body name="box3" pos="0 0.98 0.2">
            <geom mass="0.1" size="1 0.02 0.15" rgba="0 1 0 1" type="box" friction="0 0"/>
        </body>
        <body name="box4" pos="0 -0.98 0.2">
            <geom mass="0.1" size="1 0.02 0.15" rgba="0 1 0 1" type="box" friction="0 0"/>
        </body>


        <body name="floor" pos="0 0 0.025">
            <geom condim="3" size="1.0 1.0 0.02" rgba="0 1 0 1" type="box" friction="0 0"/>
            <camera euler="0 0 0" fovy="60" name="rgb" pos="0 0 2.5"></camera>
        </body>
    </worldbody>
    <actuator>
        <motor gear="2000.0" joint="slide0"/>
        <motor gear="2000.0" joint="slide1"/>
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)

# to speed up computation we need the off screen rendering
viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
viewer.cam.elevation = -90
df = pd.DataFrame(columns=['x','y','x_vel','y_vel'])
for i in range(9999):
    sim.data.ctrl[0] = (i==0) * np.random.uniform(-3,3) #speed at t==0
    sim.data.ctrl[1] = (i==0) * np.random.uniform(-3,3)
    x, y, _ = sim.data.qpos
    x_vel,y_vel,_ = sim.data.qvel
    df = df.append({'x':x,'y':y, 'x_vel': x_vel,'y_vel': y_vel}, ignore_index=True)
    viewer.render(480, 480, 0)
    # data = np.asarray(viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)
    data = np.asarray(viewer.read_pixels(480, 480, depth=False)[:, :, :], dtype=np.uint8)
    # save data
    if data is not None:
        name = ("frames/img_%.4d.png" % i)
        cv2.imwrite(name, data)

    print(i)
    sim.step()
df.to_csv('df.csv', index=False)
