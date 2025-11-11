# Code from mengine by Zachary Erickson
# Adapted by Hayden Feddock

import os, time
import numpy as np
from screeninfo import get_monitors
import pybullet as p

envir = None
asset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')

class Env:
    def __init__(self, time_step=0.02, gravity=[0, 0, -9.81], render=True, gpu_rendering=False, seed=1001, deformable=False):
        global envir
        envir = self
        self.time_step = time_step
        self.gravity = np.array(gravity)
        self.id = None
        self.render = render
        self.gpu_rendering = gpu_rendering
        self.view_matrix = None
        self.deformable = deformable
        self.seed(seed)
        self.asset_dir = asset_dir
        self.visual_items = []
        if self.render:
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
        else:
            self.id = p.connect(p.DIRECT)
        # self.util = Util(self.id)

        self.reset()

    def seed(self, seed=1001):
        np.random.seed(seed)
        self.np_random = None

    def disconnect(self):
        p.disconnect(self.id)

    def reset(self):
        if self.deformable:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=self.id)
        else:
            p.resetSimulation(physicsClientId=self.id)
        # if self.gpu_rendering:
        #     self.util.enable_gpu()
        self.set_gui_camera()
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 5, 10], physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], physicsClientId=self.id)
        self.last_sim_time = time.time()

    def set_gui_camera(self, look_at_pos=[0, 0, 0.75], distance=1, yaw=0, pitch=-30):
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=look_at_pos, physicsClientId=self.id)

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

def step_simulation(steps=1, realtime=True, local_env=None):
    local_env = local_env if local_env is not None else envir
    for _ in range(steps):
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.id) # Enable rendering
        p.stepSimulation(physicsClientId=local_env.id)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=env.id) # Disable rendering, this allows us to create and delete objects without object flashing
        if realtime and local_env.render:
            local_env.slow_time()

def compute_collision_detection(env=None):
    env = env if env is not None else envir
    p.performCollisionDetection(physicsClientId=env.id)

def redraw(env=None):
    env = env if env is not None else envir
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.id)

def get_keys():
    specials = {p.B3G_ALT: 'alt', p.B3G_SHIFT: 'shift', p.B3G_CONTROL: 'control', p.B3G_RETURN: 'return', p.B3G_LEFT_ARROW: 'left_arrow', p.B3G_RIGHT_ARROW: 'right_arrow', p.B3G_UP_ARROW: 'up_arrow', p.B3G_DOWN_ARROW: 'down_arrow'}
    # return {chr(k) if k not in specials else specials[k] : v for k, v in p.getKeyboardEvents().items()}
    return [chr(k) if k not in specials else specials[k] for k in p.getKeyboardEvents().keys()]


