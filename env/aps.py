import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import mpu6050


class CustomAPSEnv:
    def __init__(self):
        self.gravity = 9.8
        self.mass = 1.0
        self.length = 1.0
        self.time_step = 0.05
        self.max_speed = 8
        self.max_torque = 2.0
        self.viewer = None

        self.action_space = SimpleBox(low=-1.2, high=1.2, shape=(1,), dtype=np.float32)
        self.observation_space = SimpleBox(low=-8, high=8, shape=(3,), dtype=np.float32)
        self.state = None
        self.max_steps = 200  
        self.current_step = 0  
        self.now = 0
        self.start = 0
        self.action_ls=[]
        self.action_time=[]
        self.reset()
        

    def reset(self):
        self.start = time.time()
        f = {'r':'f','f':'r'}
        assert self.action_ls==[] or (len(self.action_ls)+1==len(self.action_time))
        for i in range(len(self.action_ls)):
            action = self.action_ls[i]
            motor.loop(f[action[0]]+action[1:])
            time.sleep(self.action_time[i+1])
        if random.choice([0, 1]):
            motor.loop('f2')
            time.sleep(0.7)
            motor.loop('b')
        self.action_ls=[]
        self.action_time=[]
        while True:
            try:
                start_time = time.time()
                angle1 = mpu6050.get_angle()
                angle2 = mpu6050.get_angle()
                end_time = time.time()
                break
            except:
                continue
        self.state = np.array([np.cos(angle2), np.sin(angle2), np.array(angle2-angle1)/(start_time-end_time)])
        self.current_step = 0
        return self.state

    def step(self, action):
        action = action[0]+str(np.clip(float(action[1:]), -self.max_torque, self.max_torque))
        self.now = time.time()
        self.action_ls.append(action)
        self.action_time.append(self.now - self.start)
        self.start = self.now
        motor.loop(action)
        while True:
            try:
                start_time = time.time()
                angle1 = mpu6050.get_angle()
                angle2 = mpu6050.get_angle()
                end_time = time.time()
                break
            except:
                continue
        self.state = np.array([np.cos(angle2), np.sin(angle2), np.array(angle2-angle1)/(start_time-end_time)])
        f = {'r':-1.0,'f':1.0}
        self.state[2] = f[action[0]]*float(action[1:])
        self.current_step += 1  
        done = self.current_step >= self.max_steps  
        # reward = -(self.state[0])**2 + -(self.state[1]+1)**2 + -self.state[2]**2 - float(action[1:])**2
        reward = -(self.state[0])**2 + -(self.state[1]+1)**2 + - float(action[1:])**2
        return self.state, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass
