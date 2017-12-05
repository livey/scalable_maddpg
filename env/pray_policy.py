import numpy as np
import math

v_prey=0.1

class Pray:
    def __init__(self,max_edge):
        self.max_edge=max_edge
        print('create pray')

    def action(self, obs):
        # ignore observation and just act based on keyboard events

        length = obs.shape[0]
        prey_x = obs[0, 2]
        prey_y = obs[0, 3]
        gravity_x = 0
        gravity_y = 0
        for i in range(0, length):
            gravity_x += obs[i, 0]
            gravity_y += obs[i, 1]
        gravity_x = gravity_x / length
        gravity_y = gravity_y / length
        x=prey_x-gravity_x
        y=prey_y-gravity_y
        theta=np.arctan2(y,x)
        proba = np.random.random()
        u = np.zeros(5)
        if proba > 0.3:
            theta=2 * np.random.random() - 1
            dis_x = math.cos(theta * math.pi)
            dis_y = math.sin(theta * math.pi)
            if dis_x > 0:
                u[1]=dis_x
            if dis_x < 0:
                u[2]=-dis_x
            if dis_y > 0:
                u[3]=dis_y
            if dis_y < 0:
                u[4]=-dis_y
            if sum(u)==0:
                u[0]=1
        elif proba >0.8:
            dis_x = math.cos(theta * math.pi)
            dis_y = math.sin(theta * math.pi)
            if dis_x > 0:
                u[1] = dis_x
            if dis_x < 0:
                u[2] = -dis_x
            if dis_y > 0:
                u[3] = dis_y
            if dis_y < 0:
                u[4] = -dis_y
            if sum(u) == 0:
                u[0] = 1

        elif proba >0.9:
            theta=theta+math.pi/2
            dis_x = math.cos(theta * math.pi)
            dis_y = math.sin(theta * math.pi)
            if dis_x > 0:
                u[1] = dis_x
            if dis_x < 0:
                u[2] = -dis_x
            if dis_y > 0:
                u[3] = dis_y
            if dis_y < 0:
                u[4] = -dis_y
            if sum(u) == 0:
                u[0] = 1

        else:
            theta = theta - math.pi / 2
            dis_x = math.cos(theta * math.pi)
            dis_y = math.sin(theta * math.pi)
            if dis_x > 0:
                u[1] = dis_x
            if dis_x < 0:
                u[2] = -dis_x
            if dis_y > 0:
                u[3] = dis_y
            if dis_y < 0:
                u[4] = -dis_y
            if sum(u) == 0:
                u[0] = 1

        if prey_x+v_prey*u[1] >= self.max_edge:
            u[1] = (self.max_edge-prey_x)/v_prey
        if prey_x-v_prey*u[2] <= -self.max_edge:
            u[2] = -(self.max_edge-prey_x)/v_prey
        if prey_y+v_prey*u[3] >= self.max_edge:
            u[3] = (self.max_edge-prey_y)/v_prey
        if prey_y-v_prey*u[4] <= -self.max_edge:
            u[4] = -(self.max_edge-prey_y)/v_prey

        return u