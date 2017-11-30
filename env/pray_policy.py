import numpy as np


max_edge = 0.2
class Pray:
    def __init__(self):
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
        proba = np.random.random()
        if proba > 0.8:
            u = np.zeros(5)  # 5-d because of no-move action
            if np.random.random() < 0.5: u[1] += 1.0
            if np.random.random() < 0.5: u[2] += 1.0
            if np.random.random() < 0.5: u[3] += 1.0
            if np.random.random() < 0.5: u[4] += 1.0
            if sum(u) == 0:
                u[0] += 1.0
            if u[1] > 0 and prey_x >= max_edge:
                u[1] = 0
            if u[2] > 0 and prey_x <= -max_edge:
                u[2] = 0
            if u[3] > 0 and prey_y >= max_edge:
                u[3] = 0
            if u[4] > 0 and prey_y <= -max_edge:
                u[4] = 0
        else:
            x = gravity_x - prey_x
            y = gravity_y - prey_y

            u = np.zeros(5)
            if x < 0: u[1] += 1
            if x > 0: u[2] += 1
            if y > 0: u[4] += 1
            if y < 0: u[3] += 1
            if sum(u) == 0:
                u[0] += 1.0
            if u[1] > 0 and prey_x >= max_edge:
                u[1] = 0
            if u[2] > 0 and prey_x <= -max_edge:
                u[2] = 0
            if u[3] > 0 and prey_y >= max_edge:
                u[3] = 0
            if u[4] > 0 and prey_y <= -max_edge:
                u[4] = 0

        return u