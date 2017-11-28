import numpy as np
from envs import Environ

env = Environ(3)
env.reset()
env.render()

env.render()

env.render()

while(True):
    next_state, reward, done = env.step(np.ones((4, 2)))
    env.re_create_env(4)
    env.render()
    print(next_state)