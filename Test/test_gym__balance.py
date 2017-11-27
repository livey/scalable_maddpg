import gym
env =gym.make('CartPole-v0')
env.reset()
for _ in range(100000):
    env.render()
    if env.step(env.action_space.sample()):
        env.reset()