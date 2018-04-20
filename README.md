# scalable_maddpg
scalable multi agent reinforcement learning. Details can be found in the [Report](scalable-multi-agent.pdf)

## to do list
- [x] tune L2, does LSTM parameters need L2 regulizer?
- [x] fix environments
- [x] fix rewards
- [x] decrease the frequency of summaries
- [x] rearrange main.py
- [x] prey boundary problem
- [x] modify the initial position of the agents and prey
- [ ] add another network for prey
- [ ] add summary for rewards of each episode
## Alternative to Gym
An alternative of Gym environment is created (env.py). The rendering implementation of the envrionment is matplot. So, it would be much easier to use. However, you need to implement the prey policy by yourself. 
## Results
Here, we have done two independent runs. In each run, from episode 1 to episodes 3x10^4, three agents were in the game. At episode 3x10^4, we added three more agents into this game. Here we show the mean Q value of all the agents in our experiments. 

<img src="https://github.com/livey/scalable_maddpg/blob/master/Notes/fig1.png" width="500"  />

## Demo results
In this demon, the prey walks randomly. Agents learn to catch the prey. 

<img src="https://github.com/livey/scalable_maddpg/blob/master/Notes/replay.gif" width="500"  />
