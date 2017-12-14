# scalable_maddpg
salable multi agent reinforcement learning

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

## results
Here, we have done two independent runs. In each run, from episode 1 to episodes 3x10^4, three agents were in the game. At episode 3x10^4, we added three more agents into this game. Here we show the mean Q value of all the agents in our experiments. 
![image](https://github.com/livey/scalable_maddpg/blob/master/Notes/fig1.png)

### demon results
![image](https://github.com/livey/scalable_maddpg/blob/master/Notes/replay.gif)
