import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario





class Scenario(BaseScenario):
    def __init__(self,numOfadversaries,max_edge):
        self.num_adversaries = numOfadversaries
        self.max_edge=max_edge
        self.num_good_agents = 1
        self.num_agents = self.num_adversaries + self.num_good_agents

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.agents_num=self.num_agents
        num_landmarks =0
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            ##############
            agent.index = i
            
            agent.collide = True
            agent.silent = True
            
            ##########################
            agent.adversary = True if i < self.num_adversaries else False
            
            agent.size = 0.075 if agent.adversary else 0.05
            #agent.accel = 3.0 if agent.adversary else 4.0
            agent.accel = 200.0 if agent.adversary else 250.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.max_edge, +self.max_edge, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9*self.max_edge, +0.9*self.max_edge, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
     #   main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        adversaries = self.adversaries(world)
        main_reward =0
        if agent.adversary:
            Rewa = self.adversary_reward(agent,world)
            main_reward = Rewa[agent.index]
        return main_reward

    #def agent_reward(self, agent, world):
    #    # Agents are negatively rewarded if caught by adversaries
    #    rew = 0
      #  shape = False
      #  adversaries = self.adversaries(world)
      #  if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
      #      for adv in adversaries:
      #          rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
      #  if agent.collide:
      #      for a in adversaries:
      #          if self.is_collision(a, agent):
      #              rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        #def bound(x):
        #    if x < 0.9:
        #        return 0
        #    if x < 1.0:
        #        return (x - 0.9) * 10
        #    return min(np.exp(2 * x - 2), 10)
        #for p in range(world.dim_p):
        #    x = abs(agent.state.p_pos[p])
        #    rew -= bound(x)

        #return rew

    
    #############################
    def adversary_reward(self,agent,world):
        rew=np.zeros((self.num_adversaries,1))
        shape=True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                for a in agents:
                   rew[adv.index] -= 0.1*np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
        if agent.collide:
                for ag in agents:
                    for adv in adversaries:
                        if self.is_collision(ag,adv):
                            rew += 8
                            rew[adv.index] +=7
                        if abs(adv.state.p_pos[0])>self.max_edge or abs(adv.state.p_pos[1])>self.max_edge:
                            rew[adv.index] -= 20
                        #else:
                        #    for rueadv in adversaries:
                        #        for a in agents:
                        #            rew[adv.index] += 0.1 / np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
        return rew
    
    ##
    #
    #
    ##############################
    def observation(self,agent,world):
        global_observation=np.zeros((self.num_agents-1,2*world.dim_p))
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for ag in adversaries:
            for i in range(0,world.dim_p):
                global_observation[ag.index,i]= ag.state.p_pos[i]
        for a in agents:
            for j in range(0,self.num_agents-1):
                for i in range(world.dim_p,2*world.dim_p):
                    global_observation[j,i]= a.state.p_pos[i-world.dim_p]
        #n = world.dim_p*(self.num_agents-1)*2
        #G = np.reshape(global_observation,[n])
        G=global_observation
        return G
    
    
    
    


