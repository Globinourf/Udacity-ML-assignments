import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
#import sys

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
	self.Q = self.Q_init()
	self.alpha = 0.1 #float(sys.argv[1]) # 0.1 or 0.4 or 0.7
	self.gamma = 0.4 #float(sys.argv[2]) # 0.1 or 0.4 or 0.7
	self.epsilon = 0.01 #float(sys.argv[3]) # 0.01 or 0.05 or 0.1
	self.possible_directions = [None, 'forward', 'left', 'right']
	self.state = {
		'next_waypoint': None,
		'light': 'green',
		'oncoming': None,
		'right': None,
		'left': None
        }
	self.last_action = None
	self.last_reward = 0

    def Q_init(self):
	return np.ones((512,4))

    def get_state_index(self, state):
	index = self.possible_directions.index(state['next_waypoint']) + ['green', 'red'].index(state['light'])*4 + self.possible_directions.index(state['oncoming'])*4*2 + self.possible_directions.index(state['right'])*4*2*4 + self.possible_directions.index(state['left'])*4*2*4*4
	return index

    def get_action_index(self, action):
	return self.possible_directions.index(action)

    def get_max_Q(self, state):
	Qs = []
	for action in self.possible_directions:
		Qs.append(self.Q[self.get_state_index(state), self.get_action_index(action)])
	return np.max(Qs)

    def choose_action(self):
	Qs = list(map(lambda x: self.Q[self.get_state_index(self.state), self.get_action_index(x)], self.possible_directions))
	#print 'Qs :', Qs
	if (Qs[0] == Qs[1] and Qs[1] == Qs[2] and Qs[2] == Qs[3]):
		action_to_take = random.choice(self.possible_directions)
	else:
		action_to_take = self.possible_directions[Qs.index(np.max(Qs))]
	if (random.random() < self.epsilon):
		#print 'action_to_take was randomly chosen by epsilon'
		action_to_take = random.choice(self.possible_directions)
	#print 'action_to_take :', action_to_take	
	return action_to_take

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
	self.state = {
		'next_waypoint': None,
		'light': 'green',
		'oncoming': None,
		'right': None,
		'left': None
        }
	self.last_action = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
	previous_state = self.state
	self.state = {
		'next_waypoint': self.next_waypoint,
		'light': inputs['light'],
		'oncoming': inputs['oncoming'],
		'right': inputs['right'],
		'left': inputs['left']
        }
        # TODO: Select action according to your policy
	previous_action = self.last_action
        #action = random.choice(self.possible_directions)
	action = self.choose_action()
	self.last_action = action

        # Execute action and get reward
	previous_reward = self.last_reward
        reward = self.env.act(self, action)
	self.last_reward = reward

        # TODO: Learn policy based on state, action, reward
	index_previous_state = self.get_state_index(previous_state)
	index_previous_action = self.get_action_index(previous_action)
	max_Q = self.get_max_Q(self.state)

	self.Q[index_previous_state, index_previous_action] = (1-self.alpha)*self.Q[index_previous_state, index_previous_action] + self.alpha*(previous_reward + self.gamma*max_Q)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
