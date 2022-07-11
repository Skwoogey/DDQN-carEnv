import numpy as np
import random
from IPython.display import clear_output
from collections import deque

from env import CarEnvironment, Vector2

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten, BatchNormalization 
from tensorflow.keras.optimizers import Adam, RMSprop

action_dict = [
    Vector2( 0.0,  0.0), # don't move
    Vector2(-1.0,  0.0), # turn left
    Vector2( 1.0,  0.0), # turn right
    Vector2( 0.0,  1.0), # forward
    Vector2(-1.0,  1.0), # forward left
    Vector2( 1.0,  1.0), # forward right
    Vector2( 0.0, -1.0), # backward
    Vector2(-1.0, -1.0), # backward left
    Vector2( 1.0, -1.0), # backward right
]

enviroment = CarEnvironment("track1.txt", "track1_gates.txt", Vector2(100, 370), -np.pi/2.0)

NUM_OF_INPUTS = 22

class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = NUM_OF_INPUTS
        self._action_size = 9
        self._optimizer = optimizer
        self.maxlen = 10000
        self.batch_size = 128
        
        self.expirience_replay = deque(maxlen=self.maxlen)
        self.states = np.ndarray((self.batch_size, 1, self._state_size))
        self.next_states = np.ndarray((self.batch_size, 1, self._state_size))
        self.target = np.ndarray((self.batch_size, 1, self._action_size))
        self.next_target = np.ndarray((self.batch_size, 1, self._action_size))
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.tau = 0.1
        self.epsilon_high = 0.5
        self.epsilon_low = 0.01
        self.decay_period = 100000.0
        self.epsilon = self.epsilon_high
        self.replay_size = 0
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.hard_update()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
        self.replay_size = min(self.replay_size + 1, self.maxlen)
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self._state_size)))
        model.add(Dense(64, activation='relu', ))
        #model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(self._action_size,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                    bias_initializer=tf.keras.initializers.Constant(-0.2)))
        
        model.compile(loss='huber', optimizer=self._optimizer)
        return model

    def soft_update(self):
        target_weights = np.array(self.target_network.get_weights())
        local_weights = np.array(self.q_network.get_weights())
        self.target_network.set_weights((1.0 - self.tau) * target_weights +local_weights * self.tau)

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state, t = 0.0):
        self.epsilon = self.epsilon_high - min(1.0, t / self.decay_period)*(self.epsilon_high - self.epsilon_low)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(9)
        
        q_values = self.q_network.predict(np.reshape(state, (1, 1, self._state_size)))
        return np.argmax(q_values[0])

    def act_only(self, state):
        q_values = self.q_network.predict(np.reshape(state, (1, 1, self._state_size)))
        return np.argmax(q_values[0])

    def retrain(self, use_last=False):
        minibatch = None
        if use_last:
            minibatch = [self.expirience_replay[x] for x in range(-self.batch_size, self.replay_size)]
        else:
            minibatch = random.sample(self.expirience_replay, self.batch_size)
        
        
        for i in range(self.batch_size):
            self.states[i] = minibatch[i][0]
            self.next_states[i] = minibatch[i][3]
            
        self.target = self.q_network.predict(self.states)
        self.next_target = self.target_network.predict(self.next_states)
        #print('pred target', self.target)
        
        for i in range(self.batch_size):
            if minibatch[i][4]:
                self.target[i][minibatch[i][1]] = minibatch[i][2]
            else:
                self.target[i][minibatch[i][1]] = minibatch[i][2] + self.gamma * np.amax(self.next_target[i, 1])
            
        #print(self.states.shape, self.target.shape)
        #print('new target', self.target)
        self.q_network.fit(self.states, self.target, verbose=0)

    def __len__(self):
        return self.replay_size

optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

def log(*objects):
    print(*objects, file=log.file)
    print(*objects)
log.file = open("log.txt", "w")

num_of_episodes = 25000
agent.q_network.summary()
full_state = np.ndarray((1, NUM_OF_INPUTS), dtype=np.float64)
full_next_state = np.ndarray((1, NUM_OF_INPUTS), dtype=np.float64)
zeros = np.zeros((1, NUM_OF_INPUTS), dtype=np.float64)
total_steps = 0
showcase_freq = 5
steps_per_episode = 50
steps_for_crossing_gate = -20

for e in range(0, num_of_episodes):
    # Reset the enviroment
    full_state[0] = np.array(enviroment.reset())
    #full_state[1] = zeros
    # Initialize variables
    total_reward = 0
    log("episode:", e)
    log("epsilon", agent.epsilon)
    timestep = 0
    while timestep != steps_per_episode:
        #print(timestep)
        # Run Action
        action = agent.act(full_state, total_steps)
        
        # Take action    
        next_state, reward, terminated, info = enviroment.step(action_dict[action]) 
        #print(reward)
        
        full_next_state[0] = np.array(next_state)
        #full_next_state[1] = (full_next_state[0] - full_state[0]) * 20.0
        #print(next_state.shape)
        agent.store(full_state, action, reward, full_next_state, terminated)
        
        full_state = full_next_state
            
        if len(agent) > agent.batch_size:
            agent.retrain()
            #agent.soft_update()

        if (total_steps + 1) % 100 == 0:
            print('target model update')
            agent.hard_update()
            pass

        if info:
            print("gate cleared")
            timestep += steps_for_crossing_gate

        timestep += 1
        total_steps += 1
        total_reward += reward
        enviroment.render()
        if terminated:
            break

    log("Total reward: ", total_reward)

    if (e + 1) % showcase_freq == 0 and True:
        #agent.target_network.save_weights("models\\car_" + str(e + 1) + ".h5")
        log("Showcase")
        full_state[0] = np.array(enviroment.reset())
        #full_state[1] = zeros

        terminated = False
        total_reward = 0
        timestep = 0
        while timestep != steps_per_episode:
            action = agent.act_only(full_state)
            next_state, reward, terminated, info = enviroment.step(action_dict[action])
            full_next_state[0] = np.array(next_state)
            #full_next_state[1] = (full_next_state[0] - full_state[0]) * 20.0
            total_reward += reward
            full_state = full_next_state

            timestep += 1
            if info:
                timestep += steps_for_crossing_gate

            enviroment.render()
            if terminated:
                break

        log("showcase reward: ", total_reward)

log.file.close()