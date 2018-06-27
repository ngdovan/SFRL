# -*- coding: utf-8 -*-
import random
# import gym
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras import backend as K
from game import mygym as gym
# import warnings as w
# w.simplefilter(action='ignore', category=FutureWarning)
# w.resetwarnings()


EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size, myId=0, random_planing=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = .01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(use_per_session_threads=True))
        with self.graph.as_default(), self.session.as_default():
            self.model = self._build_model()
            init_op = tf.global_variables_initializer()
            self.session.run(init_op)
        self.target_session = tf.Session(graph=self.graph, config=tf.ConfigProto(use_per_session_threads=True))
        with self.graph.as_default(), self.session.as_default():
            self.target_model = self._build_model()
            init_op = tf.global_variables_initializer()
            self.target_session.run(init_op)
        self.update_target_model()
        self.last_rand_act = 0
        self.last_reward = 0
        self.Id = myId
        self.random_planing = random_planing

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))  # input_dim=self.state_size,
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        with self.graph.as_default(), self.session.as_default():
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.last_reward = reward
        self.last_rand_act = action

    def act(self, state):
        if self.random_planing or np.random.rand() <= self.epsilon:
            # ran = self.last_rand_act
            # if self.last_reward < -0.01:
            #     ran = random.randrange(self.action_size)
            #     while ran == self.last_rand_act:
            #         ran = random.randrange(self.action_size)
            # else:
            #     p = random.randint(1, 101)
            #     if p > 90:
            #         ran = random.randrange(self.action_size)
            # return ran
            return random.randrange(self.action_size)
        with self.graph.as_default(), self.session.as_default():
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        with self.graph.as_default(), self.session.as_default():
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = self.model.predict(state)
                # print(target)
                if done:
                    target[0][action] = reward
                else:
                    a = self.model.predict(next_state)[0]
                    t = self.target_model.predict(next_state)[0]
                    target[0][action] = reward + self.gamma * t[np.argmax(a)]
                self.model.fit(state, target, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        with self.graph.as_default(), self.session.as_default():
            self.model.load_weights(name)

    def save(self, name):
        with self.graph.as_default(), self.session.as_default():
            self.model.save_weights(name)
