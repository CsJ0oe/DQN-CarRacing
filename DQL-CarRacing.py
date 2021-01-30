# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D, Dense, Flatten, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym


# %%
env = gym.make('CarRacing-v0')
print(env.observation_space)
print(env.observation_space.shape)
print(env.action_space)
print(env.action_space.shape)


# %%
import numpy as np

class CarRacingDiscrit:

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.action_space = 10*10*10
        self.observation_space = 96*96*3

    def step(self, action):
        v1 = int(     action        ) % 10
        v2 = int( int(action) / 10  ) % 10
        v3 = int( int(action) / 100 ) % 10
        v1 = ( v1 - 5 ) / 5
        v2 = ( v2     ) / 10
        v3 = ( v3     ) / 10
        state, reward, done, info = self.env.step([v1, v2, v3])
        return state, reward, done, info
 
    def seed(self, s):
        return env.seed(s)

    def reset(self):
        return self.env.reset()
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# %%
# Get the environment and extract the number of actions.
env = CarRacingDiscrit()
#env.seed(123)
nb_actions = 10*10*10
print(env.observation_space)
print(env.action_space)


# %%
# Next, we build a very simple model.
model = Sequential()
model.add(Reshape((96, 96, 3), input_shape=(1, 96, 96, 3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8192, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1000, activation="relu"))
model.summary()


# %%
# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length=1)
agent = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=nb_actions,
                 nb_steps_warmup=500, target_model_update=1e-2)
agent.compile(Adam(lr=1e-3), metrics=['mse'])


# %%
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=1000)


# %%
# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)


# %%
# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, nb_max_episode_steps=1000, visualize=False)


# %%



