#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:02:41 2025

@author: anton
"""

# %% imports

import inspect
import sys                                                                
sys.path.insert(1, '/home/anton/')                                        
import Zeta 

from Zeta.convex import PositiveOrthant
from Zeta.toric import ToricDatum

class ToricDatumAlgebra:
    def __init__(self, T):
        self._toric_datum = T.simplify()
        self.rank = 0

    def toric_datum(self, objects=None, name=None):
        return self._toric_datum

    def find_good_basis(self, objects=None, name=None):
        return None

    def change_basis(self, A, check=True):
        return self



from Zeta.reps import RepresentationProcessor
from Zeta.ask import AskProcessor

from Zeta.convex import RationalSet
#from Zeta.toric import ToricDatum

from Zeta import logger, smurf, surf, torus, toric, abstract, cycrat, triangulate, reps, subobjects, ask, cico, addmany

from sage.all import *

import bisect
import numpy as np
import sympy as sp
import IPython
import random
import copy
import math
import torch
from torch import nn

# %% import for baselines 

import gymnasium as gym
from stable_baselines3 import DQN




# %% random matrices from GL(n,Z)


def random_GL_nZ(n, num_ops=30, max_coeff=3):
    """
    Generate a "random" invertible n x n integer matrix.
    - num_ops: number of random elementary operations
    - max_coeff: max integer to add/subtract in row operations
    """
    A = Matrix.identity(n)  # identity matrix

    for _ in range(num_ops):
        op = random.choice(["swap", "negate", "add"])
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if op == "swap" and i != j:
            A[i, :], A[j, :] = A[j, :], A[i, :]
        elif op == "negate":
            A[i, :] *= -1
        elif op == "add" and i != j:
            k = random.randint(-max_coeff, max_coeff)
            A[i, :] += k * A[j, :]
    return A


# %% Gymnasium class for the change of basis

class AlgebraBasisEnv(gym.Env):
    """
    Gym environment for algebra basis optimization.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, L, max_steps=10):
        super().__init__()
        self.L = L  # Your Sage algebra object
        #self.L_in = copy.deepcopy(L) # will not change
        self.n = L.rank  # dimension of algebra
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: (type, i, j)
        # 0: swap(i, j), 1: negate(i), 2: add(i, j)
        # We'll encode as a flat discrete space.
        self.action_space = gym.spaces.Discrete(3 * self.n * self.n)

        # observation will be the flattened (1,n,n,n) table of coordinates with the time-step infront 
        obs_shape = (self.n * self.n * self.n+1,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # keep integer transform matrix A
        self.A = Matrix.identity(self.n)
        #weight on previous turn to compute reward
        self.pr_weight = L.toric_datum('subalgebras').weight()

    # ---------- helpers ----------
    def _table_to_ndarray(self):
        """
        Convert self.L.table (list-of-lists of Sage vectors) to a numpy array
        of shape (n, n, n) with float values.
        """
        table = self.L.table  # assumed list-of-lists
        n = self.n
        arr = np.zeros((n, n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                v = table[i][j]

                # assume v is a vector-like object of length n (Sage vector)
                try:
                    coords = list(v)
                except Exception:
                    # if it's not directly listable, try converting via tuple()
                    coords = tuple(v)

                if len(coords) != n:
                    raise ValueError(
                        f"Entry L.table[{i}][{j}] has length {
                            len(coords)}, expected {n}"
                    )

                # convert coordinates to floats (Sage rationals support float())
                for k, c in enumerate(coords):
                    try:
                        arr[i, j, k] = float(c)
                    except Exception:
                        # last-resort: try Sage numeric conversion .n()
                        if hasattr(c, "n"):
                            arr[i, j, k] = float(c.n())
                        else:
                            # fallback to 0.0 (shouldn't happen for well-formed tables)
                            arr[i, j, k] = 0.0
        return arr

    def _get_obs(self):
        arr = self._table_to_ndarray()
        return np.concatenate([
            np.array([self.current_step], dtype=np.float32),
            arr.flatten().astype(np.float32)])

    def _count_zeros(self, arr, tol=1e-9):
        "Count entries that are (close to) zero in a numpy array"
        return int(np.sum(np.isclose(arr, 0.0, atol=tol)))

    # ---------- gym methods ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.L = self.L.change_basis(self.A.inverse())
        self.A = Matrix.identity(self.n)  
        obs = self._get_obs()
        self.pr_weight = L.toric_datum('subalgebras').weight()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # decode action (same encoding you had before)
        t = action // (self.n * self.n)
        i = (action // self.n) % self.n
        j = action % self.n

        if t == 0 and i != j:
            self.A[i, :], self.A[j, :] = self.A[j, :], self.A[i, :]
        elif t == 1:
            self.A[i, :] *= -1
        elif t == 2 and i != j:
            self.A[i, :] += self.A[j, :]
        
        # apply change of basis to L 
        self.L = L.change_basis(self.A)
        print(self.A)
       

        #Getting corresponding toric datum to evaluate the reward
        #tor_d = L.toric_datum('ideals')
        tor_d = self.L.toric_datum('subalgebras')
        tor_w = tor_d.weight()
        
        
        print(tor_w)
        print(self.pr_weight)
        #reward = float(self.pr_weight - tor_w)
        reward = float((self.pr_weight - tor_w)/10 +sign(self.pr_weight - tor_w)) 
        print(reward)
        self.pr_weight=tor_w

        obs = self._get_obs()
        terminated = False  # unless you define a true "solved" condition
        truncated = self.current_step >= self.max_steps
        info = {
            "weight": tor_w,
            "A": copy.deepcopy(self.A)
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        arr = self._table_to_ndarray()
        print("Current A:\n", self.A)
        print("Table shape (n,n,n):", arr.shape)
        print("Example (i,j) -> coordinates:", arr[0, 0])  # adjust as needed


# %% Setting up the algebra (must be Zeta algebra)
# L = Zeta.lookup('(ZZ^3,*)')
T = random_GL_nZ(5)
print(T)
L = Zeta.lookup('Fil4').change_basis(T)
print(L)
env = AlgebraBasisEnv(L)
print(env.L.toric_datum('subalgebras').weight())

# %% initial, unspoiled algebra
env0 = AlgebraBasisEnv(Zeta.lookup('Fil4'))
print(env0.L.toric_datum('subalgebras').weight())

# %% creating environment and model
#env0._count_zeros(env0._table_to_ndarray())
env = AlgebraBasisEnv(L)
#model = DQN("MlpPolicy", env, verbose=1)
print("weight:")
print(env.L.toric_datum('subalgebras').weight())

# %% printing out layers of the model
'''
for name, module in model.policy.q_net.named_children():
    print(name, module)
'''

# %% custom model layers

policy_kwargs = dict(
    net_arch=[256, 256, 256, 128]  # 3 hidden layers
)

model = DQN("MlpPolicy", env,  policy_kwargs=policy_kwargs, verbose=1, learning_rate = 0.0001, gamma=1.0, 
            exploration_fraction=0.3,
            exploration_final_eps=0.05)

# %% training!
#model.verbose = 1  # how much info prints during training 0,1 or 2
model.learn(total_timesteps=100)


# %% analysing and trying the model

# print("Last recorded loss:", model.replay_buffer.sample(batch_size=1))
obs, _ = env.reset()  # reset the invironment to innitial state

# run the agent step by step

done = False
while not done:
    # ask the model for an action
    action, _states = model.predict(obs, deterministic=False)

    # apply the action
    obs, reward, terminated, truncated, info = env.step(action)

    # check if episode ended
    done = terminated or truncated


# %% see the results

print("Final basis matrix A:")
print(env.A)

print("Final multiplication table:")
print(env.L.table)

#print("Number of zeros:")
#env._count_zeros(env._table_to_ndarray())

print("weight:")
print(env.L.toric_datum('subalgebras').weight())
