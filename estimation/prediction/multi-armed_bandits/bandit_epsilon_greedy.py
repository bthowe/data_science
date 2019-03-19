import scipy.stats as stats
import numpy as np
import random
import sys
import theano
from theano import as_op
import theano.tensor as tt
from scipy import stats
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
from pymc3 import Continuous
from sklearn.neighbors import KernelDensity
from pymc3.distributions.continuous import Interpolated

# in the command line I ran < theano-cache clear > and things started running a lot faster

class BanditEG(object):
    '''
    Implements an online, learning strategy to solve the Multi-Armed Bandit problem using the epsilon-greedy algorithm,
     where rewards can be continuous or binary and there are no features.

    parameters:
        num_bandits: the number of arms
        epsilon: fraction of time in exploration
    methods:
        choose_arm(arms_to_consider): in exploitation, choose arm to play according highest mean reward, from subset of
         arms in arms_to_consider.
        update(arm, reward): updates the history of rewards for the arm played
    attributes:
        plays: a dictionary with the indexed arms as keys and the value corresponding to the number of times the arm has been played
        rewards: a dictionary with the indexed arms as keys and the value corresponding to the sum of the rewards the arm has received
        N: count of the number of updates
        mean_submissions: mean reward for each arm
    '''

    def __init__(self, num_bandits, epsilon):
        '''
        INPUT: Bandits, function

        Initializes the BanditStrategy given an instance of the Bandits class
        and a choice function.
        '''
        self.num_bandits = num_bandits
        self.epsilon = epsilon
        self.rewards = np.zeros(self.num_bandits).astype(int)
        self.plays = np.zeros(self.num_bandits).astype(int)
        self.N = 0
        self.mean_submissions = self.rewards / self.plays

    def choose_arm(self, arms_to_consider):
        if self.N == 0:
            best_arm = self.num_bandits / 2
        else:
            if random.random() < self.epsilon:  # exploration
                best_arm = np.random.choice(arms_to_consider)
            else:
                best_arm = -1
                best_score = -1
                for arm in arms_to_consider:
                    if self.plays[arm] > 0:
                        score = self.rewards[arm] / self.plays[arm]
                    else:
                        score = 0
                    if score > best_score:
                        best_arm = arm
                        best_score = score
        return best_arm

    def update(self, arm, reward):
        self.rewards[arm] += reward
        self.plays[arm] += 1
        self.mean_submissions = self.rewards / self.plays
        self.N += 1
