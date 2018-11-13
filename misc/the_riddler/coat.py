import os
import sys
import json
import datetime
import webbrowser
import numpy as np
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict
from scipy.stats import binom, norm
from flask import Flask, request, render_template, jsonify, redirect

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class Guesser(object):
    def __init__(self, rule):
        self.rule = rule

        self.remaining = np.array(range(1, 1001))
        self.value = np.random.choice(range(1, 1001))

    def guess(self):
        for guess in self.rule:
            guess_val = int(self.remaining.shape[0] * guess) + (self.remaining[0] - 1)
            if guess_val > self.value:
                self.remaining = self.remaining[self.remaining < guess_val]
            elif guess_val < self.value:
                self.remaining = self.remaining[self.remaining > guess_val]
            else:
                return []
        return self.remaining


def minimizer():
    from scipy.optimize import minimize

    fun = lambda x: np.mean([len(Guesser(x).guess()) for _ in range(10000)])
    x0 = [.75] * 8
    bounds = tuple((0, 1) for _ in range(8))
    print(minimize(fun, x0, bounds=bounds))

def main():
    minimizer()

    sys.exit()
    # rule = [.75] * 8
    rule = [.5] * 8
    # rule = [1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2]

    dist = [len(Guesser(rule).guess()) for _ in range(1000)]
    print(np.mean(dist))


if __name__ == '__main__':
    main()