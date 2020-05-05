#!/usr/bin/env python3
import z3
import glob
import pickle
import numpy as np

from brute_force import probe
from time import process_time
from sklearn.svm import SVC

STRATEGIES = [
    #TO BE HARDCODED
]

def strategy_test(problem, strategy):
    start = process_time()

    g = z3.Goal()
    g.add(z3.parse_smt2_file(problem))
    t = z3.Then(*strategy)
    result = t(g).as_expr()

    end = process_time()

    if result == z3.unknown:
        return float('inf')
    return end - start


def svm(problems, timeout, filename):
    problems = glob.glob(PROBLEMS, recursive=True)

    problems = random.sample(problems, TRAINING)

    data = []
    for problem in problems:
        data.append(probe(problem))

    labels = [np.argmax([-1 * strategy_test(problem, strategy) for strategy in STRATEGIES]) for problem in problems]

    clf = SVC(gamma='auto')
    clf.fit(np.array(data), np.array(labels))

    with open(filename, "wb") as f:
        pickle.dump(clf, f)
