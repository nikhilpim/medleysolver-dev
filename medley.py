#!/usr/bin/env python3
import z3
import os
import glob
import sys
import numpy as np
import signal
import datetime
import subprocess
from numpy.linalg import norm
from scipy import spatial
from time import process_time
from collections import namedtuple, OrderedDict
import pickle
import random
from compute_features import get_features
from exponential import ExponentialDist
from common import *

# arguments
TIMEOUT = 60.0

# data
CSV_HEADER  = "Instance,Result,Time\n"
Result      = namedtuple('Result', ('problem', 'result', 'elapsed'))

# constants
SAT_RESULT     = 'sat'
UNSAT_RESULT   = 'unsat'
UNKNOWN_RESULT = 'unknown'
TIMEOUT_RESULT = 'timeout (%.1f s)' % TIMEOUT
ERROR_RESULT   = 'error'

TIMERS = OrderedDict({solver:ExponentialDist(l=5) for solver in SOLVERS.keys()})

def run_problem(solver, invocation, problem, timeout):
    # pass the problem to the command
    print(solver)
    command = "%s %s" %(invocation, problem)
    # get start time
    start = datetime.datetime.now().timestamp()
    # run command
    process = subprocess.Popen(
        command,
        shell      = True,
        stdout     = subprocess.PIPE,
        stderr     = subprocess.PIPE,
        preexec_fn = os.setsid
    )
    # wait for it to complete
    try:
        process.wait(timeout=timeout)
    # if it times out ...
    except subprocess.TimeoutExpired:
        # kill it
        print('TIMED OUT:', repr(command), '... killing', process.pid, file=sys.stderr)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except:
            pass
        # set timeout result
        elapsed = timeout
        output  = TIMEOUT_RESULT
    # if it completes in time ...
    else:
        # measure run time
        end     = datetime.datetime.now().timestamp()
        elapsed = end - start
        # get result
        stdout = process.stdout.read().decode("utf-8", "ignore")
        stderr = process.stderr.read().decode("utf-8", "ignore")
        output = output2result(problem, stdout + stderr)
    # make result
    result = Result(
        problem  = problem.split("/", 2)[-1],
        result   = output,
        elapsed  = elapsed if output == 'sat' or output == 'unsat' else timeout
    )
    return result

class Solved_Problem:
    def __init__(self, problem, datapoint, solve_method, time, result):
        self.problem = problem
        self.datapoint = datapoint
        self.solve_method = solve_method
        self.time = time
        self.result = result

def add_strategy(problem, datapoint, solver, solved, all):
    """Returns success or failure of entering problem into solved"""
    elapsed = 0
    _,res = None,None
    rewards = []
    solver_list = solver
    for s in list(solver_list):
        solver = SOLVERS[s]
        if s == solver_list[-1]:
            res = run_problem(s, solver, problem, TIMEOUT-elapsed)
        else:
            res = run_problem(s, solver, problem, 0.05)

        elapsed += res.elapsed
        rewards.append((1 - (len(SOLVERS) * res.elapsed) / TIMEOUT) ** 4)
        if elapsed > TIMEOUT:
            elapsed = TIMEOUT
            break
        if (res.result == SAT_RESULT or res.result == UNSAT_RESULT):
            TIMERS[s].add_sample(res.elapsed)
            solved.append(Solved_Problem(problem, datapoint, s, res.elapsed, res.result))
            break

    all.append(Solved_Problem(problem, datapoint, solver, elapsed, res.result))
    return rewards


def main(problem_dir):
    problems = glob.glob(problem_dir, recursive=True)
    problems = np.random.choice(problems, size=min(TRAINING_SAMPLE, len(problems)), replace=False)

    solved = []
    all = []
    success = False
    ctr = 0

    alternative_times = []
    last_five = []
    for prob in problems:
        point = np.array(get_features(prob))
        last_five.append(point)
        point = point / (np.array(last_five).max(axis=0)+ 1e-10)
        if len(last_five) > 5: last_five.pop(0)
        start = datetime.datetime.now().timestamp()
        if solved and np.random.rand() >= EPSILON * (EPSILON_DECAY ** ctr):
            candidates = sorted(solved, key=lambda entry: norm(entry.datapoint - point))[:len(solved) // 10 + 1]
            fast = sorted(candidates, key=lambda entry: entry.time)
            order = list(OrderedDict((x.solve_method, True) for x in fast).keys())
            remaining = [x for x in SOLVERS.keys() if x not in order]
            random.shuffle(remaining)
            order = order + remaining
            success = add_strategy(prob, point, order, solved, all)
        else:
            r = list(SOLVERS.keys())
            random.shuffle(r)
            success = add_strategy(prob, point, r, solved, all)
        ctr += 1
        end = datetime.datetime.now().timestamp()
        alternative_times.append(end-start)


    res = [(entry.problem, entry.result, entry.solve_method, entry.time) for entry in all]
    res = [t[3] for t in res]
    with open("medley_times.pickle", "wb") as f:
        pickle.dump(res, f)

    with open("medley_all.pickle", "wb") as f:
        pickle.dump([(entry.problem, entry.result, entry.solve_method, entry.time) for entry in all], f)



if __name__ == '__main__':
    np.random.seed(234971)
    main(PROBLEM_DIR)
