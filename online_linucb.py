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
from samplers import ThompsonSampling


# arguments
TIMEOUT = 60.0
RESULTS_DIR = "results"
ALPHA = 2.358
# data
CSV_HEADER  = "Instance,Result,Time\n"
Result      = namedtuple('Result', ('problem', 'result', 'elapsed'))

# constants
SAT_RESULT     = 'sat'
UNSAT_RESULT   = 'unsat'
UNKNOWN_RESULT = 'unknown'
TIMEOUT_RESULT = 'timeout (%.1f s)' % TIMEOUT
ERROR_RESULT   = 'error'

SOLVERS = OrderedDict({
    "Z3"   : "z3 -T:63",
    "CVC4" : "cvc4 --tlimit=63000",
    "BOOLECTOR" : "./tools/boolector-3.2.1/build/bin/boolector -t 63",
    "YICES": "./tools/yices-2.6.2/bin/yices-smt2 --timeout=63"

})

EPSILON = 0.88          #probability with which to randomly search
EPSILON_DECAY = 0.95
TRAINING_SAMPLE = 250
SPEEDUP_WEIGHT = 0.8
SIMILARITY_WEIGHT = 0.2

PROBLEM_DIR = "datasets/qf_abv/*.smt2"

PROBES = [
    'size',
    'num-exprs',
    'num-consts',
    'arith-avg-deg',
    'arith-max-bw',
    'arith-max-bw',
    'arith-avg-bw',
    'depth',
    'num-bool-consts',
    'num-arith-consts',
    'num-bv-consts'
]

def output2result(problem, output):
    # it's important to check for unsat first, since sat
    # is a substring of unsat
    if 'UNSAT' in output or 'unsat' in output:
        return UNSAT_RESULT
    if 'SAT' in output or 'sat' in output:
        return SAT_RESULT
    if 'UNKNOWN' in output or 'unknown' in output:
        return UNKNOWN_RESULT

    # print(problem, ': Couldn\'t parse output', file=sys.stderr)
    return ERROR_RESULT


def run_problem(solver, invocation, problem):
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
        process.wait(timeout=TIMEOUT)
    # if it times out ...
    except subprocess.TimeoutExpired:
        # kill it
        print('TIMED OUT:', repr(command), '... killing', process.pid, file=sys.stderr)
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        # set timeout result
        elapsed = TIMEOUT
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
        elapsed  = elapsed if output == 'unsat' or output == 'sat' else TIMEOUT

    )
    return result

def use_z3_solver(goal):
    s = z3.Solver()
    s.add(goal)
    return s.check()

def use_z3_tactic(goal):
    strategy = ["simplify", "solve-eqs", "smt"]
    t = z3.Then(*strategy)
    try:
        if t(g).as_expr():
            return z3.sat
        else:
            return z3.unsat
    except:
        return z3.unknown

class Solved_Problem:
    def __init__(self, problem, datapoint, solve_method, time, result):
        self.problem = problem
        self.datapoint = datapoint
        self.solve_method = solve_method
        self.time = time
        self.result = result

def probe(smtlib):
    g = z3.Goal()
    g.add(z3.parse_smt2_file(smtlib))
    results = [z3.Probe(x)(g) for x in PROBES]
    return results

def featurize_problems(problem_dir):
    problems = glob.glob(problem_dir, recursive=True)
    problems = np.random.choice(problems, size=min(TRAINING_SAMPLE, len(problems)), replace=False)
    # problems = sorted(problems)
    data = []
    for problem in problems:
        data.append(probe(problem))
    ret = np.array(data)
    ret = ret / (ret.max(axis=0) + 1e-6)
    return problems, ret

def add_strategy(problem, datapoint, solver, solved, all):
    """Returns success or failure of entering problem into solved"""
    res = run_problem(solver, SOLVERS[solver], problem)
    if (res.result == SAT_RESULT or res.result == UNSAT_RESULT):
        solved.append(Solved_Problem(problem, datapoint, solver, res.elapsed, res.result))
    all.append(Solved_Problem(problem, datapoint, solver, res.elapsed, res.result))

    return (1 - res.elapsed/TIMEOUT) ** 3


def main(problem_dir):
    problems = glob.glob(problem_dir, recursive=True)
    problems = np.random.choice(problems, size=min(TRAINING_SAMPLE, len(problems)), replace=False)
    solved = []
    all = []
    success = False
    ctr = 0

    alternative_times = []
    d = len(PROBES)
    thetas = [np.zeros((d, 1)) for _ in SOLVERS]
    As = [np.identity(d) for _ in SOLVERS]
    Bs = [np.zeros((d, 1)) for _ in SOLVERS]
    last_five = []
    for prob in problems:
        point = np.array(probe(prob))
        last_five.append(point)
        point = point / (np.array(last_five).max(axis=0)+ 1e-10)
        if len(last_five) > 5: last_five.pop(0)
        point = point.reshape((len(point), 1))
        # print(ctr, EPSILON * (EPSILON_DECAY ** ctr))
        start = datetime.datetime.now().timestamp()
        thetas = [np.linalg.inv(As[i]) @ Bs[i] for i in range(len(SOLVERS))]
        ps = [thetas[i].T @ point + ALPHA * np.sqrt(point.T @ np.linalg.inv(As[i]) @ point) for i in range(len(SOLVERS))]
        choice = np.random.choice(np.flatnonzero(np.isclose(ps,max(ps)))) #if np.random.rand() >= EPSILON * (EPSILON_DECAY ** ctr) else np.random.choice(list(range(len(SOLVERS))))
        reward = add_strategy(prob, point, list(SOLVERS.keys())[choice], solved, all)
        As[choice] = As[choice] + point @ point.T
        Bs[choice] = Bs[choice] + reward * point
        ctr += 1
        end = datetime.datetime.now().timestamp()
        alternative_times.append(end-start)

    with open("linucb_true.pickle", "wb") as f:
        pickle.dump(alternative_times, f)
    print("all", all)
    print("solved", solved)
    res = [(entry.problem, entry.result, entry.solve_method, entry.time) for entry in all]
    res = [t[3] for t in res]
    with open("linucb_times.pickle", "wb") as f:
        pickle.dump(res, f)

    with open("linucb_all.pickle", "wb") as f:
        pickle.dump([(entry.problem, entry.result, entry.solve_method, entry.time) for entry in all], f)

    print([(entry.problem, entry.result, entry.solve_method, entry.time) for entry in all])
    print("Number solved: ", len(solved))
    print("Number unsolved: ", len(problems) - len(solved))



if __name__ == '__main__':
    np.random.seed(234971)
    main(PROBLEM_DIR)
