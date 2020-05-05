#!/usr/bin/env python3
import z3
import glob
import numpy as np
from numpy.linalg import norm
from scipy import spatial
from time import process_time

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


STRATEGIES = [
    use_z3_solver,
    use_z3_tactic,
]

EPSILON = 0.05          #probability with which to randomly search
TRAINING_SAMPLE = 10
SPEEDUP_WEIGHT = 0.5
SIMILARITY_WEIGHT = 0.5

PROBLEM_DIR = "./instances/calypto/*.smt2"

PROBES = [
    'size',
    'num-exprs',
    'num-consts',
    'arith-avg-deg',
    'arith-max-bw',
    'arith-avg-bw'
]

class Solved_Problem:
    def __init__(self, problem, datapoint, solve_method, time, result):
        self.problem = problem
        self.datapoint = datapoint
        self.solve_method = solve_method
        self.time = time
        self.result = result

def use_z3_solver(goal):
    s = z3.Solver()
    s.add(goal)
    return s.check()

def probe(smtlib):
    g = z3.Goal()
    g.add(z3.parse_smt2_file(smtlib))
    results = [z3.Probe(x)(g) for x in PROBES]
    return results

def featurize_problems(problem_dir):
    problems = glob.glob(problem_dir, recursive=True)
    problems = np.random.choice(problems, size=TRAINING_SAMPLE)
    data = []
    for problem in problems:
        data.append(probe(problem))
    return problems, np.array(data)

def add_strategy(problem, datapoint, function, solved):
    """Returns success or failure of entering problem into solved"""
    start = process_time()

    g = z3.Goal()
    g.add(z3.parse_smt2_file(problem))
    result = function(g)

    end = process_time()

    if result == z3.unknown:
        return False

    solved.append(Solved_Problem(problem, datapoint, function, end-start, result))
    return True


def main(problem_dir):
    problems, X = featurize_problems(problem_dir)

    solved = []
    success = False
    for prob, point in zip(problems, X):
        if solved and np.random.rand() >= EPSILON:
            closest = min(solved, key=lambda entry: SPEEDUP_WEIGHT * entry.time + SIMILARITY_WEIGHT * norm(entry.datapoint - point))
            success = add_strategy(prob, point, closest.solve_method, solved)

        if not success:
            rand_function = np.random.choice(STRATEGIES)
            add_strategy(prob, point, rand_function, solved)


    print([(entry.problem, entry.result, entry.solve_method) for entry in solved])
    print("Number solved: ", len(solved))
    print("Number unsolved: ", len(problems) - len(solved))



if __name__ == '__main__':
    main(PROBLEM_DIR)
