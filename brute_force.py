#!/usr/bin/env python3

import z3
import glob
import random
import json
import os
import numpy as np

from time import process_time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

PROBLEMS = "instances/**/*.smt2*"
TRAINING = 10
CLUSTERS = 4
STRATEGY_DEPTH = 2
TIMEOUT = 3000
TRIALS = 10
CHILD_EXITED = False
CURRENT_TIME = 0


# "memory",           amount of used memory in megabytes.
# "depth",            depth of the input goal.
# "size",             number of assertions in the given goal.
# "num-exprs",        number of expressions/terms in the given goal.
# "num-consts",       number of non Boolean constants in the given goal.
# "num-bool-consts",  number of Boolean constants in the given goal.
# "num-arith-consts", number of arithmetic constants in the given goal.
# "num-bv-consts",    number of bit-vector constants in the given goal.
# "arith-max-deg",    max polynomial total degree of an arithmetic atom.
# "arith-avg-deg",    avg polynomial total degree of an arithmetic atom.
# "arith-max-bw",     max coefficient bit width.
# "arith-avg-bw",     avg coefficient bit width.

probes = [
    'size',
    'num-exprs',
    'num-consts',
    'arith-avg-deg',
    'arith-max-bw',
    'arith-avg-bw'
]

TACTICS = [
    "simplify",
    "propagate-values",
    "propagate-ineqs",
    "normalize-bounds",
    "solve-eqs",
    "elim-uncnstr",
    "add-bounds",
    "pb2bv",
    "lia2pb",
    "ctx-simplify",
    "bit-blast",
    "max-bv-sharing",
    "aig",
    "sat",
    "skip",
]

def sigh(signum, frame):
    global CHILD_EXITED
    if signum == signal.SIGALRM:
        print("custom thread didn't finish before the deadline!")
        exit(1)

    if signum == signal.SIGCHLD:
        (pid, status) = os.waitpid(-1, 0)
        CHILD_EXITED = status == 0

def brute_force(problems, strategy_depth, timeout, trials):
    global CHILD_EXITED
    global CURRENT_TIME
    results = {}
    for problem, cluster in problems:
        print("Cluster #%d" % cluster)
        g = z3.Goal()
        g.add(z3.parse_smt2_file(problem))

        start = process_time()
        s = Tactic("smt")
        result = t(g).as_expr()
        finish = process_time()

        curr_min = finish - start
        best_strategy = ["smt"]


        for _ in range(trials):
            new_strategy = list(np.random.choice(TACTICS, size=strategy_depth, replace=False))
            new_strategy += ["smt"]

            print("Candidate Strategy:", new_strategy)

            t = z3.Then(*new_strategy)

            CHILD_EXITED = False
            CURRENT_TIME = -1
            if (os.fork() == 0):
                signal.signal(signal.SIGALRM, sigh) #sets sigh as handler for SIGALRM
                signal.alarm(timeout / 1000) #process will receive alarm in t/1000 seconds, forcing exit

                start = process_time()
                try:
                    result = t(g).as_expr()
                except:
                    print("timeout")
                    exit(1)

                finish = process_time()

                CURRENT_TIME = finish - start
                CHILD_EXITED = True
                exit(0)



            else:
                signal.signal(signal.SIGCHLD, sigh)
                time.sleep(timout / 1000 + 5) #will be interrupted on child exit
                if not CHILD_EXITED:
                    result = z3.unknown
                    CURRENT_TIME = float('inf')



            print("Result:", result, "\nIn", CURRENT_TIME)

            if ((curr_min == -1 or CURRENT_TIME < curr_min) and (result == z3.unsat or result ==  z3.sat)):
                print("Solved in", finish-start, "seconds")
                curr_min = CURRENT_TIME
                best_strategy = new_strategy

        results[cluster] = best_strategy
    return results

def probe(smtlib):
    g = z3.Goal()
    g.add(z3.parse_smt2_file(smtlib))
    results = [z3.Probe(x)(g) for x in probes]
    return results

def main():
    problems = glob.glob(PROBLEMS, recursive=True)

    problems = random.sample(problems, TRAINING)

    data = []
    for problem in problems:
        data.append(probe(problem))
    kmeans = KMeans(init='k-means++', n_clusters=CLUSTERS, n_init=10)
    kmeans.fit(data)

    probe_heading = ",".join(probes)



    solvable_problems, solvable_data = [], []
    for i in range(len(problems)):
        problem = problems[i]
        result = z3.Solve(z3.parse_smt2_file(problem))
        if result == z3.sat or result == z3.unsat:
            solvable_problems.append(problem)
            solvable_data.append(data[i])


    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, solvable_data)
    toSolve = [(solvable_problems[closest[i]], i) for i in range(len(closest))]

    print("\n\n")
    for x in toSolve:
        print(x[0])

    final_strategies = brute_force(toSolve, STRATEGY_DEPTH, TIMEOUT, TRIALS)

    with open("tmp.json", 'w') as outfile:
        json.dump(final_strategies, outfile)


if __name__ == '__main__':
    main()
