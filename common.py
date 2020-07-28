from collections import namedtuple, OrderedDict


SOLVERS = OrderedDict({
    "Z3"   : "z3 -T:63",
    "CVC4" : "cvc4 --tlimit=63000",
    "BOOLECTOR" : "./tools/boolector-3.2.1/build/bin/boolector -t 63",
    "YICES": "./tools/yices-2.6.2/bin/yices-smt2 --timeout=63"

})

EPSILON = 0.88          #probability with which to randomly search
EPSILON_DECAY = 0.92
TRAINING_SAMPLE = 2000

RESULTS_DIR = "results"
PROBLEM_DIR = "datasets/bv/**/*.smt2"

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


SAT_RESULT     = 'sat'
UNSAT_RESULT   = 'unsat'
UNKNOWN_RESULT = 'unknown'
ERROR_RESULT   = 'error'
