# based on https://towardsdatascience.com/an-introduction-to-genetic-algorithms-the-concept-of-biological-evolution-in-optimization-fc96e78fa6db

import numpy as np
from scipy.spatial.distance import cdist

def evaluate(x):
    to_int = (x * [128, 64, 32, 16, 8, 4, 2, 1]).sum()
    return np.sin(to_int / 256 * np.pi)

def initialize():
    return np.random.choice([0, 1], size=8)

def select(n_matings, n_parents):
    return np.random.randint(0, n_matings, (n_matings, n_parents))

def crossover(parent_a, parent_b):
    rnd = np.random.choice([False, True], size=8)

    offspring = np.empty(8, dtype=np.bool)
    offspring[rnd] = parent_a[rnd]
    offspring[~rnd] = parent_b[~rnd]
    return offspring

def mutate(o):
    rnd = np.random.random(8) < 0.125

    mut = o.copy()
    mut[rnd] = ~mut[rnd]
    return mut

def survival(f, n_survivors):
    return np.argsort(-f)[:n_survivors]

def eliminate_duplicates(X):
    D = cdist(X, X)
    D[np.triu_indices(len(X))] = np.inf
    return np.all(D > 1e-32, axis=1)


if __name__ == '__main__':
    pop_size = 5
    n_gen = 15

    # fix random seed
    np.random.seed(1)

    # initialization
    X = np.array([initialize() for _ in range(pop_size)])
    F = np.array([evaluate(x) for x in X])

    # for each generation execute the loop until termination
    for k in range(n_gen):

        # select parents for the mating
        parents = select(pop_size, 2)

        # mating consisting of crossover and mutation
        _X = np.array([mutate(crossover(X[a], X[b])) for a, b in parents])
        _F = np.array([evaluate(x) for x in _X])

        # merge the population and offsprings
        X, F = np.row_stack([X, _X]), np.concatenate([F, _F])

        # perform a duplicate elimination regarding the x values
        I = eliminate_duplicates(X)
        X, F = X[I], F[I]

        # follow the survival of the fittest principle
        I = survival(F, pop_size)
        X, F = X[I], F[I]

        # print the best result each generation
        print(k+1, F[0], X[0].astype(np.int))