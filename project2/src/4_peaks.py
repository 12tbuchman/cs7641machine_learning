import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose import DiscreteOpt, FourPeaks, random_hill_climb, simulated_annealing, genetic_alg, mimic
import random

random.seed(42)
RANDOM_STATE = 42
LENGTH = 20

init_state = [random.randint(0, 1) for _ in range(LENGTH)]
print(init_state)
fitness = FourPeaks()
problem = DiscreteOpt(length=LENGTH, fitness_fn=fitness, maximize=True)

rhc_best_state, rhc_best_fitness = random_hill_climb(problem, max_attempts=10, max_iters=10, restarts=10,
                                                     init_state=init_state, random_state=RANDOM_STATE)
print(rhc_best_state)
print(rhc_best_fitness)

sa_best_state, sa_best_fitness = simulated_annealing(problem, max_attempts=10, max_iters=10, init_state=init_state,
                                                     random_state=RANDOM_STATE)
print(sa_best_state)
print(sa_best_fitness)

ga_best_state, ga_best_fitness = genetic_alg(problem, max_attempts=10, max_iters=10, random_state=RANDOM_STATE)
print(ga_best_state)
print(ga_best_fitness)

mimic_best_state, mimic_best_fitness = mimic(problem, max_attempts=10, max_iters=10, random_state=RANDOM_STATE)
print(mimic_best_state)
print(mimic_best_fitness)
