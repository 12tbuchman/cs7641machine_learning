from mlrose import DiscreteOpt, Knapsack, random_hill_climb, simulated_annealing, genetic_alg, mimic

problem = DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)