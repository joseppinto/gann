import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import pandas as pd

# Weights of recall, accuracy and auc respectively
FUNCTION_WEIGHTS = (0.5, 0.5,)

# INITIAL CONFIGURATION SPACE
# Possible batch sizes in first generation of nn architectures
INITIAL_BATCH_SIZES = [8, 16, 32]
# Possible initial classification thresholds
INITIAL_CLASSIF_THRESHOLDS = [0.25, 0.5, 0.75]
# Possible learning rates in first generation of nn architectures
INITIAL_LEARNING_RATES = [0.1, 0.01, 0.001]
# Possible initial dense layer sizes
INITIAL_DENSE_SIZE = [16, 32, 64]
# Possible dense layer activations
# (encoded in decimal part since we're using floats and we can't have decimal places in layer sizes)
DENSE_ACTIVATIONS = {.1: 'tanh',
                     .2: 'relu',
                     .3: 'linear',
                     .4: 'sigmoid'
                     }
# Possible initial dropout rates
INITIAL_DROPOUT_RATES = [0, 0.05, 0.1]

# Fraction of the population that gets selected and persists
SELECTED_RATIO = 0.2

# MATING RATE
MATING_RATE = 0.5

# MUTATION RATES
# Chance of an individual going through mutation process
OVERALL_MUTATION_RATE = 1
# Factor that influences batch size mutation probability
BS_MUTATION_FACTOR = .25
# Factor that influences classification threshold mutation probability
CT_MUTATION_FACTOR = .25
# Factor that influences learning rate mutation probability
LR_MUTATION_FACTOR = .5
# Factor that influences dropout layer mutation
DROPOUT_MUTATION_FACTOR = .25
# Factor that influences the mutation probability of dense layers activation
ACTIVATION_MUTATION_FACTOR = .1
# Factor that influences the mutation probability of dense layers' size
LSIZE_MUTATION_FACTOR = .25
# Factor that influences the layer block mutation probability
BLOCK_MUTATION_FACTOR = .2


# Function to initialize individual (used in the first generation)
def init_individual(container):
    ind = np.array([], dtype='float16')
    # Add batch size
    ind = np.append(ind, random.choice(INITIAL_BATCH_SIZES) + random.choice(INITIAL_CLASSIF_THRESHOLDS))
    # Add learning rate
    ind = np.append(ind, random.choice(INITIAL_LEARNING_RATES))

    # Random number of initial layer groups (dense + dropout + normalization)
    # Initial structure for the networks
    n_layer_groups = random.randint(1, 10)
    for i in range(n_layer_groups):
        # Add a dense layer
        ind = np.append(ind, random.choice(list(DENSE_ACTIVATIONS.keys())) + random.choice(INITIAL_DENSE_SIZE))
        # Add a dropout layer
        ind = np.append(ind, random.choice(INITIAL_DROPOUT_RATES))
        # Add a normalization layer
        ind = np.append(ind, -1)

    return container(ind)


def mate(child1, child2):
    # First element is the batch size
    # The children's batcj size is the average of their parents'
    child1[0] = int((child1[0] + child2[0])/2) + \
                (child1[0] - int(child1[0]) + child2[0] - int(child2[0]))/2     # Preserve decimals (classif. threshold)
    child2[0] = child1[0]

    # Same for the learning rate
    child1[1] = (child1[1] + child2[1])/2
    child2[1] = child1[1]

    # For each block, each children has a 50-50 chance of getting
    # that same block equal to each parent
    min_len = int((min(child1.size, child2.size) - 2)/3)
    for block in range(min_len):
        l1 = child1[block + 2]
        l2 = child1[block + 3]
        l3 = child1[block + 4]
        if random.random() > .5:
            child1[block + 2] = child2[block + 2]
            child1[block + 3] = child2[block + 3]
            child1[block + 4] = child2[block + 4]
        if random.random() > .5:
            child2[block + 2] = l1
            child2[block + 3] = l2
            child2[block + 4] = l3


def mutate(container, individual):
    r = np.random.rand(int((individual.size - 2)/3))    # Random numbers that determine mutation in layer blocks
    c = 0                                               # Counter of block variation (they might be deleted or duped)
    for i in range(r.size):
        if r[i] < BLOCK_MUTATION_FACTOR:
            individual = np.delete(individual, [(i + c) * 3 + 2,    # Delete layer block
                                                (i + c) * 3 + 3,
                                                (i + c) * 3 + 4])
            c -= 1
        elif r[i] < 2*BLOCK_MUTATION_FACTOR:
            individual = np.insert(individual, (i + c) * 3 + 5,
                                   individual[(i + c) * 3 + 2:(i + c) * 3 + 5]
                                   )
            c += 1

    r = np.random.rand(individual.size)         # Random numbers that determine mutation in single genes
    # Mutate batch size
    if r[0] < BS_MUTATION_FACTOR:
        individual[0] = int(int(individual[0]) * abs(np.random.normal(1, 1))) + \
                            (individual[0] - int(individual[0]))

    if r[0] < CT_MUTATION_FACTOR:
        individual[0] = int(individual[0]) + \
                            (individual[0] - int(individual[0])) * np.random.normal(1, 0.25)

    # Mutate learning rate
    if r[1] < LR_MUTATION_FACTOR:
        individual[1] *= abs(np.random.normal(1, 1))

    # Mutate single layers
    for i in range(2, individual.size):
        if i % 3 == 2:      # Dense layer
            if r[i] < LSIZE_MUTATION_FACTOR:
                individual[i] = round(int(individual[i] * abs(np.random.normal(1, 1))) +
                                      individual[i] - int(individual[i]), 1)  # Preserve decimal part
            elif r[i] < ACTIVATION_MUTATION_FACTOR + LSIZE_MUTATION_FACTOR:
                individual[i] = round(int(individual[i]) + random.choice(list(DENSE_ACTIVATIONS.keys())), 1)    # Mutate activation
        elif i % 3 == 0:    # Dropout layer
            if r[i] < DROPOUT_MUTATION_FACTOR:
                individual[i] *= abs(np.random.normal(1, 1))
    return container(individual)


def init_ga(eval_func, train_x, train_y, test_x, test_y, eval_epochs=10):
    creator.create("Fitness", base.Fitness, weights=FUNCTION_WEIGHTS)
    creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual",          # Register function to initialize individual
                     init_individual,       # Function that initializes individual
                     creator.Individual
                     )

    toolbox.register("population",          # Register function to initialize population
                     tools.initRepeat,      # Repeat initialization of elements (in this case individuals)
                     list,                  # Base class of the population (list of individuals)
                     toolbox.individual     # Individual initializer
                     )                      # (i.e. neural network architecture)

    # Define evaluation function from function and data passed in arguments
    def evaluate(individual):
        return eval_func(individual, eval_epochs, train_x, train_y, test_x, test_y),

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    # We have to pass the container, because the mutation may cause it to change size
    toolbox.register("mutate", mutate, creator.Individual)
    # We'll use one of the framework's selection functions
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def append_row(df, gen, i):
    f = i.fitness.values
    row = {'gen': gen,
           'fitness': f[0] * FUNCTION_WEIGHTS[0] + f[1] * FUNCTION_WEIGHTS[1],
           'recall': f[0],
           'auc': f[1],
           'layers': i.size - 2,
           'classif_threshold': i[0] - int(i[0]),
           'lr': i[1],
           'batch_size': int(i[0]),
           'architecture': str(list(i))
           }
    return df.append(row, ignore_index=True)


def run_ga(file, toolbox, pop_size=10, max_gens=10):
    df = pd.read_csv(file)
    if df.shape[0] > 0:
        g = max(df['gen'].unique())
        pop = [creator.Individual(np.array(eval(x), dtype='float16')) for x in df['architecture'].values]
        fitness = [(x,) for x in zip(list(df['recall'].values), list(df['auc'].values))]
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit[0]
    else:
        g = 0
        pop = toolbox.population(n=pop_size)
        # Evaluate the entire population
        fitness = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit[0]
            df = append_row(df, g, ind)
        df.to_csv(file, index=False)

    while g < max_gens:
        g = g + 1
        print(f"Gen {g}")
        # Select the next generation individuals
        inds_kept = int(pop_size * SELECTED_RATIO)
        offspring = toolbox.select(pop, inds_kept)
        # Clone the selected individuals
        offspring = [toolbox.clone(x) for x in offspring * int(1 / SELECTED_RATIO)]
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[inds_kept::2], offspring[inds_kept + 1::2]):
            if random.random() < MATING_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for i in range(inds_kept, len(offspring)):
            if random.random() < OVERALL_MUTATION_RATE:
                offspring[i] = toolbox.mutate(offspring[i])

        # Reevaluate individuals that changed since last gen
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit[0]

        pop[:] = offspring

        for i in pop:
            df = append_row(df, g, i)
        df.to_csv(file, index=False)
    return df
