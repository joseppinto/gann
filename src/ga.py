import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

# INITIAL CONFIGURATION SPACE
# Possible batch sizes in first generation of nn architectures
INITIAL_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# Possible learning rates in first generation of nn architectures
INITIAL_LEARNING_RATES = [0.1, 0.01, 0.01]
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

# MUTATION RATES
# Chance of an individual going through mutation process
OVERALL_MUTATION_RATE = 0.5
# Factor that influences batch size mutation probability
BS_MUTATION_FACTOR = .05
# Factor that influences learning rate mutation probability
LR_MUTATION_FACTOR = .05
# Factor that influences dropout layer mutation
DROPOUT_MUTATION_FACTOR = .1
# Factor that influences the mutation probability of dense layers activation
ACTIVATION_MUTATION_FACTOR = .2
# Factor that influences the mutation probability of dense layers' size
LSIZE_MUTATION_FACTOR = .1
# Factor that influences the layer block mutation probability
BLOCK_MUTATION_FACTOR = .05


# Function to initialize individual (used in the first generation)
def init_individual():
    ind = np.array(dtype='float16')
    # Add batch size
    ind.append(random.choice(INITIAL_BATCH_SIZES))
    # Add learning rate
    ind.append(random.choice(INITIAL_LEARNING_RATES))

    # Random number of initial layer groups (dense + dropout + normalization)
    # Initial structure for the networks
    n_layer_groups = random.randint(1, 5)
    for i in range(n_layer_groups):
        # Add a dense layer
        ind.append(random.choice(DENSE_ACTIVATIONS.keys()) + random.choice(INITIAL_DENSE_SIZE))
        # Add a dropout layer
        ind.append(random.choice(INITIAL_DROPOUT_RATES))
        # Add a normalization layer
        ind.append(-1)

    return ind


def mate(child1, child2):
    # First element is the batch size
    # The children's batcj size is the average of their parents'
    child1[0] = (child1[0] + child2[0])/2
    child2[0] = child1[0]

    # Same for the learning rate
    child1[1] = (child1[1] + child2[1])/2
    child2[1] = child1[1]

    # For each block, each children has a 50-50 chance of getting
    # that same block equal to each parent
    min_len = (min(child1.size, child2.size) - 2)/3
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


def mutate(individual):
    r = np.random.rand((individual.size - 2))   # Random numbers that determine mutation in layer blocks
    c = 0                                       # Counter of variation of blocks (that might be deleted or duped)
    for i in range(r.size):
        if r < BLOCK_MUTATION_FACTOR:
            np.delete(individual, [(r[i] + c) * 3 + 2,    # Delete layer block
                                   (r[i] + c) * 3 + 3,
                                   (r[i] + c) * 3 + 4])
            c -= 1
        elif r < 2*BLOCK_MUTATION_FACTOR:
            np.insert(individual, [(r[i] + c) * 3 + 2,
                                   (r[i] + c) * 3 + 3,
                                   (r[i] + c) * 3 + 4],
                                  [individual[(r[i] + c) * 3 + 2],
                                   individual[(r[i] + c) * 3 + 3],
                                   individual[(r[i] + c) * 3 + 4]]
                      )
            c += 1

    r = np.random.rand(individual.size)         # Random numbers that determine mutation in single genes
    # Mutate batch size
    if r[0] < BS_MUTATION_FACTOR:
        individual[0] /= 2
    elif r[0] < 2*BS_MUTATION_FACTOR:
        individual[0] *= 2

    # Mutate learning rate
    if r[1] < LR_MUTATION_FACTOR:
        individual[1] /= 2
    elif r[1] < 2*LR_MUTATION_FACTOR:
        individual[1] *= 2

    # Mutate single layers
    for i in range(2, individual.size):
        if i % 3 == 2:      # Dense layer
            if r[i] < LSIZE_MUTATION_FACTOR:
                individual[i] = int(individual[i] / 2) + individual[i] % 1  # Preserve decimal part
            elif r[i] < 2 * LSIZE_MUTATION_FACTOR:
                individual[i] = int(individual[i] * 2) + individual[i] % 1  # Preserve decimal part
            elif r[i] < ACTIVATION_MUTATION_FACTOR + 2 * LSIZE_MUTATION_FACTOR:
                individual[i] = int(individual[i]) + random.choice(DENSE_ACTIVATIONS.keys())  # Mutate activation
        elif i % 3 == 0:    # Dropout layer
            if r[i] < DROPOUT_MUTATION_FACTOR:
                individual[i] *= 2
            elif r[i] < 2 * DROPOUT_MUTATION_FACTOR:
                individual[i] /= 2


def init_ga(eval_func, train_x, train_y, test_x, test_y):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.array(), fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual",          # Register function to initialize individual
                     init_individual()      # Function that initializes individual
                     )                      # (i.e. layers/hyperparameters in a network)

    toolbox.register("population",          # Register function to initialize population
                     tools.initRepeat,      # Repeat initialization of elements (in this case individuals)
                     np.array(),            # Base class of the population (list of individuals)
                     toolbox.individual     # Individual initializer
                     )                      # (i.e. neural network architecture)

    # Define evaluation function from function and data passed in arguments
    def evaluate(individual):
        return eval_func(individual, train_x, train_y, test_x, test_y)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=OVERALL_MUTATION_RATE)
    # We'll use the framework's selection function
    toolbox.register("select", tools.selBest, k=10)

    return toolbox


def run_ga(toolbox, pop_size=100):
    pop = toolbox.population(n=pop_size)
    # TODO: COMPLETE THIS FUNCTION
