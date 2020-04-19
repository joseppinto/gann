import random
from deap import base
from deap import creator
from deap import tools

# Possible batch sizes in first generation of nn architectures
INITIAL_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# Possible learning rates in first generation of nn architectures
INITIAL_LEARNING_RATES = [0.1, 0.01, 0.01]
# Possible initial dense layer sizes
INITIAL_DENSE_SIZE = [16, 32, 64]
# Possible initial dense layer activations
# 't' = 'tanh', 'r' = 'relu', 'l' = 'linear', 's' = 'sigmoid'
INITIAL_DENSE_ACTIVATIONS = ['t', 'r', 'l', 's']
# Possible initial dropout rates
INITIAL_DROPOUT_RATES = [0, 0.05, 0.1]


# Function to initialize individual (used in the first generation)
def init_individual():
    ind = ""
    # Add batch size
    ind += f"{random.choice(INITIAL_BATCH_SIZES)},"
    # Add learning rate
    ind += f"{random.choice(INITIAL_LEARNING_RATES)}"

    # Random number of initial layer groups (dense + dropout + normalization)
    # Initial structure for the networks
    n_layer_groups = random.randint(1, 5)
    for i in range(n_layer_groups):
        genes = ","
        # Add a dense layer
        genes += f"D{random.choice(INITIAL_DENSE_ACTIVATIONS)}{random.choice(INITIAL_DENSE_SIZE)},"
        # Add a dropout layer
        genes += f"d0{random.choice(INITIAL_DROPOUT_RATES)},"
        # Add a normalization layer
        genes += "N00"

        # Add block of layers to individual's dna
        ind += genes

    return ind


def run_ga(eval_func, train_x, train_y, test_x, test_y):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual",  # Register function to initialize individual
                     init_individual()  # Function that initializes individual
                     )  # (i.e. layers/hyperparameters in a network)

    toolbox.register("population",  # Register function to initialize population
                     tools.initRepeat,  # Function that repeats initialization of elements (in this case individuals)
                     list,  # Base class of the population (list of individuals)
                     toolbox.individual  # Individual initializer
                     )  # (i.e. neural network architecture)

    # Define evaluation function from function and data passed in arguments
    def evaluate(individual):
        return eval_func(individual, train_x, train_y, test_x, test_y)

    toolbox.register("evaluate", evaluate)
    # TODO :
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("select", tools.selTournament, tournsize=3)
