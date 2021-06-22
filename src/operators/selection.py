from random import sample, choices, random
from operator import attrgetter
from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics, \
    crowded_comparison_operator
from utilities.stats import trackers
import math


def selection(population):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.

    :param population: input population
    :return: selected population
    """

    return params['SELECTION'](population)


def tournament(population):
    """
    Given an entire population, draw <tournament_size> competitors randomly and
    return the best. Only valid individuals can be selected for tournaments.

    :param population: A population from which to select individuals.
    :return: A population of the winners from tournaments.
    """

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    print('Generation_size is: ', params['GENERATION_SIZE'])

    while len(winners) < params['GENERATION_SIZE']:
        # Randomly choose TOURNAMENT_SIZE competitors from the given
        # population. Allows for re-sampling of individuals.

        competitors = sample(available, params['TOURNAMENT_SIZE'])
        #  print("Winners length is: %d\n" %(len(winners)))
        #  print("available length is: %d\n" % (len(available)))

        """print('Two competitors are found\n')
        print('Print elements')
        print(*competitors)
        print('\n--------Print competitors:---------')
        for i, j in enumerate(competitors):
            print('Competitor ', i, ' is ', j)
            print(j.genome)
        print('--------Competitors are printed--------\n')"""

        # Return the single best competitor.
        winners.append(max(competitors))

    # Return the population of tournament winners.
    return winners


def truncation(population):
    """
    Given an entire population, return the best <proportion> of them.

    :param population: A population from which to select individuals.
    :return: The best <proportion> of the given population.
    """

    # Sort the original population.
    population.sort(reverse=True)

    # Find the cutoff point for truncation.
    cutoff = int(len(population) * float(params['SELECTION_PROPORTION']))

    # Return the best <proportion> of the given population.
    return population[:cutoff]


def nsga2_selection(population):
    """Apply NSGA-II selection operator on the *population*. Usually, the
    size of *population* will be larger than *k* because any individual
    present in *population* will appear in the returned list at most once.
    Having the size of *population* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *population*. For more
    details on the NSGA-II operator see [Deb2002]_.
    
    :param population: A population from which to select individuals.
    :returns: A list of selected individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """

    selection_size = params['GENERATION_SIZE']
    tournament_size = params['TOURNAMENT_SIZE']

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    # Compute pareto front metrics.
    pareto = compute_pareto_metrics(available)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(available, pareto, tournament_size))

    return winners


def pareto_tournament(population, pareto, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the
    individual and the crowding distance.

    :param population: A population from which to select individuals.
    :param pareto: The pareto front information.
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """
    
    # Initialise no best solution.
    best = None
    
    # Randomly sample *tournament_size* participants.
    participants = sample(population, tournament_size)
    
    for participant in participants:
        if best is None or crowded_comparison_operator(participant, best,
                                                       pareto):
            best = participant
    
    return best


# Set attributes for all operators to define multi-objective operators.
nsga2_selection.multi_objective = True


def two_max_genome_length(available_origin, N):
    final_list = []
    print('Print available: ', *available_origin)
    available = available_origin.copy()

    for i in range(0, N):
        # max1 = 0
        longest_length_individual = [0,0] # (length, individual)

        for j in range(len(available)):
            if len(available[j].genome) > longest_length_individual[0]:
                longest_length_individual[0] = len(available[j].genome)
                longest_length_individual[1] = available[j]

        available.remove(longest_length_individual[1])
        final_list.append(longest_length_individual[1])

    return final_list


def genome_tournament(population):
    """
    Genome length and tournament selection is a variant of tournament.
    This selection method select the winner with best fitness or longest
    genome length. The probability of the choice  is determined by the
    fitness gap of last two generations. It's more possible to use longest
    genome length if there is a high positive fitness increase because we
    want to enhance the diversity of the populations at the beginning stage.
    At the middle stage, the fitness gap is decreasing, it would be more
    likely to choose tournament selection because a high survival pressure
    is needed.

    :param population: A population from which to select individuals.
    :return: A population of the winners from tournaments.
    """

    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    max_fitness_gap = 0
    normalization = {'min_value': 0,
                     'min_max_range': max_fitness_gap}

    while len(winners) < params['GENERATION_SIZE']:
        # Randomly choose TOURNAMENT_SIZE competitors from the given
        # population. Allows for re-sampling of individuals.
        competitors = sample(available, params['TOURNAMENT_SIZE'])
        fitness_or_genome_length =[max, max_genome_length]

        if (max(trackers.best_fitness_list)-min(trackers.best_fitness_list)) != 0:

            fitness_gap = trackers.best_fitness_list[-2] - trackers.best_fitness_list[-1]
            if fitness_gap > max_fitness_gap:
                max_fitness_gap = fitness_gap

            p = 0.5 + 1 / (1 + exp(-4*(fitness_gap - normalization['min_value'])/(normalization['min_max_range'])))
        else:
            p = 0.5

        print('P is: ', p)

        weight = [p, 1-p]
        method = choices(fitness_or_genome_length, k=1, weights=weight)
        f = method[0]

        winners.append(f(competitors))

    # Return the population of tournament winners.
    return winners


def max_genome_length(available):

    return max(available, key=attrgetter('genome_length'))


def brp_exponential(population):
    """
    Biased randomized selection.
    Assign the probabilities to individuals in a population. The probability is exp(-r) where r is individuals'
    descending order by fitness. This algorithm intends to select the individuals with higher fitness but the worse
    individuals still have chances to be selected.

    :param population: A population from which to select individuals.
    :return: A population of the winners.
    """
    # Initialise list of winners.
    winners = []
    population.sort(reverse=True)  # For assigning probabilities to sorted individuals.
    beta = 0.2  # This parameter is for tuning the stochastic factor. When it is closing to 0, it is more likely to
    #  a better individual. when it is closing to 1, it will be a pure random selecting process.

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    # Before  reaching the generation size, select winner from population and add it into winners list.
    while len(winners) < params['GENERATION_SIZE']:
        #  print("Available size in brp tournament is: %d \n" % (len(available)))
        #  print("winners size in brp tournament is: %d \n" % (len(winners)))
        index = int(math.log(random()) / math.log(1 - beta))
        index = index % len(available)
        winners.append(available[index])

    # Return the population of tournament winners.
    return winners


def brp_linear(population):
    """
    Biased randomized selection.
    Assign the probabilities to individuals in a population. The probability is 1/r where r is individuals'
    descending order by fitness. This algorithm intends to select the individuals with higher fitness but the worse
    individuals still have chances to be selected.

    :param population: A population from which to select individuals.
    :return: A population of the winners.
    """

    # Initialise list of winners.
    print("-----CALLING LINEAR-----")
    winners = []
    population.sort(reverse=True)

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    while len(winners) < params['GENERATION_SIZE']:
        #  print("Available size in brp tournament is: %d \n" % (len(available)))
        #  print("winners size in brp tournament is: %d \n" % (len(winners)))
        index = int(len(available) * (1 - math.sqrt(random())))
        winners.append(available[index])

    # Return the population of tournament winners.
    return winners
