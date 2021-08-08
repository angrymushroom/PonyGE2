from random import randint, random, sample, choice
from typing import List, Any, Tuple, Union

from algorithm.parameters import params
from representation import individual
from representation.latent_tree import latent_tree_crossover, latent_tree_repair
from utilities.representation.check_methods import check_ind
from sklearn.cluster import AgglomerativeClustering, OPTICS, DBSCAN, MeanShift, AffinityPropagation
from sklearn.preprocessing import MinMaxScaler
import math
from numpy import isnan


def cs(parents):
    """
    Specify a crossover function to run.

    :param parents: A population of parent individuals on which crossover is to
    be performed.
    :return: Call a function.
    """
    return params['CROSSOVER_SELECTION'](parents)


def crossover(parents):
    """
    Perform crossover on a population of individuals. The size of the crossover
    population is defined as params['GENERATION_SIZE'] rather than params[
    'POPULATION_SIZE']. This saves on wasted evaluations and prevents search
    from evaluating too many individuals.
    
    :param parents: A population of parent individuals on which crossover is to
    be performed.
    :return: A population of fully crossed over individuals.
    """

    # Initialise an empty population.
    cross_pop = []

    while len(cross_pop) < params['GENERATION_SIZE']:
        
        # Randomly choose two parents from the parent population.
        inds_in = sample(parents, 2)

        # Perform crossover on chosen parents.
        inds_out = crossover_inds(inds_in[0], inds_in[1])
        
        if inds_out is None:
            # Crossover failed.
            pass
        
        else:
            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop


def crossover_between_clusters(parents):
    """
    Perform crossover on a population of individuals. Cluster parents into k
    clusters then select two parents from two different clusters to do the crossover.

    :param parents: A population of parent individuals on which crossover is to
    be performed.
    :return: A population of fully crossed over individuals.
    """

    # Initialise an empty population.
    cross_pop = []
    # cluster the population into k clusters. set k in sub_population().
    # sub_list = sub_population(parents)  # cluster the population into k clusters. set k in sub_population
    sub_list = sub_population_genome_similarity(parents, params['CLUSTERS'])

    total_individuals = 0
    for i in sub_list:
        total_individuals += len(i)

    """print('NUMBER OF PARENTS:', len(parents))
    print('NUMBER OF INDIVIDUALS IN SUB-POPULATION:', total_individuals)

    print('----LENGTH OF SUB_LIST----', len(sub_list))
    for i in range(len(sub_list)):
        print('length of cluster %d is %d' % (i, len(sub_list[i])))"""

    # print('EXAMPLE:\n', sub_list[0])

    # Remove 20% of the worst individuals in each sub-population
    if params['REMOVE']:
        for i in sub_list:
            eliminate_bad_individual(i)

    while len(cross_pop) < params['GENERATION_SIZE']:
        # If there are more than 2 cluster, randomly select two clusters to do the crossover
        if len(sub_list) > 2:
            cluster_list = sample(sub_list, 2)  # randomly select two clusters
            # select one individual as a parent from each clusters
            ind_1, ind_2 = sample(cluster_list[0], 1), sample(cluster_list[1], 1)

        # If there is only one cluster left, it have to do the crossover within this cluster
        elif len(sub_list) == 1:
            ind_1, ind_2 = sample(sub_list[0], 1), sample(sub_list[0], 1)

        # If there are two clusters, only do the crossover between these two clusters
        elif len(sub_list) == 2:
            ind_1, ind_2 = sample(sub_list[0], 1), sample(sub_list[1], 1)

        # Perform crossover on chosen parents.
        inds_out = crossover_inds(ind_1[0], ind_2[0])

        if inds_out is None:
            # Crossover failed.
            pass
        else:
            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop


def crossover_in_same_cluster(parents):
    """
    Perform crossover on a population of individuals. Cluster parents into k
    clusters then select two parents from one cluster to do the crossover.

    :param parents: A population of parent individuals on which crossover is to
    be performed.
    :return: A population of fully crossed over individuals.
    """

    # Initialise an empty population.
    cross_pop = []
    # cluster the population into k clusters. set k in sub_population().
    sub_list = sub_population(parents)  # cluster the population into k clusters. set k in sub_population

    # Remove 20% of the worst individuals in each sub-population
    if params['REMOVE']:
        for i in sub_list:
            eliminate_bad_individual(i)

    while len(cross_pop) < params['GENERATION_SIZE']:

        cluster_list = sample(sub_list, 1)  # randomly select one clusters

        # Select another cluster if the cluster size less than 2
        while len(cluster_list[0]) < 2:
            cluster_list = sample(sub_list, 1)

        # select one individual as a parent from each clusters
        ind_1, ind_2 = sample(cluster_list[0], 2)

        # Perform crossover on chosen parents.
        inds_out = crossover_inds(ind_1, ind_2)

        if inds_out is None:
            # Crossover failed.
            pass
        else:
            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop


def crossover_one_elite_cluster(parents):
    """
    Perform crossover on a population of individuals.
    Cluster parents into k clusters then select two parents from two different clusters to do the crossover.

    Identify a elite cluster which has the highest mean fitness and always select one individual from it to do the
    crossover. Randomly select another individual from an another arbitrary clusters.

    :param parents: A population of parent individuals on which crossover is to be performed.
    :return: A population of fully crossed over individuals.
    """

    # Initialise an empty population.
    cross_pop = []

    # Cluster the population into k clusters. set k in sub_population().
    sub_list = sub_population(parents)  # a list contains sub populations lists

    # Remove 20% of the worst individuals in each sub-population
    if params['REMOVE']:
        for i in sub_list:
            eliminate_bad_individual(i)

    # Divide sub_list into a elite population and a list of average population
    elite_population, average_population = elite_cluster(sub_list)
    # print('\n----length of elite cluster: {}----\n'.format(len(elite_population)))
    while len(cross_pop) < params['GENERATION_SIZE']:
        # Biased randomized selection. select the better individual in elite_population
        # ind_1 = [brp_selection(elite_population)]

        ind_1 = sample(elite_population, 1)
        # Randomly select another individual from an arbitrary average population.
        selected_average_population = sample(average_population, 1)
        ind_2 = sample(selected_average_population[0], 1)

        # Perform crossover on chosen parents.
        inds_out = crossover_inds(ind_1[0], ind_2[0])

        if inds_out is None:
            # Crossover failed.
            pass
        else:
            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop


def crossover_inds(parent_0, parent_1):
    """
    Perform crossover on two selected individuals.
    
    :param parent_0: Parent 0 selected for crossover.
    :param parent_1: Parent 1 selected for crossover.
    :return: Two crossed-over individuals.
    """

    # Create copies of the original parents. This is necessary as the
    # original parents remain in the parent population and changes will
    # affect the originals unless they are cloned.
    ind_0 = parent_0.deep_copy()
    ind_1 = parent_1.deep_copy()

    # Crossover cannot be performed on invalid individuals.
    if not params['INVALID_SELECTION'] and (ind_0.invalid or ind_1.invalid):
        s = "operators.crossover.crossover\nError: invalid individuals " \
            "selected for crossover."
        raise Exception(s)

    # Perform crossover on ind_0 and ind_1.
    inds = params['CROSSOVER'](ind_0, ind_1)

    # Check each individual is ok (i.e. does not violate specified limits).
    checks = [check_ind(ind, "crossover") for ind in inds]

    if any(checks):
        # An individual violates a limit.
        return None

    else:
        # Crossover was successful, return crossed-over individuals.
        return inds


def variable_onepoint(p_0, p_1):
    """
    Given two individuals, create two children using one-point crossover and
    return them. A different point is selected on each genome for crossover
    to occur. Note that this allows for genomes to grow or shrink in
    size. Crossover points are selected within the used portion of the
    genome by default (i.e. crossover does not occur in the tail of the
    individual).
    
    :param p_0: Parent 0
    :param p_1: Parent 1
    :return: A list of crossed-over individuals.
    """

    # Get the chromosomes.
    genome_0, genome_1 = p_0.genome, p_1.genome

    # Uniformly generate crossover points.
    max_p_0, max_p_1 = get_max_genome_index(p_0, p_1)
        
    # Select unique points on each genome for crossover to occur.
    pt_0, pt_1 = randint(1, max_p_0), randint(1, max_p_1)

    # Make new chromosomes by crossover: these slices perform copies.
    if random() < params['CROSSOVER_PROBABILITY']:
        c_0 = genome_0[:pt_0] + genome_1[pt_1:]
        c_1 = genome_1[:pt_1] + genome_0[pt_0:]
    else:
        c_0, c_1 = genome_0[:], genome_1[:]

    # Put the new chromosomes into new individuals.
    ind_0 = individual.Individual(c_0, None)
    ind_1 = individual.Individual(c_1, None)

    """print('----PRINT OFFSPRINGS----')
    print(ind_0, ind_1)
    print('----OFFSPRINGS GENOME----')
    print(ind_0.genome, ind_1.genome)"""

    return [ind_0, ind_1]


def fixed_onepoint(p_0, p_1):
    """
    Given two individuals, create two children using one-point crossover and
    return them. The same point is selected on both genomes for crossover
    to occur. Crossover points are selected within the used portion of the
    genome by default (i.e. crossover does not occur in the tail of the
    individual).

    :param p_0: Parent 0
    :param p_1: Parent 1
    :return: A list of crossed-over individuals.
    """

    # Get the chromosomes.
    genome_0, genome_1 = p_0.genome, p_1.genome

    # Uniformly generate crossover points.
    max_p_0, max_p_1 = get_max_genome_index(p_0, p_1)
    
    # Select the same point on both genomes for crossover to occur.
    pt = randint(1, min(max_p_0, max_p_1))
    
    # Make new chromosomes by crossover: these slices perform copies.
    if random() < params['CROSSOVER_PROBABILITY']:
        c_0 = genome_0[:pt] + genome_1[pt:]
        c_1 = genome_1[:pt] + genome_0[pt:]
    else:
        c_0, c_1 = genome_0[:], genome_1[:]
    
    # Put the new chromosomes into new individuals.
    ind_0 = individual.Individual(c_0, None)
    ind_1 = individual.Individual(c_1, None)

    return [ind_0, ind_1]


def fixed_twopoint(p_0, p_1):
    """
    Given two individuals, create two children using two-point crossover and
    return them. The same points are selected on both genomes for crossover
    to occur. Crossover points are selected within the used portion of the
    genome by default (i.e. crossover does not occur in the tail of the
    individual).

    :param p_0: Parent 0
    :param p_1: Parent 1
    :return: A list of crossed-over individuals.
    """

    genome_0, genome_1 = p_0.genome, p_1.genome

    # Uniformly generate crossover points.
    max_p_0, max_p_1 = get_max_genome_index(p_0, p_1)

    # Select the same points on both genomes for crossover to occur.
    a, b = randint(1, max_p_0), randint(1, max_p_1)
    pt_0, pt_1 = min([a, b]), max([a, b])
    
    # Make new chromosomes by crossover: these slices perform copies.
    if random() < params['CROSSOVER_PROBABILITY']:
        c_0 = genome_0[:pt_0] + genome_1[pt_0:pt_1] + genome_0[pt_1:]
        c_1 = genome_1[:pt_0] + genome_0[pt_0:pt_1] + genome_1[pt_1:]
    else:
        c_0, c_1 = genome_0[:], genome_1[:]

    # Put the new chromosomes into new individuals.
    ind_0 = individual.Individual(c_0, None)
    ind_1 = individual.Individual(c_1, None)
    
    return [ind_0, ind_1]


def variable_twopoint(p_0, p_1):
    """
    Given two individuals, create two children using two-point crossover and
    return them. Different points are selected on both genomes for crossover
    to occur. Note that this allows for genomes to grow or shrink in size.
    Crossover points are selected within the used portion of the genome by
    default (i.e. crossover does not occur in the tail of the individual).

    :param p_0: Parent 0
    :param p_1: Parent 1
    :return: A list of crossed-over individuals.
    """
    
    genome_0, genome_1 = p_0.genome, p_1.genome
    
    # Uniformly generate crossover points.
    max_p_0, max_p_1 = get_max_genome_index(p_0, p_1)
    
    # Select the same points on both genomes for crossover to occur.
    a_0, b_0 = randint(1, max_p_0), randint(1, max_p_1)
    a_1, b_1 = randint(1, max_p_0), randint(1, max_p_1)
    pt_0, pt_1 = min([a_0, b_0]), max([a_0, b_0])
    pt_2, pt_3 = min([a_1, b_1]), max([a_1, b_1])
    
    # Make new chromosomes by crossover: these slices perform copies.
    if random() < params['CROSSOVER_PROBABILITY']:
        c_0 = genome_0[:pt_0] + genome_1[pt_2:pt_3] + genome_0[pt_1:]
        c_1 = genome_1[:pt_2] + genome_0[pt_0:pt_1] + genome_1[pt_3:]
    else:
        c_0, c_1 = genome_0[:], genome_1[:]
    
    # Put the new chromosomes into new individuals.
    ind_0 = individual.Individual(c_0, None)
    ind_1 = individual.Individual(c_1, None)
    
    return [ind_0, ind_1]


def subtree(p_0, p_1):
    """
    Given two individuals, create two children using subtree crossover and
    return them. Candidate subtrees are selected based on matching
    non-terminal nodes rather than matching terminal nodes.
    
    :param p_0: Parent 0.
    :param p_1: Parent 1.
    :return: A list of crossed-over individuals.
    """

    def do_crossover(tree0, tree1, shared_nodes):
        """
        Given two instances of the representation.tree.Tree class (
        derivation trees of individuals) and a list of intersecting
        non-terminal nodes across both trees, performs subtree crossover on
        these trees.
        
        :param tree0: The derivation tree of individual 0.
        :param tree1: The derivation tree of individual 1.
        :param shared_nodes: The sorted list of all non-terminal nodes that are
        in both derivation trees.
        :return: The new derivation trees after subtree crossover has been
        performed.
        """
        
        # Randomly choose a non-terminal from the set of permissible
        # intersecting non-terminals.
        crossover_choice = choice(shared_nodes)
    
        # Find all nodes in both trees that match the chosen crossover node.
        nodes_0 = tree0.get_target_nodes([], target=[crossover_choice])
        nodes_1 = tree1.get_target_nodes([], target=[crossover_choice])

        # Randomly pick a node.
        t0, t1 = choice(nodes_0), choice(nodes_1)

        # Check the parents of both chosen subtrees.
        p0 = t0.parent
        p1 = t1.parent
    
        if not p0 and not p1:
            # Crossover is between the entire tree of both tree0 and tree1.
            
            return t1, t0
        
        elif not p0:
            # Only t0 is the entire of tree0.
            tree0 = t1

            # Swap over the subtrees between parents.
            i1 = p1.children.index(t1)
            p1.children[i1] = t0

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t1 is now a whole
            # individual, it has no parent.
            t0.parent = p1
            t1.parent = None
    
        elif not p1:
            # Only t1 is the entire of tree1.
            tree1 = t0

            # Swap over the subtrees between parents.
            i0 = p0.children.index(t0)
            p0.children[i0] = t1

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t0 is now a whole
            # individual, it has no parent.
            t1.parent = p0
            t0.parent = None
    
        else:
            # The crossover node for both trees is not the entire tree.
       
            # For the parent nodes of the original subtrees, get the indexes
            # of the original subtrees.
            i0 = p0.children.index(t0)
            i1 = p1.children.index(t1)
        
            # Swap over the subtrees between parents.
            p0.children[i0] = t1
            p1.children[i1] = t0
        
            # Set the parents of the crossed-over subtrees as their new
            # parents.
            t1.parent = p0
            t0.parent = p1
        
        return tree0, tree1

    def intersect(l0, l1):
        """
        Returns the intersection of two sets of labels of nodes of
        derivation trees. Only returns matching non-terminal nodes across
        both derivation trees.
        
        :param l0: The labels of all nodes of tree 0.
        :param l1: The labels of all nodes of tree 1.
        :return: The sorted list of all non-terminal nodes that are in both
        derivation trees.
        """
        
        # Find all intersecting elements of both sets l0 and l1.
        shared_nodes = l0.intersection(l1)
        
        # Find only the non-terminals present in the intersecting set of
        # labels.
        shared_nodes = [i for i in shared_nodes if i in params[
            'BNF_GRAMMAR'].non_terminals]
        
        return sorted(shared_nodes)

    if random() > params['CROSSOVER_PROBABILITY']:
        # Crossover is not to be performed, return entire individuals.
        ind0 = p_1
        ind1 = p_0
    
    else:
        # Crossover is to be performed.
    
        if p_0.invalid:
            # The individual is invalid.
            tail_0 = []
            
        else:
            # Save tail of each genome.
            tail_0 = p_0.genome[p_0.used_codons:]

        if p_1.invalid:
            # The individual is invalid.
            tail_1 = []

        else:
            # Save tail of each genome.
            tail_1 = p_1.genome[p_1.used_codons:]
        
        # Get the set of labels of non terminals for each tree.
        labels1 = p_0.tree.get_node_labels(set())
        labels2 = p_1.tree.get_node_labels(set())

        # Find overlapping non-terminals across both trees.
        shared_nodes = intersect(labels1, labels2)

        if len(shared_nodes) != 0:
            # There are overlapping NTs, cross over parts of trees.
            ret_tree0, ret_tree1 = do_crossover(p_0.tree, p_1.tree,
                                                shared_nodes)
        
        else:
            # There are no overlapping NTs, cross over entire trees.
            ret_tree0, ret_tree1 = p_1.tree, p_0.tree
        
        # Initialise new individuals using the new trees.
        ind0 = individual.Individual(None, ret_tree0)
        ind1 = individual.Individual(None, ret_tree1)

        # Preserve tails.
        ind0.genome = ind0.genome + tail_0
        ind1.genome = ind1.genome + tail_1

    return [ind0, ind1]


def get_max_genome_index(ind_0, ind_1):
    """
    Given two individuals, return the maximum index on each genome across
    which operations are to be performed. This can be either the used
    portion of the genome or the entire length of the genome.
    
    :param ind_0: Individual 0.
    :param ind_1: Individual 1.
    :return: The maximum index on each genome across which operations are to be
             performed.
    """

    if params['WITHIN_USED']:
        # Get used codons range.
        
        if ind_0.invalid:
            # ind_0 is invalid. Default to entire genome.
            max_p_0 = len(ind_0.genome)
        
        else:
            max_p_0 = ind_0.used_codons
    
        if ind_1.invalid:
            # ind_1 is invalid. Default to entire genome.
            max_p_1 = len(ind_1.genome)
        
        else:
            max_p_1 = ind_1.used_codons
    
    else:
        # Get length of entire genome.
        max_p_0, max_p_1 = len(ind_0.genome), len(ind_1.genome)
        
    return max_p_0, max_p_1


def LTGE_crossover(p_0, p_1):
    """Crossover in the LTGE representation."""

    # crossover and repair.
    # the LTGE crossover produces one child, and is symmetric (ie
    # xover(p0, p1) is not different from xover(p1, p0)), but since it's
    # stochastic we can just run it twice to get two individuals
    # expected to be different.
    g_0, ph_0 = latent_tree_repair(
        latent_tree_crossover(p_0.genome, p_1.genome),
        params['BNF_GRAMMAR'], params['MAX_TREE_DEPTH'])
    g_1, ph_1 = latent_tree_repair(
        latent_tree_crossover(p_0.genome, p_1.genome),
        params['BNF_GRAMMAR'], params['MAX_TREE_DEPTH'])

    # wrap up in Individuals and fix up various Individual attributes
    ind_0 = individual.Individual(g_0, None, False)
    ind_1 = individual.Individual(g_1, None, False)

    ind_0.phenotype = ph_0
    ind_1.phenotype = ph_1

    # number of nodes is the number of decisions in the genome
    ind_0.nodes = ind_0.used_codons = len(g_0)
    ind_1.nodes = ind_1.used_codons = len(g_1)

    # each key is the length of a path from root
    ind_0.depth = max(len(k) for k in g_0)
    ind_1.depth = max(len(k) for k in g_1)
    
    # in LTGE there are no invalid individuals
    ind_0.invalid = False
    ind_1.invalid = False
   
    return [ind_0, ind_1]


def sub_population(population):
    """
    Divide the population into k clusters.
    """
    # df = [[i.fitness, len(i.genome), i.nodes, i.depth, i.used_codons] for i in population]
    df = [[len(i.genome), i.nodes, i.depth, i.used_codons] for i in population]

    # Normalization
    scalar = MinMaxScaler()
    df = scalar.fit_transform(df)

    model = AgglomerativeClustering(n_clusters=params['CLUSTERS'], affinity='euclidean', linkage='ward')

    y_hat = model.fit_predict(df)  # predict the cluster label

    # A list of tuples. Each tuple contains an individual and the corresponding cluster label
    individuals_with_labels = zip(population, y_hat)
    individuals_with_labels = list(individuals_with_labels)

    # calculate the number of clusters
    y_hat_set = set(y_hat)
    cluster_number = len(y_hat_set)

    sub_list = [[] for i in range(cluster_number)]

    # put the individual into proper sub-population in sub_list
    for i in individuals_with_labels:
        cluster_index = i[1]  # cluster
        sub_list[cluster_index].append(i[0])

    return sub_list


def sub_population_genome_similarity(population, n_clusters):
    """
    Cluster the individuals by different gene.

    Randomly select n individuals as the centroids of n clusters. Then put the rest  individuals into the nearest
    cluster. The distance between a individual and a cluster is defined as the number of the different genes between
    the individual and the centroid individual of a cluster.

    :param population: parents.
    :param n_clusters: hyper-parameter. the number of clusters.
    :return: a list contains n clusters of individuals.
    """
    centroids = sample(population, n_clusters)
    sub_list = [[] for i in range(n_clusters)]

    # put centroids into the sub_list
    for i in range(len(centroids)):
        sub_list[i].append(centroids[i])

    for individual in population:
        score_list = []
        for centroid in centroids:
            score = genome_similarity(centroid, individual)
            score_list.append(score)
        index_max = max(range(len(score_list)), key=score_list.__getitem__)
        sub_list[index_max].append(individual)

    # remove the sub-population if the population is too small
    for i in sub_list[:]:
        if len(i) < 5:
            sub_list.remove(i)

    return sub_list


def genome_similarity(centroid, individual):
    """
    Calculate the genome distance between two individuals. The distance is defined as the number of different
    genes of two individuals
    """
    score = set(centroid.genome).intersection(individual.genome)
    return score


def eliminate_bad_individual(population, eliminate_rate=.2):
    """
    Eliminate the eliminate_rate percent of bad individuals in a population.
    """
    population.sort(reverse=True)
    population_size = len(population)
    n_eliminate = round(population_size * eliminate_rate)

    for i in range(n_eliminate):
        population.pop(-1)


def elite_cluster(sub_list):
    """
    Divide a list of sub-population into an elite population and some average populations.

    :param sub_list: a list of population clusters
    :return: elite_population is a list of individuals. average_population is a list of lists
    """
    # print('length of original sub_list: {}'.format(len(sub_list)))

    elite_population = None
    average_population = None

    # Initialise a list to contain the fitness mean.
    fitness_mean_list = [0 for i in range(len(sub_list))]  # initialise a list to contain the fitness mean.

    # Calculate the mean of fitness for each sub-population.
    for i in range(len(sub_list)):
        fitness_mean_list[i] = 0
        for j in sub_list[i]:
            if not isnan(j.fitness):
                fitness_mean_list[i] += j.fitness
            elif isnan(j.fitness):
                fitness_mean_list[i] += 0
        fitness_mean_list[i] /= len(sub_list[i])

    # print('----fitness_mean_list-----\n', fitness_mean_list)

    # Select the population with the lowest fitness mean as elite population.
    # The left populations are the average population.
    ffs = params['FITNESS_FUNCTION']  # Get the fitness function class
    # When the fitness function is maximising, select the cluster with the highest mean fitness as the elite cluster
    if ffs.maximise:
        min_max_value = max(fitness_mean_list)
    # When the fitness function is minimising, select the cluster with the lowest mean fitness as the elite cluster
    else:
        min_max_value = min(fitness_mean_list)

    # Set the best sub-population to be the elite population and the others to be the average populations.
    for i in range(len(fitness_mean_list)):
        if fitness_mean_list[i] == min_max_value:
            elite_population = sub_list.pop(i)
            average_population = sub_list
            break

    # print('-----ELITE CLUSTER FITNESS-----\n', min_max_value)

    return elite_population, average_population


def brp_selection(elite_population):
    """
    Use biased randomized algorithm to select one parent from elite population.
    Biased randomized algorithm: sort the individuals by their fitness and assign
    them exponential skewed probabilities to be selected.
    """
    elite_population.sort(reverse=True)
    beta = params['BETA']
    index = int(math.log(random()) / math.log(1 - beta))
    index = index % len(elite_population)
    ind_1 = elite_population[index]

    return ind_1


# Set attributes for all operators to define linear or subtree representations.
variable_onepoint.representation = "linear"
fixed_onepoint.representation = "linear"
variable_twopoint.representation = "linear"
fixed_twopoint.representation = "linear"
subtree.representation = "subtree"
LTGE_crossover.representation = "latent tree"
