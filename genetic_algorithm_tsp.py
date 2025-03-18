import random as rd
import math
import numpy as np
from collections import OrderedDict

class GeneticAlgorithmTSP:

    def __init__(self, graph, city_names, generations=20, population_size=10, tournament_size=4, mutationRate=0.1, fitness_selection_rate=0.1):
        """
        Initialize the GeneticAlgorithmTSP object.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - city_names: List of city names.
        - generations: Number of generations to run the algorithm.
        - population_size: Size of the population of routes.
        - tournament_size: Number of routes to select for crossover.
        - mutationRate: Probability of mutation for a genome.
        - fitness_selection_rate: Percentage of fittest routes to carry over to the next generation.
        """
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutationRate = mutationRate
        self.fitness_selection_rate = fitness_selection_rate

        # storing genetic diversity values
        self.genetic_diversity_values = []

        # Mapping between city names and characters
        self.city_map = OrderedDict((char, city) for char, city in zip(range(32, 54), graph.vertices()))
        self.city_mapping = {char: city for char, city in zip(range(32, 54), city_names)}
        # I don't know why this happens
        #self.city_map[32], self.city_map[33] = self.city_map[33], self.city_map[32]

    def calculate_genetic_diversity(self, population):
        """
        Calculate genetic diversity within the population.

        Parameters:
        - population: List of routes in the current population.

        Returns:
        - Genetic diversity as a float value.
        """
        # Convert routes to a matrix for easier distance calculations
        routes_matrix = np.array([list(route) for route in population])

        # Calculate pairwise genetic distances between routes
        pairwise_distances = np.sum(routes_matrix[:, None, :] != routes_matrix[None, :, :], axis=2)

        # Calculate average genetic distance
        total_distance = np.sum(pairwise_distances)
        num_pairs = len(population) * (len(population) - 1)
        average_distance = total_distance / num_pairs

        # Genetic diversity is the inverse of average distance
        genetic_diversity = 1 / (1 + average_distance)

        return genetic_diversity

    def get_genetic_diversity_values(self):
        """
        Get the genetic diversity values for each generation.

        Returns:
        - List of genetic diversity values.
        """
        return self.genetic_diversity_values

    def minCostIndex(self, costs):
        """
        Return the index of the minimum cost in the costs list.
        """
        return min(range(len(costs)), key=costs.__getitem__)

    def find_fittest_path(self, graph):
        """
        Find the fittest path through the TSP graph using genetic algorithm.

        Parameters:
        - graph: The Graph object representing the TSP graph.

        Returns:
        - fittest_route: List of cities representing the fittest route.
        - fittest_fitness: The fitness (minimum cost) of the fittest route.
        """
        population = self.randomizeCities(graph.vertices())
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fitness_selection_rate)

        if number_of_fits_to_carryover > self.population_size:
            raise ValueError('Fitness rate must be in [0, 1].')

        print('Optimizing TSP Route for Graph:')

        for generation in range(1, self.generations + 1):
            #print('\nGeneration: {0}'.format(generation))
            # this print is wrong because it just gets them ordered, now how they are originally ordered in population
            #print('Population: {0}'.format([self.city_mapping.get(city_map_key, city_map_key) for city_map_key in self.city_map.keys()]))
            #print('Population: {0}'.format([self.city_mapping.get(char, char) for char in population]))

            new_population = self.create_next_generation(graph, population, number_of_fits_to_carryover)
            population = new_population

            fittest_index, fittest_route, fittest_fitness = self.get_fittest_route(graph, population)
            fittest_route = [list(OrderedDict(self.city_mapping).values())[
                                 list(OrderedDict(self.city_map).values()).index(char)] if char in list(
                OrderedDict(self.city_map).values()) else char for char in fittest_route]
            #fittest_route = [self.city_mapping.get(char, char) for char in fittest_route]

            #print('Fittest Route: {0}\nFitness(minimum cost): {1}'.format(fittest_route, fittest_fitness))

            genetic_diversity = self.calculate_genetic_diversity(population)
            self.genetic_diversity_values.append(round(genetic_diversity, 4))
            #print('Genetic Diversity: {:.4f}'.format(genetic_diversity))

            if self.converged(population):
                print("Converged", population)
                print('\nConverged to a local minima.')
                break

        return fittest_route, fittest_fitness

    def create_next_generation(self, graph, population, number_of_fits_to_carryover):
        """
        Create the next generation of routes based on the current population.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes in the current population.
        - number_of_fits_to_carryover: Number of fittest routes to carry over to the next generation.

        Returns:
        - new_population: List of routes in the new generation.
        """
        new_population = self.add_fittest_routes(graph, population, number_of_fits_to_carryover)
        new_population += [self.mutate(self.crossover(*self.select_parents(graph, population))) for _ in
                           range(self.population_size - number_of_fits_to_carryover)]
        return new_population

    def add_fittest_routes(self, graph, population, number_of_fits_to_carryover):
        """
        Add the fittest routes to the next generation.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes in the current population.
        - number_of_fits_to_carryover: Number of fittest routes to carry over to the next generation.

        Returns:
        - sorted_population: Sorted list of routes based on fitness.
        """
        sorted_population = [x for _, x in sorted(zip(self.computeFitness(graph, population), population))]
        return sorted_population[:number_of_fits_to_carryover]

    def get_fittest_route(self, graph, population):
        """
        Get the fittest route from the population.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes in the current population.

        Returns:
        - fittest_index: Index of the fittest route in the population.
        - fittest_route: List of cities representing the fittest route.
        - fittest_fitness: The fitness (minimum cost) of the fittest route.
        """
        fitness = self.computeFitness(graph, population)
        fittest_index = self.minCostIndex(fitness)
        return fittest_index, population[fittest_index], fitness[fittest_index]

    def select_parents(self, graph, population):
        """
        Select two parents for crossover using tournament selection.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes in the current population.

        Returns:
        - parent1: First parent selected for crossover.
        - parent2: Second parent selected for crossover.
        """
        return self.tournamentSelection(graph, population), self.tournamentSelection(graph, population)

    def randomizeCities(self, graph_nodes):
        """
        Randomly generate routes for the initial population.

        Parameters:
        - graph_nodes: List of nodes in the TSP graph.

        Returns:
        - List of random routes.
        """
        # Nodes without the start city
        nodes = [node for node in graph_nodes if node != self.graph.start_city]

        return [
            self.graph.start_city + ''.join(rd.sample(nodes, len(nodes))) + self.graph.start_city
            for _ in range(self.population_size)
        ]

    def computeFitness(self, graph, population):
        """
        Compute the fitness (cost) for each route in the population.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes.

        Returns:
        - List of fitness values (costs).
        """
        return [graph.getPathCost(path) for path in population]

    def tournamentSelection(self, graph, population):
        """
        Perform tournament selection to choose a parent for crossover.

        Parameters:
        - graph: The Graph object representing the TSP graph.
        - population: List of routes in the current population.

        Returns:
        - Selected parent for crossover.
        """
        tournament_contestants = rd.choices(population, k=self.tournament_size)
        return min(tournament_contestants, key=lambda path: graph.getPathCost(path))

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate an offspring.

        Parameters:
        - parent1: First parent for crossover.
        - parent2: Second parent for crossover.

        Returns:
        - Offspring generated through crossover.
        """
        offspring_length = len(parent1) - 2  # excluding first and last city

        offspring = ['' for _ in range(offspring_length)]

        index_low, index_high = self.computeTwoPointIndexes(parent1)

        # Copy the genes from parent1 to the offspring
        offspring[index_low: index_high + 1] = list(parent1)[index_low: index_high + 1]

        # The remaining genes are copied from parent2
        empty_place_indexes = [i for i in range(offspring_length) if offspring[i] == '']
        for i in parent2[1: -1]:  # exclude the start and end cities
            if '' not in offspring or not empty_place_indexes:
                break
            if i not in offspring:
                offspring[empty_place_indexes.pop(0)] = i

        offspring = [self.graph.start_city] + offspring + [self.graph.start_city]
        return ''.join(offspring)


    def mutate(self, genome):
        """
        Mutate a genome with a probability specified by mutationRate.

        Parameters:
        - genome: The genome to be mutated.

        Returns:
        - Mutated genome.
        """
        if rd.random() < self.mutationRate:
            index_low, index_high = self.computeTwoPointIndexes(genome)
            return self.swap(index_low, index_high, genome)
        else:
            return genome

    def computeTwoPointIndexes(self,parent):
        """
        Compute two crossover points for the two-point crossover.

        Parameters:
        - parent: Parent for which crossover points are computed.

        Returns:
        - Tuple containing two crossover points (index_low, index_high).
        """
        index_low = rd.randint(1, len(parent) - 3)
        index_high = rd.randint(index_low+1, len(parent) - 2)

        # make sure the difference between the two indexes is less than half the length of the parent
        if index_high - index_low > math.ceil(len(parent) // 2):
              return self.computeTwoPointIndexes(parent)
        else:
              return index_low, index_high

    def swap(self, index_low, index_high, string):
        """
        Swap two elements in a string.

        Parameters:
        - index_low: Index of the first element to be swapped.
        - index_high: Index of the second element to be swapped.
        - string: The string in which elements are swapped.

        Returns:
        - String with elements swapped.
        """
        string = list(string)
        string[index_low], string[index_high] = string[index_high], string[index_low]
        return ''.join(string)

    def converged(self, population):
        """
        Check if all genomes in the population are the same.

        Parameters:
        - population: List of genomes.

        Returns:
        - True if all genomes are the same, False otherwise.
        """
        return all(genome == population[0] for genome in population)