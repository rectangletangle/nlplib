''' A prototype for selection based genetic algorithms. '''


import random
import itertools

from nlplib.core.base import Base

class Individual (Base) :
    def __init__ (self, name) :
        self.name = name

        self.traits = {}

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.name, self.traits, *args, **kw)

def reproduce (selection, *individuals) :
    descendant = selection.individual()

    first, *others = individuals

    for trait_name in first.traits.keys() :
        descendant.traits[trait_name] = random.choice(individuals).traits[trait_name]

    return descendant

class Selection (Base) :
    def __init__ (self, name, fitness, mutate, reproducing_percentage=0.25, mutation_percentage=0.1) :
        self.name = name

        self.fitness = fitness
        self.mutate = mutate

        self.reproducing_percentage = reproducing_percentage
        self.mutation_percentage = mutation_percentage

        self.population = set()

        self._counter = itertools.count()

    def _individual_name (self) :
        return (self.name, next(self._counter))

    def individual (self) :
        individual = Individual(self._individual_name())
        self.population.add(individual)
        return individual

    def _sample_size (self, population, percentage) :
        return int(len(population) * percentage)

    def _reproduce (self) :
        total_fittness = []

        def fitness (individual) :
            level_of_fitness = self.fitness(individual)
            total_fittness.append(level_of_fitness)
            return level_of_fitness

        sorted_by_fitness = sorted(self.population, key=fitness, reverse=True)

        fittest_individuals = sorted_by_fitness[:self._sample_size(sorted_by_fitness, self.reproducing_percentage)]

        former_size = len(self.population)
        self.population.clear()
        while len(self.population) < former_size :
            breeding_group = random.sample(fittest_individuals, 2)

            descendant = reproduce(self, *breeding_group)

            self.population.add(descendant)

        return total_fittness

    def _mutate (self) :
        sample = random.sample(self.population, self._sample_size(self.population, self.mutation_percentage))

        for individual in sample :
            self.mutate(individual)

    def generation (self) :
        avg_fittness = self._reproduce()
        self._mutate()

        return avg_fittness

def _test_new_individual (ut) :
    selection = Selection('foo', None, None)
    ut.assert_equal(selection.individual().name, ('foo', 0))
    ut.assert_equal(selection.individual().name, ('foo', 1))

def __test__ (ut) :
    _test_new_individual(ut)

def __demo__ () :

    from nlplib.exterior.util import plot
    from nlplib.general.math import avg

    def fitness (individual) :
        return individual.traits['value']

    def mutate (individual) :
        individual.traits['value'] += round(random.triangular(-10, 10))

    selection = Selection('foo', fitness, mutate)

    for _ in range(100) :
        individual = selection.individual()
        individual.traits['value'] = 0
        mutate(individual)
        selection.population.add(individual)

    fitnesses = [selection.generation() for _ in range(100)]

     # Fitness increases with each successive generation.
    plot([avg(fitness) for fitness in fitnesses])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

