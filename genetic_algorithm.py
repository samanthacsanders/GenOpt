import random
from population import Population
from individual import SVCIndividual
from individual import DTIndividual
from individual import MLPIndividual

class GeneticAlgorithm(object):
	
	def __init__(self, fitness_calc, learner, elitism=True, uniform_rate=0.5, mutate_rate=0.015, tournament_size=5):
		self.elitism = elitism
		self.uniform_rate = uniform_rate
		self.mutate_rate = mutate_rate
		self.tournament_size = tournament_size
		self.fitness_calc = fitness_calc
		self.learner = learner
		
	def evolve_pop(self, pop):
		"""Evolve a population"""
		new_pop = Population(pop.size, pop.fitness_calc, False, pop.indiv_type)
		if self.elitism: 
			new_pop.individuals.append(pop.get_fittest())
		
		""" Crossover """	
		while len(new_pop.individuals) <= pop.size:	
			indiv1 = self.tourn(pop)
			indiv2 = self.tourn(pop)
			new_pop.individuals.append(self.crossover(indiv1, indiv2))
				
		""" Mutate """
		if not self.elitism:
			self.mutate(new_pop.individuals[0])
				
		for i in range(1, pop.size):
			self.mutate(new_pop.individuals[i])
			
		return new_pop
		
	def crossover(self, indv1, indv2):
		"""Crossover two individuals with a uniform crossover rate"""
		if self.learner == 'DT':
			new_indv = DTIndividual(self.fitness_calc)
		elif self.learner == 'SVM':
			new_indv = SVCIndividual(self.fitness_calc)
		elif self.learner == 'MLP':
			new_indv = MLPIndividual(self.fitness_calc)

		for key, _ in new_indv.genes.iteritems():
			if random.random() < self.uniform_rate:
				new_indv.genes[key].current = indv1.genes[key].current
			else: 
				new_indv.genes[key].current = indv2.genes[key].current
			
		return new_indv
		
	def mutate(self, indv):
		"""Mutate each gene in an individual with probability 'mutate_rate'"""
		for _, gene in indv.genes.iteritems():
			if random.random() < self.mutate_rate:
				gene.mutate()
		return indv
	
	def tourn(self, pop):
		"""Tournament selection"""
		tournament = Population(self.tournament_size, pop.fitness_calc, False, pop.indiv_type)
		tournament.individuals = random.sample(pop.individuals, self.tournament_size)
		return tournament.get_fittest()
		