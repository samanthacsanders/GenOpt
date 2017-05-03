import random
from individual import SVCIndividual, DTIndividual, MLPIndividual

class Population():

	def __init__(self, size, fit_calc, initialize, indiv_type):
		self.fittest = None
		self.individuals = []
		self.size = size
		self.fitness_calc = fit_calc
		self.indiv_type = indiv_type
		
		if initialize:
			if indiv_type == 'SVM':
				default_indiv = SVCIndividual(self.fitness_calc)
			elif indiv_type == 'DT':
				default_indiv = DTIndividual(self.fitness_calc)
			elif indiv_type == 'MLP':
				default_indiv = MLPIndividual(self.fitness_calc)

			self.individuals.append(default_indiv)
			
			for i in range(size-1):
				if indiv_type == 'SVM':
					indiv = SVCIndividual(self.fitness_calc)
				elif indiv_type == 'DT':
					indiv = DTIndividual(self.fitness_calc)
				elif indiv_type == 'MLP':
					indiv = MLPIndividual(self.fitness_calc)
					
				indiv.randomize()
				self.individuals.append(indiv)
		
	def get_fittest(self):
		"""Returns the fittest individual in the population"""
		if self.fittest is None:
			best_fitness = self.fitness_calc.min_fitness
			# if all individuals have equal fitness, choose a random individual
			self.fittest = random.choice(self.individuals)

			for indiv in self.individuals:
				temp = indiv.get_fitness()
				if temp > best_fitness:
					best_fitness = temp
					self.fittest = indiv
		return self.fittest