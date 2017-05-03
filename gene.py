import random
import sys

class Gene(object):
	
	def __init__(self, current):
		self.current = current
		
	def get_current(self):
		return self._current
		
				
class FloatGene(Gene):
	
	def __init__(self, current, std, min, max):
		self.std = std
		self.min = min
		self.max = max
		Gene.__init__(self, current)
			
	def mutate(self): 
		while True:
			try:
				rand_val = random.gauss(self.get_current(), self.std)
				self.set_current(rand_val)
			except:
				e = sys.exc_info()[0]
				continue
			break
		
	def randomize(self):
		self.set_current(random.uniform(self.min, self.max))
		
	def set_current(self, value):
		if value < self.min or value > self.max:
			raise ValueError()
		self._current = value
	
	current = property(Gene.get_current, set_current)
		
class IntGene(Gene):

	def __init__(self, current, min, max):
		self.min = min
		self.max = max
		self.std = (max - min + 1) / 2.0
		Gene.__init__(self, current)
		
	def mutate(self):
		# taking into consideration if min == max
		while True and (self.min != self.max):
			try:
				rand_val = int(round(random.gauss(self.get_current(), self.std)))
				self.set_current(rand_val)
			except:
				e = sys.exc_info()[0]
				continue
			break
		
	def randomize(self):
		if self.min != self.max:
			self.set_current(random.randint(self.min, self.max))
		
	def set_current(self, value):
		if value < self.min or value > self.max:
			raise ValueError()
		self._current = value
		
	current = property(Gene.get_current, set_current)
	
class ListGene(Gene):

	def __init__(self, current, values):
		self.values = values
		Gene.__init__(self, current)
		
	def mutate(self):
		new_value = random.choice(self.values)

		# Do not set the new value to be the same as the current value
		while new_value == self.get_current():
			new_value = random.choice(self.values)
			
		self.set_current(new_value)
		
	def randomize(self):
		self.set_current(random.choice(self.values))
		
	def set_current(self, value):
		if value not in self.values:
			raise ValueError()
		self._current = value
		
	current = property(Gene.get_current, set_current)
		
class BoolGene(Gene):

	def __init__(self, current):
		Gene.__init__(self, current)

	def mutate(self):
		# Toggle the current value
		self.set_current(not self.get_current())

	def randomize(self):
		self.set_current(bool(random.getrandbits(1)))

	def set_current(self, value):
		if not isinstance(value, (bool,)):
			raise ValueError()
		self._current = value

	current = property(Gene.get_current, set_current)

class TupleGene(Gene):

	def __init__(self, current, size, gene_list):
		"""
		gene_list: {list} A TupleGene is made up of other gene types.
		The variable gene_list is a list of the genes that make
		up the values in the tuple. Must be the length of the maximum
		size of the tuple

		size: {IntGene} The number of values in the tuple

		current: {tuple} the default tuple
		"""
		
		self.size = size
		
		self.gene_list = gene_list
		Gene.__init__(self, current)

	def mutate(self):
		tuple_values = []
		self.size.mutate()

		for i in range(self.size.current):
			self.gene_list[i].mutate()
			while self.gene_list[i].current == 0:
				self.gene_list[i].mutate()
			tuple_values.append(self.gene_list[i].current)

		self.set_current(tuple(tuple_values))

	def randomize(self):
		tuple_values = []
		self.size.randomize()

		for i in range(self.size.current):
			self.gene_list[i].randomize()
			while self.gene_list[i].current == 0:
				self.gene_list[i].randomize()
			tuple_values.append(self.gene_list[i].current)

		self.set_current(tuple(tuple_values))

	def set_current(self, value):
		self._current = value


	current = property(Gene.get_current, set_current)


