from gene import FloatGene, ListGene, IntGene, BoolGene, TupleGene

class Individual():

	def __init__(self, fitness_calc):
		self.fitness = None
		self.genes = None
		self.fitness_calc = fitness_calc
		
	def randomize(self):
		for _, gene in self.genes.iteritems():
			gene.randomize()
			
	def get_fitness(self):
		"""Returns the fitness of the individual"""
		if self.fitness is None:
			self.fitness = self.fitness_calc.get_fitness(self.genes)
		return self.fitness
	
class SVCIndividual(Individual):

	def __init__(self, fitness_calc):
		Individual.__init__(self, fitness_calc)
		self.length = 5 #number of parameters we're optimizing
		
		c = FloatGene(current=1.0, std=0.1, min=0.0, max=1.0)
		kernel = ListGene(current='rbf', values=['rbf', 'linear', 'sigmoid', 'poly'])
		degree = IntGene(current=3, min=2, max=5)
		gamma = FloatGene(current=0.2, std=0.1, min=0.0, max=1.0)
		coef0 = FloatGene(current=0.0, std=0.1, min=0.0, max=1.0)
		
		self.genes = {'c': c, \
					  'kernel': kernel, \
					  'degree': degree, \
					  'gamma': gamma, \
					  'coef0': coef0}
		
	def get_params_str(self):
		"""Returns the parameters of the SVM individual as a string"""
		c = str(self.genes['c'].current)
		ker = str(self.genes['kernel'].current)
		deg = str(self.genes['degree'].current)
		g = str(self.genes['gamma'].current)
		coef = str(self.genes['coef0'].current)
		
		params = 'C=%s, kernel=%s, degree=%s, gamma=%s, coef0=%s' % (c, ker, deg, g, coef)
		return params
		
class DTIndividual(Individual):

	def __init__(self, fitness_calc):
		Individual.__init__(self, fitness_calc)
		self.length = 8 # number of parameters
		
		criterion = ListGene(current='gini', values=['gini', 'entropy'])
		splitter = ListGene(current='best', values=['best', 'random'])
		max_features = ListGene(current=None, values=[None, 'sqrt', 'log2'])
		max_depth = ListGene(current=None, values=[None, 10, 100, 1000])
		min_samples_split = IntGene(current=2, min=2, max=10)
		min_samples_leaf = IntGene(current=1, min=1, max=20)
		min_weight_fraction_leaf = FloatGene(current=0.0, std=0.25, min=0.0, max=0.5)
		max_leaf_nodes = ListGene(current=None, values=[None, 10, 100, 1000])
		
		self.genes = {'criterion': criterion, \
					  'splitter': splitter, \
					  'max_features': max_features, \
					  'max_depth': max_depth, \
					  'min_samples_split': min_samples_split, \
					  'min_samples_leaf': min_samples_leaf, \
					  'min_weight_fraction_leaf': min_weight_fraction_leaf, \
					  'max_leaf_nodes': max_leaf_nodes}
					  
	def get_params_str(self):
		"""Returns the parameters of the DecisionTree individual as a string"""
		c = str(self.genes['criterion'].current)
		s = str(self.genes['splitter'].current)
		max_f = str(self.genes['max_features'].current)
		max_d = str(self.genes['max_depth'].current)
		min_ss = str(self.genes['min_samples_split'].current)
		min_sl = str(self.genes['min_samples_leaf'].current)
		min_wfl = str(self.genes['min_weight_fraction_leaf'].current)
		max_ln = str(self.genes['max_leaf_nodes'].current)
		
		params = 'criterion=%s, splitter=%s, max_features=%s, max_depth=%s, min_samples_split=%s, \
				  min_samples_leaf=%s, min_weight_fraction_leaf=%s, max_leaf_nodes=%s' \
				  % (c, s, max_f, max_d, min_ss, min_sl, min_wfl, max_ln)

		return params

class MLPIndividual(Individual):

	def __init__(self, fitness_calc):
		Individual.__init__(self, fitness_calc)
		self.length = 18 # number of parameters

		num_hidden_layers = IntGene(current=1, min=1, max=3) # the default tuple length is 1 (100,)
		hidden_layer_sizes = TupleGene(current=(100,), size=num_hidden_layers, gene_list=[IntGene(100, 1, 200), IntGene(0, 0, 200), IntGene(0, 0, 200)])
		activation = ListGene(current='relu', values=['relu', 'identity', 'logistic', 'tanh'])
		solver = ListGene(current='adam', values=['adam', 'lbfgs', 'sgd'])
		alpha = FloatGene(current=0.0001, std=0.01, min=0.00001, max=1)
		batch_size = ListGene(current='auto', values=['auto', 10, 100, 1000])
		learning_rate = ListGene(current='constant', values=['constant', 'invscaling', 'adaptive'])
		max_iter = IntGene(current=200, min=50, max=5000)
		tol = FloatGene(current=1e-4, std=0.01, min=1e-6, max=1e-1)
		learning_rate_init = FloatGene(current=0.001, std=0.01, min=0.0001, max=1)
		power_t = FloatGene(current=0.5, std=0.1, min=0.0001, max=1)
		warm_start = BoolGene(current=False)
		momentum = FloatGene(current=0.9, std=0.25, min=0, max=1)
		nesterovs_momentum = BoolGene(current=True)
		early_stopping = BoolGene(current=False)
		validation_fraction = FloatGene(current=0.1, std=0.25, min=0, max=1)
		beta_1 = FloatGene(current=0.9, std=0.25, min=0, max=1)
		beta_2 = FloatGene(current=0.999, std=0.25, min=0, max=1)
		epsilon = FloatGene(current=1e-8, std=1e-3, min=1e-10, max=1e-2)

		self.genes = {'hidden_layer_sizes': hidden_layer_sizes, \
					  'activation': activation, \
					  'solver': solver, \
					  'alpha': alpha, \
					  'batch_size': batch_size, \
					  'learning_rate': learning_rate, \
					  'max_iter': max_iter, \
					  'tol': tol, \
					  'learning_rate_init': learning_rate_init, \
					  'power_t': power_t, \
					  'warm_start': warm_start, \
					  'momentum': momentum, \
					  'nesterovs_momentum': nesterovs_momentum, \
					  'early_stopping': early_stopping, \
					  'validation_fraction': validation_fraction, \
					  'beta_1': beta_1, \
					  'beta_2': beta_2, \
					  'epsilon': epsilon}

	def get_params_str(self):
		"""Returns the parameters of the MLP individual as a string"""
		hls = str(self.genes['hidden_layer_sizes'].current)
		act = str(self.genes['activation'].current)
		sol = str(self.genes['solver'].current)
		alp = str(self.genes['alpha'].current)
		bat = str(self.genes['batch_size'].current)
		lrt = str(self.genes['learning_rate'].current)
		mxi = str(self.genes['max_iter'].current)
		tol = str(self.genes['tol'].current)
		lri = str(self.genes['learning_rate_init'].current)
		pow = str(self.genes['power_t'].current)
		war = str(self.genes['warm_start'].current)
		mom = str(self.genes['momentum'].current)
		nes = str(self.genes['nesterovs_momentum'].current)
		ear = str(self.genes['early_stopping'].current)
		val = str(self.genes['validation_fraction'].current)
		be1 = str(self.genes['beta_1'].current)
		be2 = str(self.genes['beta_2'].current)
		eps = str(self.genes['epsilon'].current)

		params = 'hidden_layer_sizes=%s, activation=%s, solver=%s, alpha=%s, batch_size=%s, learning_rate=%s, learning_rate_init=%s, power_t=%s, max_iter=%s, shuffle=True, random_state=None, tol=%s, verbose=False, warm_start=%s, momentum=%s, nesterovs_momentum=%s, early_stopping=%s, validation_fraction=%s, beta_1=%s, beta_2=%s, epsilon=%s'  % (hls, act, sol, alp, bat, lrt, lri, pow, mxi, tol, war, mom, nes, ear, val, be1, be2, eps)

		return params