from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import numpy as np
import itertools
import random

class FitnessCalc(object):

	def __init__(self, data, targets):
		self.min_fitness = 0.0
		self.max_fitness = 1.0
		self.data = data
		self.targets = targets
		self.num_classes = np.unique(self.targets).size
		self.folds = 10	

	"""	
	def cross_validate_acc(self, classifier):
		scores = cross_validation.cross_val_score(classifier, self.data, self.targets, cv=self.folds)
		return scores.mean()
	"""

	def cross_validate(self, classifier):
		self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
		skf = StratifiedKFold(n_splits=self.folds, shuffle=True)
		scores = 0
		
		for train_index, test_index in skf.split(self.data, self.targets):
			X_train, X_test = self.data[train_index], self.data[test_index]
			y_train, y_test = self.targets[train_index], self.targets[test_index]
			
			try:
				classifier.fit(X_train, y_train)
				# Setup the probability data structure
				probs = []
				prediction = []
				inst_probs = classifier.predict_proba(X_test).tolist()
				probs = zip(y_test, inst_probs)
				scores += self.MAUC(probs, self.num_classes)
			except ZeroDivisionError, e:
				print e
				scores += 0

		return scores / float(self.folds)
		
	def a_value(self, probabilities, zero_label, one_label):
		"""
		From: http://stuartlacy.co.uk/sites/default/files/MAUCpy.py.txt
		
    	Approximates the AUC by the method described in Hand and Till 2001,
    	equation 3.

    	NB: The class labels should be in the set [0,n-1] where n = # of classes.
    	The class probability should be at the index of its label in the
    	probability list.

    	I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    	for class '1' will be found in index 1 in the class probability list
    	wrapped inside the zipped list with the labels.

    	Args:
        	probabilities (list): A zipped list of the labels and the
            	class probabilities in the form (m = # data instances):
             	[(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              	(label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             	...
              	(labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             	]
        	zero_label (optional, int): The label to use as the class '0'.
            	Must be an integer, see above for details.
        	one_label (optional, int): The label to use as the class '1'.
            	Must be an integer, see above for details.

    	Returns:
        	The A-value as a floating point.
    	"""
		# Obtain a list of the probabilities for the specified zero label class
		expanded_points = []
		for instance in probabilities:
			if instance[0] == zero_label or instance[0] == one_label:
				expanded_points.append((instance[0], instance[1][zero_label]))
		sorted_ranks = sorted(expanded_points, key=lambda x: x[1])
		
		n0, n1, sum_ranks = 0, 0, 0
		# Iterate through ranks and increment counters for overall count and ranks of class 0
		for index, point in enumerate(sorted_ranks):
			if point[0] == zero_label:
				n0 += 1
				sum_ranks += index + 1  # Add 1 as ranks are one-based
			elif point[0] == one_label:
				n1 += 1
			else:
				pass  # Not interested in this class
		
		if n0 * n1 == 0:
			return 0.5 # Introducing some bias here...
		else:
			return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1) # Eqn 3
		
	def MAUC(self, data, num_classes):
		"""
    	From: http://stuartlacy.co.uk/sites/default/files/MAUCpy.py.txt
    	
    	Calculates the MAUC over a set of multi-class probabilities and
    	their labels. This is equation 7 in Hand and Till's 2001 paper.

    	NB: The class labels should be in the set [0,n-1] where n = # of classes.
    	The class probability should be at the index of its label in the
    	probability list.

    	I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    	for class '1' will be found in index 1 in the class probability list
    	wrapped inside the zipped list with the labels.

    	Args:
        	data (list): A zipped list (NOT A GENERATOR) of the labels and the
            	class probabilities in the form (m = # data instances):
             	[(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              	(label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             	...
              	(labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             	]
        	num_classes (int): The number of classes in the dataset.

    	Returns:
        	The MAUC as a floating point value.
    	"""

		# Find all pairwise comparisons of labels
		class_pairs = [x for x in itertools.combinations(xrange(num_classes), 2)]
		
		# Have to take average of A value with both classes acting as label 0 as this
		# gives different outputs for more than 2 classes
		sum_avals = 0
		for pairing in class_pairs:
			sum_avals += (self.a_value(data, zero_label=pairing[0], one_label=pairing[1]) +
						  self.a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0
		return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7
		
class SVCFitnessCalc(FitnessCalc):

	def __init__(self, data, targets):
		FitnessCalc.__init__(self, data, targets)
		
	def get_fitness(self, params):
		clf = SVC(coef0=params['coef0'].current, \
				  C=params['c'].current, \
				  degree=params['degree'].current, \
				  gamma=params['gamma'].current, \
				  kernel=params['kernel'].current, \
				  probability=True)

		fitness = self.cross_validate(clf)
		return fitness
				
class DTFitnessCalc(FitnessCalc):

	def __init__(self, data, targets):
		FitnessCalc.__init__(self, data, targets)
		
	def get_fitness(self, params):
		clf = DecisionTreeClassifier(criterion=params['criterion'].current, \
									splitter=params['splitter'].current, \
									max_features=params['max_features'].current, \
									max_depth=params['max_depth'].current, \
									min_samples_split=params['min_samples_split'].current, \
									min_samples_leaf=params['min_samples_leaf'].current, \
									min_weight_fraction_leaf=params['min_weight_fraction_leaf'].current, \
									max_leaf_nodes=params['max_leaf_nodes'].current)
		fitness = self.cross_validate(clf)
		#fitness = self.cross_validate_acc(clf)
		return fitness

class MLPFitnessCalc(FitnessCalc):

	def __init__(self, data, targets):
		FitnessCalc.__init__(self, data, targets)

	def get_fitness(self, params):
		clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'].current, \
							activation=params['activation'].current, \
							solver=params['solver'].current, \
							alpha=params['alpha'].current, \
							batch_size=params['batch_size'].current, \
							learning_rate=params['learning_rate'].current, \
							learning_rate_init=params['learning_rate_init'].current, \
							power_t=params['power_t'].current, \
							max_iter=params['max_iter'].current, \
							shuffle=True, \
							random_state=None,
							tol=params['tol'].current, \
							verbose=False, \
							warm_start=params['warm_start'].current, \
							momentum=params['momentum'].current, \
							nesterovs_momentum=params['nesterovs_momentum'].current, \
							early_stopping=params['early_stopping'].current, \
							validation_fraction=params['validation_fraction'].current, \
							beta_1=params['beta_1'].current, \
							beta_2=params['beta_2'].current, \
							epsilon=params['epsilon'].current)
		fitness = self.cross_validate(clf)
		return fitness
		
	