from genetic_algorithm import GeneticAlgorithm
import numpy as np
import pandas as pd
import time
import csv
import getopt
import sys
from threading import Thread
import threading
import thread
from fitness_calc import SVCFitnessCalc
from fitness_calc import DTFitnessCalc 
from fitness_calc import MLPFitnessCalc
from population import Population

def get_data(input_file):
	"""Takes in a .csv file and returns two numpy arrays: data and labels"""
	dataset = pd.read_csv(input_file, delimiter=',')
	X = dataset.iloc[:,1:-1].as_matrix()
	y = dataset.iloc[:,-1].as_matrix()
	return X, y
	
def run_algorithm(data, labels, output_file, learner):
	"""Runs the genetic algorithm"""
	fitness_calc = None
	if learner == 'SVM':
		fitness_calc = SVCFitnessCalc(data, labels)
	elif learner == 'DT':
		fitness_calc = DTFitnessCalc(data, labels)
	elif learner == 'MLP':
		fitness_calc = MLPFitnessCalc(data, labels)
	else:
		print 'Learner error'
	
	ga = GeneticAlgorithm(fitness_calc, learner)
	
	with open(output_file, 'a') as csvfile:
		out_writer = csv.writer(csvfile)
		out_writer.writerow(['time', 'generation', 'parameters', 'fitness'])
		
	try:
	
		start_time = time.time()
		pop = Population(size=50, fit_calc=fitness_calc, initialize=True, indiv_type=learner)
		generation = 1
		fittest = pop.get_fittest()
		fitness = fittest.get_fitness()
		best_fitness = fitness
	
		params = fittest.get_params_str()
		cur_time = time.time() - start_time
		with open(output_file, 'a') as csvfile:
			out_writer = csv.writer(csvfile)
			out_writer.writerow([cur_time, generation, params, best_fitness])
		
		while fitness < fitness_calc.max_fitness:
			pop = ga.evolve_pop(pop)
			generation += 1
			fittest = pop.get_fittest()
			fitness = fittest.get_fitness()
		
			if fitness > best_fitness:
				best_fitness = fitness
				params = fittest.get_params_str()
				cur_time = time.time() - start_time
				with open(output_file, 'a') as csvfile:
					out_writer = csv.writer(csvfile)
					out_writer.writerow([cur_time, generation, params, best_fitness])
		
		sys.exit()
	
	except ZeroDivisionError, e:
		with open(output_file, 'a') as csvfile:
			out_writer = csv.writer(csvfile)
			out_writer.writerow(['ZeroDivisionError'])
			out_writer.writerow([e])
		sys.exit()
		
	
def run_thread(data, labels, runtime, output_file, learner):
	t = Thread(target=run_clock, args=(runtime,))
	t.daemon = True
	t.start()
	run_algorithm(data, labels, output_file, learner)
	
def run_clock(runtime):
	t = threading.Timer(runtime, quit)
	t.start()
	
def quit():
	thread.interrupt_main()
	
def main():
	try:
		options, remainder = getopt.getopt(sys.argv[1:], 'i:o:l:r:', ['input=', 'output=', 'learner=', 'runtime='])
	except getopt.error, msg:
		print msg
	
	#TODO: check validity of the arguments (esp the learner)
	
	for opt, arg in options:
		if opt in ('-i', '--input'):
			input_filename = arg
		elif opt in ('-o', '--output'):
			output_file = arg
		elif opt in ('-l', '--learner'):
			learner = arg
		elif opt in ('-r', '--runtime'):
			runtime = float(arg)
	
	data, labels = get_data(input_filename)
	run_thread(data, labels, runtime, output_file, learner)

if __name__ == '__main__':
	main()