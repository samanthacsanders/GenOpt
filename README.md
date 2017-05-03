# GenOpt
Genetic algorithm for hyper-parameter optimization

## Overview
This code finds good hyper-parameter settings for a machine learning algorithm given a dataset. The user provides a dataset (.csv file), specifies the learning algorithm they would like to optimize over (MLP, SVM, or Decision Tree), and the amount of time they would like the optimization algorithm to run.

## Usage
```
python run_ga.py -d <path/to/dataset.csv> -o <path/to/output.csv> -l <MLP/SVM/DT> -r <runtime in seconds>
```

## Output
The program outputs a .csv file with the following columns:
- **Time:** The time in seconds to find the current parameters
- **Generation:** The genetic algorithm generation in which the current best parameters were found
- **Parameters:** The current best parameters
- **Fitness:** The Multi-Class AUC of the current best parameters

## Genetic Algorithm Hyper-Parameters
**Population:** 100 individuals in each generation. The initial population contains an individual with default hyper-parameters. This guarantees that the end solution will be at least as good as default hyper-parameters.

**Selection:** Tournament selection with 5 individuals per selection. Elitism is also used, carrying over the best individual from the previous generation to the next.

**Crossover:** Uniform crossover rate of 0.5

**Mutation:** 0.015 chance of mutating each gene
  - *Floating-point hyper-parameter mutation:* This code samples from a Gaussian distribution with mean set to the current value of the gene and the standard deviation is specified at the creation of the floating point gene. Sampling is repeated until a value within the specified range for the hyper-parameter is selected.
  - *Integer hyper-parameter mutation:* As with floating-point hyper-parameters, this code samples from a Gaussian distribution with mean set to the current value of the gene and the standard deviation is specified at the creation of the integer gene. The value samples is rounded to the nearest integer value. Sampling is repeated until a value within the specified range for the hyper-parameter is selected.
  - *List hyper-parameter mutation:* These gene types are used for hyper-parameters that have a mixture of types, ex: [None, 10, 100, 1000] or hyper-parameters that have string values. To mutate, the code randomly selects a member of the list of hyper-parameter settings that is different from the current setting.
  
  **Fitness Function:** The code uses 10-fold cross-validation for each model with Multi-Class AUC (MAUC) as the performance measure. See [this paper](https://link.springer.com/article/10.1023%2FA%3A1010920819831?LI=true) for an explanation of MAUC.
