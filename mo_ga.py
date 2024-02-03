import random

import numpy as np
import torch
import json
import random
import os
from copy import deepcopy

import numpy as np
import random
import torch
import torchvision
import csv
import hashlib
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from numpy import savetxt
from datetime import datetime
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt

from evaluate import Evaluate
from model import NetworkCIFAR
import operations_mapping
from utils import decode_cell, decode_operations
from optimizer import Optimizer

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.igd import IGD

def evaluate_arch(self, ind, dataset, measure):

    return random.randint(10,10)


class NAS(Problem):
    def __init__(self, n_var=6, n_obj=5, dataset='cifar10', xl=None, xu=None,pop=None,population_size=None, number_of_generations=None, crossover_prob=None, mutation_prob=None, blocks_size=None,
                         num_classes=None, in_channels=None, epochs=None, batch_size=None, layers=None, n_channels=None, dropout_rate=None, retrain=None,
                         resume_train=None, cutout=None, multigpu_num=None,medmnist_dataset=None,is_medmnist=None, save_dir=None, seed=0,
                 objectives_list=None, args=None):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.xl = xl
        self.xu = xu
        self._save_dir = save_dir
        self._n_generation = 0
        self._n_evaluated = 0
        self.archive_obj = []
        self.archive_var = []
        self.seed = seed
        self.dataset = dataset
        self.pop = pop
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.crossover_prob =crossover_prob
        self.mutation_prob =mutation_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.resume_train = resume_train
        self.cutout = cutout
        self.multigpu_num = multigpu_num
        self.blocks_size =blocks_size
        self.evaluator = Evaluate(self.batch_size,medmnist_dataset,is_medmnist)
        self.retrain = retrain
        self.attentions = [i for i in range(0, len(operations_mapping.attentions))]

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        individuals =[]
        for j in range(x.shape[0]):
            indv= []
            for i in range(32):
                if i % 2 == 0:
                    indv.append(x[j][i])
                else:
                    indv.append(int(random.choice(self.pop.params_choices[str(i)])))
                    indv.append(random.choice(self.attentions))

            individuals.append(indv)
        individuals = np.asarray(individuals)
        decoded_individuals = [NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(decode_operations(individuals[i], self.pop.indexes)), self.dropout_rate, 'FP32', False) for i in
                                    range(0, individuals.shape[0], 1)]

        for i in range(individuals.shape[0]):
            loss = self.evaluator.evaluate_zero_cost(decoded_individuals[i],self.epochs)
            objs[i][0] = -loss['synflow']
            objs[i][1] = loss['params']
        # logging.info('Generation: {}'.format(self._n_generation))
        #

        #
        # for i in range(x.shape[0]):
        #     for j in range(len(self.objectives_list)):
        #         # all objectives assume to be MINIMIZED !!!!!
        #         obj = evaluate_arch(self, ind=x[i], dataset=self.dataset, measure=self.objectives_list[j])
        #
        #         if 'accuracy' in self.objectives_list[j] or self.objectives_list[j] == 'synflow':
        #             objs[i, j] = -1 * obj
        #             print(obj)
        #             print(objectives_list[j])
        #         else:
        #             objs[i, j] = obj
        #     self.archive_obj, self.archive_var = archive_check(objs[i], self.archive_obj, self.archive_var, x[i])
        #     self._n_evaluated += 1
        #
        #     igd_dict, igd_norm_dict = calc_IGD(self, x=x, objs=objs)
        #
        # igd_norm_list = []
        # for dataset in self.datasets:
        #     igd_temp = list(igd_norm_dict[dataset].values())
        #     igd_temp.insert(0, dataset)
        #     igd_norm_list.append(igd_temp)
        #
        # headers = ['']
        # for j in range(1, len(self.objectives_list)):
        #     headers.append('test accuracy - ' + self.objectives_list[j])
        # logging.info(tabulate(igd_norm_list, headers=headers, tablefmt="grid"))
        #
        # self._n_generation += 1
        out["F"] = objs


class MOGA(Optimizer):
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num,medmnist_dataset,is_medmnist):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num,medmnist_dataset,is_medmnist)

    def evolve(self):
        pop_size = 5
        seed = 50
        n_gens = 5
        objectives_list = ['synflow', 'params']
        xl= [ 0.0 for i in range(48)]
        xu= [ 0.99 for i in range(48)]
        xl = np.asarray(xl)
        xu = np.asarray(xu)
        n_obj = len(objectives_list)
        n_var = 48  # NATS-Bench
        problem = NAS(objectives_list=objectives_list, n_var=n_var,
                      n_obj=n_obj,
                      xl=xl, xu=xu,pop = self.pop,population_size=self.population_size, number_of_generations = self.number_of_generations, crossover_prob = self.crossover_prob, mutation_prob= self.mutation_prob, blocks_size=self.blocks_size,
                         num_classes=self.num_classes, in_channels=self.in_channels, epochs=self.epochs, batch_size=self.batch_size, layers=self.layers, n_channels=self.n_channels, dropout_rate=self.dropout_rate, retrain=self.retrain,
                         resume_train=self.resume_train, cutout=self.cutout, multigpu_num=self.multigpu_num,medmnist_dataset = self.medmnist_dataset,is_medmnist = self.is_medmnist)

        algorithm = NSGA2(pop_size=pop_size,
                          sampling=FloatRandomSampling(),
                          crossover=TwoPointCrossover(prob=0.9),
                          mutation=PolynomialMutation(prob=1.0 / n_var),
                          eliminate_duplicates=True)

        stop_criteria = ('n_gen', n_gens)

        results = minimize(
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            save_history=True,
            termination=stop_criteria
        )
        print(results.F)
        # Access the Pareto front and Pareto set
        pareto_front = results.F
        pareto_set = results.X
        # Assuming 'results' contains your optimization results
        # results.F contains the objective values

        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population

        for algo in results.history:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])


# metric = IGD(pf, zero_to_one=True)
#
# igd = [metric.do(_F) for _F in hist_F]
#
# plt.plot(n_evals, igd,  color='black', lw=0.7, label="Avg. CV of Pop")
# plt.scatter(n_evals, igd,  facecolor="none", edgecolor='black', marker="p")
# plt.axhline(10**-2, color="red", label="10^-2", linestyle="--")
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD")
# plt.yscale("log")
# plt.legend()
# plt.show()