import math
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
from medmnist import INFO
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
from mealpy import Tuner
from mealpy.evolutionary_based.DE import L_SHADE, BaseDE
from mealpy.evolutionary_based.ES import CMA_ES
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based import PSO
from mealpy.swarm_based.ACOR import OriginalACOR
from mealpy.utils import io
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


def evaluate_arch(self, ind, dataset, measure):

    return random.randint(10,10)


class SOGA(Optimizer):
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num,medmnist_dataset,is_medmnist):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num,medmnist_dataset,is_medmnist)

    def evaluate_fitness_single_mealpy(self, solution):
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])
        data_flag = self.medmnist_dataset
        output_root = './output'
        num_epochs = self.epochs
        gpu_ids = '0'
        batch_size = 64
        download = True
        run = 'model1'
        self.grad_clip=5
        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
                individual.append(math.floor(solution[i]*len(self.attentions)))
        individuals = np.asarray(individual)
        #print(decode_operations(individual, self.pop.indexes))
        decoded_individual = NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
                                            decode_cell(decode_operations(individual, self.pop.indexes)),self.is_medmnist,
                                            self.dropout_rate, 'FP32', False)

        loss = self.evaluator.train(decoded_individual, self.epochs, self.grad_clip, 'valid', data_flag, output_root,
                                num_epochs, gpu_ids, batch_size, download, run,is_final=False)
        return  loss
    def train_final_individual(self,solution,medmnist_dataset):
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])
        data_flag = self.medmnist_dataset
        output_root = './output'
        num_epochs = 10
        gpu_ids = '0'
        batch_size = 64
        download = True
        run = 'model1'
        self.grad_clip = 5
        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                individual.append(random.choice(self.attentions))
        decoded_individual = NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
                                                  decode_cell(decode_operations(individual, self.pop.indexes)),
                                                  self.is_medmnist,
                                                  self.dropout_rate, 'FP32', False)

        # Count the number of parameters
        num_params = sum(p.numel() for p in decoded_individual.parameters())

        # Calculate the model size in MB (assuming float32 data type)
        model_size_mb = num_params * 4 / (1024 ** 2)

        print(f"Number of parameters: {num_params}")
        print(f"Model size (MB): {model_size_mb:.2f} MB")

        print(f"The genotype of individual is:")
        decode_cell(decode_operations(individual, self.pop.indexes))
        hash_indv = hashlib.md5(str(self.pop.individual).encode("UTF-8")).hexdigest()
        avg_loss =[]
        for i in range(5):
            loss = self.evaluator.train(decoded_individual, 10, self.grad_clip, 'test', data_flag, output_root,
                                num_epochs, gpu_ids, batch_size, download, run,is_final=True)
            avg_loss.append(loss)
        print("Final loss is ",avg_loss)
    def mealypy_evolve(self, algorithm, pop_size=15, epoch=20,CR=0.9,WF=0.8,de_strategy=1,medmnist_dataset=None):

        ## Design a problem dictionary for multiple objective functions above
        problem_multi = {
            "fit_func": self.evaluate_fitness_single_mealpy,
            "lb": [0 for i in range(48)],
            "ub": [0.99 for i in range(48)],
            "minmax": "max",
            "obj_weights": [1],  # Define it or default value will be [1, 1, 1]
            "save_population": True,
            "log_to": "file",
            "log_file": "result.log",  # Default value = "mealpy.log"
        }
        paras_de = {
            "epoch": [100],
            "pop_size": [100],
            "wf": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "cr": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "strategy": [1]
        }
        # term = {
        #     "max_epoch": 2
        # }

        if algorithm == 'pso':
            model = PSO.OriginalPSO(epoch=50, pop_size=50)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset)
        elif algorithm == 'de':
            wf = 0.7
            cr = 0.9
            strategy = 0
            model = BaseDE(epoch, pop_size, wf, cr, strategy)
            #tuner = Tuner(model, paras_de)
            #tuner.execute(problem=problem_multi, n_trials=5, n_jobs=6, mode="parallel", n_workers=6, verbose=True)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position,medmnist_dataset)
        elif algorithm == 'lshade':
            miu_f = 0.5
            miu_cr = 0.5
            model = L_SHADE(epoch, pop_size, miu_f, miu_cr)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset)
        elif algorithm == 'ga':
            pc = 0.9
            pm = 0.05
            model1 = BaseGA(epoch, pop_size, pc, pm)
            best_position, best_fitness = model1.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset)
        elif algorithm == 'cmaes':
            model = CMA_ES(epoch, pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset)
        elif algorithm == 'aco':
            sample_count = 25
            intent_factor = 0.5
            zeta = 1.0
            model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset)
        else:
            print("error")
        ## Define the model and solve the problem

        ## Save model to file
        io.save_model(model, "results/model.pkl")
        ## You can access them all via object "history" like this:
        model.history.save_global_objectives_chart(filename="hello/goc")
        model.history.save_local_objectives_chart(filename="hello/loc")

        model.history.save_global_best_fitness_chart(filename="hello/gbfc")
        model.history.save_local_best_fitness_chart(filename="hello/lbfc")

        model.history.save_runtime_chart(filename="hello/rtc")

        model.history.save_exploration_exploitation_chart(filename="hello/eec")

        model.history.save_diversity_chart(filename="hello/dc")

        model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[3], filename="hello/tc")
