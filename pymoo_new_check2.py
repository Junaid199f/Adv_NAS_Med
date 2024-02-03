import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

crossover = SBX()
selection = RandomSelection()
# Define the custom problem
class SimpleProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0, 0]), xu=np.array([10, 10]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Define objectives
        f1 = np.sum(x)
        f2 = np.prod(x)
        out["F"] = np.column_stack([f1, f2])

# Instantiate the custom problem
sampling = FloatRandomSampling()
problem = SimpleProblem()
mutation = PolynomialMutation(prob=1.0, eta=20)
# Define the algorithm
algorithm = NSGA2(
    pop_size=100,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
)

# Solve the problem
res = minimize(problem, algorithm, termination=("n_gen", 100), seed=1)

# Print the results
print("Best solution found:")
for solution in res.X:
    print(f"Variables: {solution}, Objectives: {problem.evaluate(solution, return_values_of=['F'])}")
