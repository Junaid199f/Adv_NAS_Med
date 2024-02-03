from mealpy.evolutionary_based import DE
import objective_function as of
from mealpy.tuner import Tuner

if __name__ == '__main__':
    algoname = 'de_tune'
    num_run = 2
    max_generations = 500
    for i in problems:
        prob = of.GetFuncDetail(i)
        fitness_function = prob.Evaluate
        repair_solution = prob.repair_solution
        dim_ = prob.get_function_details() * 2
        fname = 'Tuner_history/tunenaszc_' + i + '.csv'

        lbs = [0] * (dim_ // 2) + [10] * (dim_ // 2)
        ubs = [1] * (dim_ // 2) + [20] * (dim_ // 2)
        problem = {
            "lb": lbs,
            "ub": ubs,
            "minmax": "min",
            "fit_func": fitness_function,
            "name": "naszc_" + i,
            "log_to": None,
            "repair_solution": prob.repair_solution
        }
        paras_de = {
            "epoch": [100],
            "pop_size": [100],
            "wf": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "cr": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "strategy": [1],
            "repair_solution": [repair_solution]
        }
        # term = {
        #     "max_epoch": 2
        # }
        model = DE.BaseDE()
        tuner = Tuner(model, paras_de)
        tuner.execute(problem=problem, n_trials=5, n_jobs=6, mode="parallel", n_workers=6, verbose=True)
        # tuner.execute(problem=problem, n_trials=10, mode="parallel", n_workers=4)
        print(tuner.best_row)
        # print(tuner.best_score)
        # print(tuner.best_params)
        # print(type(tuner.best_params))
        #
        # print(tuner.best_algorithm)
        tuner.export_results(fname, save_as="csv")
        #
        # best_fitness = tuner.resolve(mode="thread", n_workers=4, termination=term)
        # print(best_fitness)
        # print(tuner.problem.get_name())
        # print(tuner.best_algorithm.get_name())