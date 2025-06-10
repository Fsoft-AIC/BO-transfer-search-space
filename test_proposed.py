import pickle as pkl
from transfer_learning_BO.Proposed_global import Proposed_global
import numpy as np
import os
from functions.function import *
import argparse
import warnings
warnings.filterwarnings("ignore")

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
        5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]

scenario_1 = ["ackley_s1", "powell_s1", "dixon_s1", "levy_s1"]
scenario_2 = ["stybtang_s2", "ackley_s2", "rosenbrock_s2", "griewank_s2"]
scenarion_3 = ["hyper_s3", "ackley_s3", "rastrigin_s3", "perm_s3"]
real_world_scenario = ["push", "rover"]

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_func', type=str, default="ackley_s1", choices = scenario_1 + scenario_2 + scenarion_3 + real_world_scenario)
    parser.add_argument('--restart', type=int, default=20)
    parser.add_argument('--max_evals', type=int, default=50)
    parser.add_argument('--save_result', type=bool, default=False)
    args = parser.parse_args()
    test_func = args.test_func
    restart = args.restart
    max_evals = args.max_evals
    save_result = args.save_result

    # For scenario 1
    if test_func == "ackley_s1":
        objective = Ackley_(dim=4)
        num_src_data = 1500
    elif test_func == "powell_s1":
        objective = Powell(dim=4)
        num_src_data = 1500
    elif test_func == "dixon_s1":
        objective = Dixon(dim=5)
        num_src_data = 1500
    elif test_func == "levy_s1":
        objective = Levy(dim=8)
        num_src_data = 1500
    # For scenario 2
    elif test_func == "stybtang_s2":
        objective = Stybtang(dim=4)
        num_src_data = 1500
    elif test_func == "ackley_s2":
        objective = Ackley(dim=5)
        num_src_data = 1500
    elif test_func == "rosenbrock_s2":
        objective = Rosenbrock(dim = 6)
        num_src_data = 1500
    elif test_func == "griewank_s2":
        objective = Griewank(dim=30)
        num_src_data = 1500
    # For scenario 3
    elif test_func == "hyper_s3":
        objective = Hyper(dim=5)
        num_src_data = 1500
    elif test_func == "ackley_s3":
        objective = Ackley(dim=6)
        num_src_data = 1500
    elif test_func == "rastrigin_s3":
        objective = Rastrigin(dim=10)
        num_src_data = 1500
    elif test_func == "perm_s3":
        objective = Perm(dim=20)
        num_src_data = 1500
    # For real world scenario
    elif test_func == "rover":
        objective = Rover()
        num_src_data = 2000
    elif test_func == "push":
        objective = Push()
        num_src_data = 2000

    result = []
    for j in range(restart):
        print("-----------------Proposed----------------------------")
        filename_initial = "initials\{}\{}_seed_{}.pkl".format(test_func, test_func, seeds[j])
        filename = "dataset\\{}\\{}_M_15_N_{}.pkl".format(test_func, test_func, num_src_data)
        dir = "result/{}".format(test_func)
        if not os.path.exists(dir):
            os.makedirs(dir) 
        filename_save_result = "result/{}/bo_proposed_seed_{}.pkl".format(test_func, seeds[j])
        with open(filename, 'rb') as f:
            dataset_ = pkl.load(f)
        proposed = Proposed_global(f = objective,
            max_evals = max_evals,
            source_data = dataset_,
            num_src_hpo_trial = num_src_data,
            draw_progress = True,
            verbose=True,
            initial_file = filename_initial)
        proposed.optimize()
        if save_result:
            with open(filename_save_result, 'wb') as f:
                data = np.minimum.accumulate(proposed.fX.reshape(-1))
                pkl.dump(data, f)
        result.append(proposed.fX.min().item())
    print("--------------------Result--------------------------")
    print("The total result of Proposed: ", np.array(result).mean())
    
    