import os
import sys
import numpy as np


def iterate(path, func):
    sweep_dirs = list(filter(lambda x: x != ".DS_Store", os.listdir(path)))
    # parameter_sweeps = {}
    func_dict = {}
    for dir in sweep_dirs:

        sweep_path = path + dir + "/"
        run_dirs = list(filter(lambda x: x != ".DS_Store", os.listdir(sweep_path)))
        res = func(sweep_path)
        # print("The res", res)
        func_dict[dir] = res

    return func_dict

def get_best_result(path):
    sweep_dirs = list(filter(lambda x: x != ".DS_Store", os.listdir(path)))
    parameter_sweeps = {}
    for dir in sweep_dirs:

        sweep_path = path + dir + "/"
        run_dirs = list(filter(lambda x: x != ".DS_Store", os.listdir(sweep_path)))

        means = [read_mean(sweep_path + run_dir + "/mean.csv") for run_dir in run_dirs]
        # print(run_folder)
        # print(len(means))
        if len(run_dirs) > 4:
            parameter_sweeps[dir] = np.mean(means)
        # mean_file = list(filter(lambda x: x == "mean.csv", os.listdir(run_folder)))[0]
        # print(mean_file)
        # f = open(run_folder + run + "/mean.csv", "r")
        # value = float(f.read())
        # print(value)
                # runs = filter(lambda elem: elem != "mean.csv", runs)
    # print(parameter_sweeps)
    s = [(k, parameter_sweeps[k]) for k in sorted(parameter_sweeps, key=parameter_sweeps.get)]
    print(s[0])

def read_mean(path):
    f = open(path, "r")
    value = float(f.read())
    return value


def main():
    path = sys.argv[1]
    get_best_result(path)
    # print(path)


if __name__ == "__main__":
    main()
