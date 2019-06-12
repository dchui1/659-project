
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
from agg_runs import aggregate_results
from scripts.rankbest import get_best_result
import argparse
import os

def main(args):
    # path = sys.argv[1]
    path = args.path
    results = get_best_result(path, aggregate_results)
    # print(results)
    print("Type of results", type(results))
    for run, result in results.items():

        fig = plt.figure()
        ax = plt.axes()
        # plotRewards(ax, rewards, stderr, fileName)
            # runResults.append(steps)
        # (steps, stderr) = averageOverRuns(runResults)
        plotRewards(ax, result[0], result[1], run)


        plt.legend()

        plt.title(run)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Average Number of Steps to Reach Goal")
        if args.save:
            # print("Save called")
            # print("Args results", args.results)
            filename = os.getcwd() + "/" + args.results + "/" + run + ".png"
            print(filename)
            plt.savefig(filename)
        else:
            plt.show()


def plotRewards(ax, rewards, stderr, label):
  (low_ci, high_ci) = confidenceInterval(rewards, stderr)
  ax.plot(rewards, label=label)
  ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

def confidenceInterval(mean, stderr):
  return (mean - stderr, mean + stderr)


def parse_args():
    parser = argparse.ArgumentParser("Plot stuff")
    parser.add_argument("-path", type=str, help="path to algorithm")
    parser.add_argument("-results", type=str, help="directory to save results to",
                        default="plots")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
