import argparse
import os
from utils.ExperimentDescription import ExperimentDescription

def parse_args():
  parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
  parser.add_argument("-e", type=str, help="path to experiment description json file")
  parser.add_argument("-b", type=str, default='results', help="base path for saving results")

  args = parser.parse_args()
  if args.b == None:
    print('Please run again using (without angle braces):')
    print('python -m scripts.find_best.py -e path/to/exp.json')
    exit(1)

  return args

args = parse_args()
exp = ExperimentDescription(args.e, 0)

base = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}'
param_results = os.listdir(base)

def getResults(path):
    with open(f'{base}/{path}/mean.csv', 'r') as f:
        return { 'path': path, 'mean': float(f.read()) }

results = map(getResults, param_results)
min_result = min(results, key = lambda x: x['mean'])

print(min_result)
