import argparse
import os
import subprocess
from utils.ExperimentDescription import ExperimentDescription

cpus = 8

def parse_args():
  parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
  parser.add_argument("-e", type=str, help="path to experiment description json file")
  parser.add_argument("-r", type=int, help="number of runs to complete")
  parser.add_argument("-b", type=str, default='results', help="base path for saving results")

  args = parser.parse_args()
  if args.b == None or args.r == None:
    print('Please run again using (without angle braces):')
    print('python -m scripts.run_sweep_local -e path/to/exp.json -r <num>')
    exit(1)

  return args

args = parse_args()
exp = ExperimentDescription(args.e, 0, args.r)
num = exp.num_permutations * args.r

runs = '{0..' + str(num-1) + '}'

parallel = f'parallel -j{cpus} python parameter_sweep.py -e {args.e} -b {args.b} -i ::: {runs}'
print(parallel)
process = subprocess.run(parallel, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
