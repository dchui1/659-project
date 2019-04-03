import argparse
import os
from utils.ExperimentDescription import ExperimentDescription

tasks_per_cpu = 8
cpus = 8
memory = 8 # in gigabytes

sbatch_args = ' '.join([
    # use martha's resource allocation account
    '--account=def-amw8',
    # largest time allotment for fastest scheduling group
    '--time=11:59:00',
    # number of cores to request
    f'--ntasks={cpus}',
    # amount of memory each core will need
    f'--mem-per-cpu={memory}G',
    # location to put the log files
    f'--output=$SCRATCH/job_output_\%j.txt',
    # email option,
    f'--mail-type=ALL',
    # email
    f'--mail-user=parash@ualberta.ca'
])

def parse_args():
  parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
  parser.add_argument("-e", type=str, help="path to experiment description json file")
  parser.add_argument("-r", type=int, help="number of runs to complete")
  parser.add_argument("-b", type=str, default='results', help="base path for saving results")

  args = parser.parse_args()
  if args.b == None or args.r == None:
    print('Please run again using (without angle braces):')
    print('python -m scripts.run_sweep_cc -e path/to/exp.json -r <num>')
    exit(1)

  return args

def bundle(it, num):
    coll = []
    for a in it:
        coll.append(a)
        if len(coll) == num:
            yield coll
            coll = []

    yield coll

args = parse_args()
cwd = os.getcwd()
exp = ExperimentDescription(args.e, 0, args.r)
num = exp.num_permutations * args.r

for jobs in bundle(range(0, num - 1), tasks_per_cpu * cpus):
    runs = ' '.join(map(str, jobs))
    parallel = f'parallel -j{cpus} --delay 1 srun -N1 -n1 python parameter_sweep.py -e {args.e} -b {args.b} -i ::: {runs}'

    slurm_file = f'''#!/bin/bash
cd {cwd}
{parallel}
    '''

    with open('auto_slurm.sh', 'w') as f:
        f.write(slurm_file)

    os.system(f'sbatch {sbatch_args} auto_slurm.sh')
    os.remove('auto_slurm.sh')
