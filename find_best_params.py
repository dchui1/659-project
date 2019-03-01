import os


# training_files = os.listdir(training_dir)
# dev_files = os.listdir(dev_dir)
# dev_files.sort()
# models_dict = dict()
# training_start_time = time.time()
# for file in filter(lambda x: x.endswith('.tra'), training_files):
file_location = "bayesian-exploration-results/tabular/gridworld/"
list_of_agents = os.listdir(file_location)
for agent in list_of_agents:
    agent_folder = file_location + agent + "/"
    list_of_param_sweeps = os.listdir(agent_folder)
    ranking = []


    for param_sweep in list_of_param_sweeps:
        mean_values_array = []
        runs = os.listdir(agent_folder + param_sweep)

        for i, run in enumerate(runs):
            file = open(agent_folder + param_sweep + "/" + run + "/mean.csv", "r")
            mean_value = file.read()
            mean_value = int(mean_value)
            
