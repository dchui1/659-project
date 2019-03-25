import os

# FOR TABULAR RESULTS
tabular_location = "bayesian-exploration-results/tabular/gridworld/"
linearQ_location = "bayesian-exploration-results/tabular/cts-gridworld/"
blr_tabularQ_loc = "bayesian-exploration-results/blr/gridworld/"
bql_loc = "bayesian-exploration-results/Bayesian Q learning/gridworld/"

def process_agent_results(file_location):
    agents = os.listdir(file_location)

    agent_results = {}
    agents = filter(lambda elem: elem != "ucb", agents)

    for agent in agents:
        agent_folder = file_location + agent + "/"
        parameterizations = os.listdir(agent_folder)
    
        results = []
        for param in parameterizations:
            param_folder = agent_folder + param + "/"
            runs = os.listdir(param_folder)
            runs = filter(lambda elem: elem != "mean.csv", runs)
       
            param_results = []
            for run in runs:
                f = open(param_folder + run + "/mean.csv", "r")
                value = float(f.read())
                param_results.append(value)

                results.append((param, sum(param_results) / len(param_results), len(param_results)))

        results = sorted(results, key=lambda elem: elem[1])[:30]
        agent_results[agent] = results
    return agent_results

agent_results = {**process_agent_results(tabular_location),
                 **process_agent_results(linearQ_location),
                 **process_agent_results(blr_tabularQ_loc),
                 **process_agent_results(bql_loc)
}

for agent, results in agent_results.items():
    print("RESULTS FOR AGENT: " + agent)
    for res in results:
        print(res[0], res[1], res[2])
    
# FOR LINEAR FUNCTION APPROXIMATION
