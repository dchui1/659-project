import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

q_array = [0.675, 0.75, 0.84, 0.92, 0.99]
# run experiments with q = 0.92, q = 1.0 (100 runs each)
# change json file accordingly
# then run this code
# and make sure that plots make sense!

for q in q_array:
    q_vals = np.load('tmp/rs/q_values_q{}.npy'.format(q))
    b_vals = np.load('tmp/rs/b_values_q{}.npy'.format(q))
    sv_allruns = np.load('tmp/rs/s_visits_allruns_q{}.npy'.format(q))
    runs = sv_allruns.shape[0]
    assert b_vals.shape == q_vals.shape
     # convert numpy to heat_map (how??)

    timesteps = q_vals.shape[0]
    num_states = q_vals.shape[1]
    num_acs = q_vals.shape[2]
    x = np.arange(0, timesteps)

    # q_vals:
    for i in range(num_states):
        fig = plt.figure()
        ax = plt.axes()
        y_left = q_vals[:, i, :][:, 0]
        y_right = q_vals[:, i, :][:, 1]
        ax.plot(x, y_left, label='left')
        ax.plot(x, y_right, label='right')
        plt.title("Q_values per 50th Timestep for state{}_q{}".format(i, q))
        plt.xlabel("Timesteps")
        plt.ylabel("Q_vals")
        plt.legend(loc='lower right')
        path_name = 'plots/rs/q{}/qvals/qvals_state{}_q{}.png'.format(q, i, q)
        plt.savefig(path_name, format='png')
    #
    # # bonus b_values:
    for i in range(num_states):
        fig = plt.figure()
        ax = plt.axes()
        y_left = b_vals[:, i, :][:, 0]
        y_right = b_vals[:, i, :][:, 1]
        ax.plot(x, y_left, label='left')
        ax.plot(x, y_right, label='right')
        plt.title("Bonuses per 50th Timestep for state{}_q{}".format(i, q))
        plt.xlabel("Timesteps")
        plt.ylabel("Bonus_vals")
        plt.legend(loc='lower right')
        path_name = 'plots/rs/q{}/bvals/bvals_state{}_q{}.png'.format(q, i, q)
        plt.savefig(path_name, format='png')

    # state visitations:
    mean_vis = np.zeros(num_states)
    stderr_array = np.zeros(num_states)
    for s_idx in range(num_states):
        mean_vis[s_idx] = np.mean(sv_allruns[:, s_idx])
        stderr_array[s_idx] = scipy.stats.sem(sv_allruns[:, s_idx])

    # plot state vis
    fig = plt.figure()
    x = ['s0', 's1', 's2', 's3', 's4', 's5']
    pos = np.arange(len(x))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_ylim(top=100)
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(x)

    plt.xlabel('States')
    plt.ylabel('Percentage of State Visitations (%)')
    plt.title('Percentage of State Visitations (Aver. 100 runs), q = {}'.format(q))
    plt.bar(pos, list(mean_vis), width, align='edge', color='r', edgecolor='k', capsize=10, yerr=stderr_array)
    # save the file and display
    path_name = 'plots/rs/q{}/histogram_q{}.png'.format(q, q)
    plt.savefig(path_name, format='png')
