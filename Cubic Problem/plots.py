from torch import tensor
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
from matplotlib import gridspec

def model_est(time_inx, x_true, x_prior_array, x_posterior_array,  p_prior_array, p_posterior_array, a, b, model_name='EKF'):
    # Plot the raw time series
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    # ax.scatter(time_inx, z_measurements, label='Measurements', color='gold')
    ax.plot(time_inx, x_true, label='x_true', linestyle='-', color='red', linewidth=0.8)
    ax.plot(time_inx, x_prior_array, label='x Predictor', linestyle='-', color='blue', linewidth=0.8)
    ax.plot(time_inx, x_posterior_array, label='x Corrector', linestyle='-', color='green', linewidth=0.8)

    ax.set_title(f'X Tracking ({model_name}) a={a}, b={b}')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('signal')
    ax.grid(False)
    ax.legend()


    # time series at once
    ax2 = fig.add_subplot(gs[1, 0])
    res = tensor(x_prior_array) - tensor(x_true)
    ax2.plot(time_inx, res, label='Error', linestyle='-', color='blue')
    ax2.plot(time_inx, torch.sqrt(tensor(p_prior_array)), label='p^0.5', linestyle='-', color='red')
    ax2.plot(time_inx, -torch.sqrt(tensor(p_prior_array)), linestyle='-', color='red')
    ax2.set_title(f'X Coordinate Predictor Error ({model_name})')
    ax2.grid(True)
    ax2.legend()

    # original data sequence.
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
    res = tensor(x_posterior_array) - tensor(x_true)
    ax3.plot(time_inx, res, label='Error', linestyle='-', color='blue')
    ax3.plot(time_inx, torch.sqrt(tensor(p_posterior_array)), label='p^0.5', linestyle='-', color='red')
    ax3.plot(time_inx, -torch.sqrt(tensor(p_posterior_array)), linestyle='-', color='red')
    ax3.set_title(f'X Coordinate Corrector Error ({model_name})')
    ax3.grid(True)
    ax3.legend()

    plt.show()



def models_compair_est(time_inx, x_true, x_posterior_results_list:list, res_inx2name:list,
                       a=None, b=None):
    '''Compare multiple models results
    UPPER  : all results
    DOWN : difference to true value
    '''
    # Create Figure
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Plot algorithms results
    ax = fig.add_subplot(gs[0, :])
    ## Plot true states
    ax.plot(time_inx, x_true, label='x_true', linestyle='-', color='red', linewidth=0.8)
    ## add each alg. results to plot
    for idx, model_results in enumerate(x_posterior_results_list):
        ax.plot(time_inx, model_results, label=f'{res_inx2name[idx]} x Predictor',
                linestyle='-', linewidth=0.8)
    ax.set_title(f'X Tracking compair for cubic problem with a={a}, b={b}')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('signal')
    ax.grid(False)
    ax.legend()

    # Plot models differences
    ax1 = fig.add_subplot(gs[1, :])
    first_error = abs(x_posterior_results_list[0] - tensor(x_true))
    second_error = abs(x_posterior_results_list[1] - tensor(x_true))
    diff = first_error - second_error
    ax1.plot(time_inx, diff, label='Error diff.', linestyle='-', color='red', linewidth=0.8)
    ax1.set_title(f'Difference  {res_inx2name[0]} Error - {res_inx2name[1]} Error')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('difference')
    ax1.grid(True)
    # ax1.legend()

    plt.show()


def models_compair_est_ByProb(time_inx, x_true, x_posterior_results_list:list,
                              p_posterior_results_list:list, res_inx2name:list,
                              a=None, b=None):

    # Create Figure
    count_models = len(res_inx2name)
    fig = plt.figure(constrained_layout=True)
    num_graph = count_models + 1 # 1 for results, models number for probabilities
    gs = gridspec.GridSpec(num_graph, 2, figure=fig)

    # Plot algorithms results
    ax = fig.add_subplot(gs[0, :])
    ## Plot true states
    ax.plot(time_inx, x_true, label='x_true', linestyle='-', color='red', linewidth=0.8)
    ## add each alg. results to plot
    for idx, model_results in enumerate(x_posterior_results_list):
        ax.plot(time_inx, model_results, label=f'{res_inx2name[idx]} x Predictor',
                linestyle='-', linewidth=0.8)
    ax.set_title(f'X Tracking compair for cubic problem with a={a}, b={b}')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('signal')
    ax.grid(False)
    ax.legend()

    # Plot models differences
    ax1 = fig.add_subplot(gs[1, :])
    first_error = abs(x_posterior_results_list[0] - tensor(x_true))
    second_error = abs(x_posterior_results_list[1] - tensor(x_true))
    diff = first_error - second_error
    ax1.plot(time_inx, diff, label='Error diff.', linestyle='-', color='red', linewidth=0.8)
    ax1.set_title(f'Difference  {res_inx2name[0]} Error - {res_inx2name[1]} Error')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('difference')
    ax1.grid(True)
    # ax1.legend()

    plt.show()

#############################
### PARTICLE FILTER PLOTS ###
#############################

def plot_test(particles, weights):
    '''plot Particles over weights'''
    plt.cla()
    plt.scatter(particles, weights, c='green', label='particles')
    # plt.scatter(z, 0, c='r', label='measurement')
    plt.xlabel('value')
    plt.ylabel('weight')
    plt.legend()
    # self.iteration += 1
    # plt.title(f'Iteration number: {self.iteration}')
    plt.show()

def par2time_plot(timeI_particles):
    ''' Plot particles over Time'''
    plt.cla()
    size = timeI_particles[0].size(0)
    time = torch.ones(size)
    t = torch.arange(size)
    for inx, par in enumerate(timeI_particles):
        plt.scatter(par, time*inx, c=t, cmap='tab20b')#, label='particles')
    # plt.scatter(z, 0, c='r', label='measurement')
    plt.xlabel('value')
    plt.ylabel('time')
    plt.legend()
    # self.iteration += 1
    # plt.title(f'Iteration number: {self.iteration}')
    plt.show()
