from torch import tensor
import torch
from plots import model_est

def run_model_with_prob(filter_model, z_measurements):
    '''
    Input model and list of measurements,
     using the model to calculate and
    Output the 'real' values and probability
    '''

    # Get Results
    x_posterior_array = []
    x_prior_array = []
    p_posterior_array = []
    p_prior_array = []
    k_ax = []

    for k, z in enumerate(z_measurements):
        k_ax.append(k)
        filter_model.kalman_step(z)
        x_posterior_array.append(filter_model.x_posterior.item())
        x_prior_array.append(filter_model.x_prior.item())
        p_posterior_array.append(filter_model.p_posterior.item())
        p_prior_array.append(filter_model.p_prior.item())

    return x_posterior_array, x_prior_array, p_posterior_array, p_prior_array, k_ax

def estimate_oneModel(filter_model, z_measurements,x_true, plot_title, a, b):

    # Run Model with measurements
    x_posterior_array, x_prior_array, p_posterior_array, p_prior_array, k_ax = \
        run_model_with_prob(filter_model, z_measurements)


    mse = torch.pow(tensor(x_posterior_array) - tensor(x_true), 2).sum() / len(x_true)

    #################################### Plots #####################################
    print(f'mse for {plot_title} is : {mse}')
    model_est(k_ax, x_true, x_prior_array, x_posterior_array, p_prior_array, p_posterior_array, a, b,
              model_name=filter_model.__class__.__name__)