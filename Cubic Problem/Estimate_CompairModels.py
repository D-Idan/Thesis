from torch import tensor
from Estimate_SoleModel import run_model_with_prob
from plots import models_compair_est, models_compair_est_ByProb


def estimate_compairModels_byDiff(filter_models:list, z_measurements,x_true, a, b):
    x_posterior_results_list = []
    res_inx2name = []
    for model in filter_models:
        # Get Results
        x_posterior_array, _, _, _, k_ax =\
            run_model_with_prob(model, z_measurements)

        x_posterior_results_list.append(tensor(x_posterior_array))
        res_inx2name.append(model.__class__.__name__)

    #################################### Plots #####################################
    models_compair_est(k_ax, x_true, x_posterior_results_list, res_inx2name,
    a = a, b = b)
    
def compairModels_byProb(filter_models:list, z_measurements,x_true, a, b):
    '''
    Print models comparison by values and probabilities estimations
    :param filter_models: list of models
    :param z_measurements: list of measurements
    :param x_true: list of true values
    :param a: Cubic problem parameter
    :param b: Cubic problem parameter
    :return: print results for each model over one figure for comparison
    '''

    # Data variables
    x_posterior_results_list = []
    p_posterior_results_list = []
    res_inx2name = []

    for model in filter_models:
        # Get Results per model
        x_posterior_array, _, p_posterior_array, _, k_ax =\
            run_model_with_prob(model, z_measurements)

        # Gather data
        x_posterior_results_list.append(tensor(x_posterior_array))
        p_posterior_results_list.append(tensor(x_posterior_array))
        res_inx2name.append(model.__class__.__name__)

    #################################### Plots #####################################
    models_compair_est_ByProb(k_ax, x_true, x_posterior_results_list,
                              p_posterior_results_list, res_inx2name,
                              a = a, b = b)