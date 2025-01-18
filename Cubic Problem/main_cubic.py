from torch import tensor
import torch
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
path_parent = Path().parent.absolute()
if path_parent not in sys.path:
    sys.path.append(path_parent)

from plots import model_est
from Filters import EKF, IEKF, Particle_Filter
from Estimate_SoleModel import estimate_oneModel
from Estimate_CompairModels import estimate_compairModels_byDiff

# Example 1
# EKF & IEKF Qubic problem
# Motion Model : x_(k+1 )=0.9x_k+√Q w_k
# Measurement Model : z_k=αx_k+βx_k^3+√R v_k
# Try: α=[0,0.1,1]    β=[1,0]
#################################### Parameters #####################################
from Parameters import a, b, m, n, Q, R, x0, p0

tested_model = 'Particle_Filter' # 'EKF' # 'IEKF' # 'Particle_Filter'
mode = 'compair' # 'sole' # 'compair'
torch.manual_seed(400)
#################################### Motion and Measurement Models #####################################

from Parameters import motion_model, measurement_model

#loop for testing every argument saperatly
arg_list = list(range(-1,-20,-3))
for val in arg_list:
    b=val

#################################### Create ground Truth and Measerement Array #####################################

    x_true = []
    zz = True
    for k in torch.arange(100):
        if zz :
            x = motion_model(tensor(0.5), Q=Q, noise=True)
            zz = False
        else:
            x = motion_model(x, Q=Q, noise=True)
        x_true.append(x)

    z_measurements = [measurement_model(xxx, a=a, b=b, R=R, noise=True) for xxx in x_true]

    #################################### Estimate Models #####################################
    # Create Model
    if mode == 'sole':
        if tested_model == 'EKF':
            filter_model = EKF(motion_model, measurement_model, a=a, b=b, x0=x0 ,p0=p0, Q=Q, R=R, m=m, n=n)
        elif tested_model == 'IEKF':
            filter_model = IEKF(motion_model, measurement_model, a=a, b=b, x0=x0 , p0=p0, Q=Q, R=R, m=m, n=n, i=10)
        elif tested_model == 'Particle_Filter':
            filter_model = Particle_Filter(motion_model, measurement_model, a=a, b=b, x0=x0 , p0=p0, Q=Q, R=R, m=m, n=n, N=500)

        estimate_oneModel(filter_model, z_measurements, x_true, val, a, b)

    if mode == 'compair':
        filter_model_EKF = EKF(motion_model, measurement_model, a=a, b=b, x0=x0, p0=p0, Q=Q, R=R, m=m, n=n)
        filter_model_IEFK = IEKF(motion_model, measurement_model, a=a, b=b, x0=x0, p0=p0, Q=Q, R=R, m=m, n=n,
                                 i=10)
        filter_model_PF = Particle_Filter(motion_model, measurement_model, a=a, b=b, x0=x0, p0=p0, Q=Q, R=R, m=m, n=n,
                                           N=1000)

        models2compair = {
            'EKF':filter_model_EKF,
            'IEKF': filter_model_IEFK,
            'Particle_Filter': filter_model_PF,
        }
        filter_models = [models2compair['IEKF'], models2compair['Particle_Filter'],]# models2compair['EKF']]
        estimate_compairModels_byDiff(filter_models, z_measurements, x_true, a, b)
