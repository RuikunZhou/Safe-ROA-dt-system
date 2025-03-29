import torch
import torch.nn as nn
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.quadrotor2d as quadrotor2d
# import neural_lyapunov_training.reverse_van_der_pol as van
# import neural_lyapunov_training.polynomial as poly
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.output_train_utils as output_train_utils
import crown_verification.build_graph_van as build_graph_van
import crown_verification.build_graph_two as build_graph_two
import crown_verification.build_graph_power as build_graph_power
from crown_verification.build_graph_van import loss_nn
from collections import OrderedDict

def create_model(
    dynamics,
    path=None,
    lyapunov_func="build_graph_van.Net",
    loss_func="build_graph_van.loss_nn",
    h="build_graph_van.h_nn",
    quadratic_lyapunov="build_graph_van.quadratic_lyapunov_nn",
    quad_norm="build_graph_van.quad_norm"
):
    hnet = eval(h)()
    lya = eval(lyapunov_func)(
        2, 2, 30
    )
    # P = torch.tensor([[16.3896, -11.1403/2],
    #               [-11.1403/2,   11.4027]])

    P = torch.tensor([[21.9377, 21.6816/2],
                  [21.6816/2,   33.6321]])

    quadratic_lyapunov_nn = eval(quadratic_lyapunov)()
    quad_norm = eval(quad_norm)(dynamics, P, 1e-4)
    loss = eval(loss_func)(
        dynamics, lya, hnet, quadratic_lyapunov_nn, quad_norm
    )
    if path is not None:
        loss.load_state_dict(torch.load(path))
    
    return loss

def create_model_noh(
    dynamics,
    path=None,
    lyapunov_func="build_graph_van.Net",
    loss_func="build_graph_van.loss_nn",
    quadratic_lyapunov="build_graph_van.quadratic_lyapunov_nn",
    quad_norm="build_graph_van.quad_norm"
):
    lya = eval(lyapunov_func)(
        4, 2, 50
    )
    
    P = torch.tensor([[75.38338277, 8.01350751, -17.16922748,  -1.85495727],
                  [8.01350751,  49.84298468, 5.94377593, 3.54815247],
                  [-17.16922748,   5.94377593,  68.96662985,  10.42966701],
                  [ -1.85495727,   3.54815247,  10.42966701,  44.34043763]])
    
    quadratic_lyapunov_nn = eval(quadratic_lyapunov)(P)
    quad_norm = eval(quad_norm)(dynamics, P, 1e-4)
    loss = eval(loss_func)(
        dynamics, lya, quadratic_lyapunov_nn, quad_norm
    )
    if path is not None:
        loss.load_state_dict(torch.load(path))
    
    return loss

def create_van_der_pol(**kwargs):
    return create_model(
        build_graph_van.dynamics(),
        **kwargs,
    )
    
def create_van_der_pol_ReLU(**kwargs):
    return create_model(
        build_graph_van.dynamics(),
        lyapunov_func="build_graph_van.ReLUNet",
        **kwargs,
    )
    
def create_two_machine_power(**kwargs):
    return create_model(
        build_graph_two.dynamics(),
        lyapunov_func="build_graph_two.Net",
        loss_func="build_graph_two.loss_nn",
        h="build_graph_two.h_nn",
        quadratic_lyapunov="build_graph_two.quadratic_lyapunov_nn",
        quad_norm='build_graph_two.quad_norm',
        **kwargs
    )
    
def create_two_machine_power_ReLU(**kwargs):
    return create_model(
        build_graph_two.dynamics(),
        lyapunov_func="build_graph_two.ReLUNet",
        loss_func="build_graph_two.loss_nn",
        h="build_graph_two.h_nn",
        quadratic_lyapunov="build_graph_two.quadratic_lyapunov_nn",
        **kwargs
    )
    
def create_power(**kwargs):
    return create_model_noh(
        build_graph_power.dynamics(),
        lyapunov_func="build_graph_power.Net",
        loss_func="build_graph_power.loss_nn",
        quadratic_lyapunov="build_graph_power.quadratic_lyapunov_nn",
        quad_norm='build_graph_power.quad_norm',
        **kwargs
    )