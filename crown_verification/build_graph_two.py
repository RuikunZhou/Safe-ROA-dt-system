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
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, num_inputs, num_layers, width):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)
        x = self.final_layer(x)
        return x

class ReLUNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width):
        super(ReLUNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.final_layer(x)
        return x

class quadratic_lyapunov_nn(nn.Module):
    def __init__(self):
        super(quadratic_lyapunov_nn, self).__init__()
        
    def forward(self,x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        # Two-machine power system quadratic form:
        # V_P(x) = 21.6816 * x1*x2 + 21.9377 * x1^2 + 33.6321 * x2^2
        res = 21.6816 * (x1 * x2) + 21.9377 * (x1 ** 2) + 33.6321 * (x2 ** 2)
        return res.unsqueeze(-1)
    

class quad_norm(nn.Module):
    def __init__(self, dynamics, P, epsilon):
        super(quad_norm, self).__init__()
        self.dynamics = dynamics
        self.register_buffer('P', P)
        
        init_state = torch.tensor([[0., 0.]], device=P.device)
        self.register_buffer('A', dynamics.linearization(init_state).squeeze(0))
        
        self.register_buffer('ATP', self.A.T @ self.P)
        
        Q = (self.A.T @ P @ self.A - P).to(torch.float64)
        r = torch.max(torch.abs(torch.linalg.eigvals(Q))) - epsilon
        self.register_buffer('r', r.to(P.dtype)) 
        
        P_norm = torch.linalg.norm(P, ord=2)
        self.register_buffer('P_norm', P_norm)
        # PTP = self.P.T @ self.P
        # P_norm_sq = torch.max(torch.abs(torch.linalg.eigvals(PTP)))
        # self.register_buffer('P_norm_sq', P_norm_sq)
    
    def forward(self, x):
        Dh = self.dynamics.compute_h_jacobian(x, self.A) # [batch, 2, 2]
        ATP = self.ATP.unsqueeze(0).expand(x.shape[0], -1, -1) # [batch, 2, 2]
        ATPDh = torch.bmm(ATP, Dh) # [batch, 2, 2]
    
        ATPDh_norm_sq = torch.sum(torch.sum(ATPDh * ATPDh, dim=2), dim=1)
        Dh_norm_sq = torch.sum(torch.sum(Dh * Dh, dim=2), dim=1)

        quad_term = 2 * torch.sqrt(ATPDh_norm_sq) + self.P_norm  * Dh_norm_sq
        return (self.r - quad_term).unsqueeze(-1)

class h_nn(nn.Module):
    def __init__(self):
        super(h_nn, self).__init__()
        
    def forward(self,x):
        x1 = x[:, 0]  
        x2 = x[:, 1]
        # h1 and h2 definitions:
        # h1 = 1 + 1/64 - ((x1 - 0.25)^2 + (x2 - 0.25)^2)
        # h2 = 1 + 1/64 - ((x1 - 0.25)^2 + (x2 + 0.25)^2)
        # h = max(h1, h2)
        
        h1 = 1 + (1/64) - ((x1 - 0.25)**2 + (x2 - 0.25)**2)
        h2 = 1 + (1/64) - ((x1 - 0.25)**2 + (x2 + 0.25)**2)
        h_val = torch.max(h1, h2).unsqueeze(-1)
        return h_val

class dynamics(nn.Module):
    def __init__(self, dt=0.1, delta=torch.pi/3):
        super(dynamics, self).__init__()
        self.dt = dt
        self.delta = torch.tensor(delta)

    def forward(self, x, u=None):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        # Two-machine power system dynamics:
        # x_next1 = x1 + dt*x2
        # x_next2 = x2 + dt*(-0.5*x2 - (sin(x1+delta)-sin(delta)))
        dx1 = x1 + self.dt * x2
        dx2 = x2 + self.dt * (-0.5 * x2 - (torch.sin(x1 + self.delta) - torch.sin(self.delta)))

        return torch.cat((dx1, dx2), dim=1)
    
        
    def linearization(self, x, u=None):
        x1 = x[:, 0:1].squeeze(-1) 
        x2 = x[:, 1:2].squeeze(-1)
        batch_size = x.shape[0]
        
        row0 = torch.stack([
            torch.zeros_like(x1),
            torch.ones_like(x1)
        ], dim=1) 
        
        row1 = torch.stack([
            -1 * torch.cos(x1 + self.delta),
            -0.5 * torch.ones_like(x1)
        ], dim=1)
        
        A = torch.stack([row0, row1], dim=1)  # [batch, ]
        I = torch.eye(2, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        F = I + self.dt * A
        
        return F
    
    def compute_h_jacobian(self, x, A, u=None):
        Df = self.linearization(x, u)
        A_expanded = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        Dh = Df - A_expanded 

        return Dh
    
class loss_nn(nn.Module):
    def __init__(self, dynamics, lyapunov, h, quadratic_lyapunov, quad_norm):
        super(loss_nn, self).__init__()
        self.dynamics = dynamics
        self.lyapunov = lyapunov
        self.h = h
        self.q = quadratic_lyapunov
        self.quad_norm = quad_norm
        
    def forward(self,x):
        new_x = self.dynamics(x, None)
        h_x = self.h(x)
        V_x = self.lyapunov(x)
        V_fx = self.lyapunov(new_x)
        V_qx = self.q(x)
        V_qfx = self.q(new_x)
        quad_norm = self.quad_norm(x)
        return torch.cat((V_fx - V_x, V_x, h_x, V_qx, V_qfx - V_qx, quad_norm, new_x[:, 0:1], new_x[:, 1:2],),dim=1)
