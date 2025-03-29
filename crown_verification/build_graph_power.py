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
            x = torch.torch.tanh(x)
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
            x = torch.torch.relu(x)
        x = self.final_layer(x)
        return x
    
class quadratic_lyapunov_nn(nn.Module):
    def __init__(self, P):
        super(quadratic_lyapunov_nn, self).__init__()
        self.register_buffer('P', P)
        
    def forward(self, x):
        return torch.sum(x * (x @ self.P), axis=1, keepdim=True)

class quad_norm(nn.Module):
    def __init__(self, dynamics, P, epsilon):
        super(quad_norm, self).__init__()
        self.dynamics = dynamics
        self.register_buffer('P', P)
        
        init_state = torch.tensor([[0., 0., 0., 0.]], device=P.device)
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
        Dh = self.dynamics.compute_h_jacobian(x, self.A)  # [batch, 4, 4]
        ATP = self.ATP.unsqueeze(0).expand(x.shape[0], -1, -1)  # [batch, 4, 4]
        ATPDh = torch.bmm(ATP, Dh)  # [batch, 4, 4]
    
        ATPDh_norm_sq = torch.sum(torch.sum(ATPDh * ATPDh, dim=2), dim=1)
        Dh_norm_sq = torch.sum(torch.sum(Dh * Dh, dim=2), dim=1)
        # Dh_norm_fourth = torch.square(Dh_norm_sq)
        # quad_term = 8 * ATPDh_norm_sq + 2 * self.P_norm_sq * Dh_norm_fourth
        # return (self.r * self.r - quad_term).unsqueeze(-1)

        quad_term = 2 * torch.sqrt(ATPDh_norm_sq) + self.P_norm * Dh_norm_sq
        return (self.r - quad_term).unsqueeze(-1)
        
    
class dynamics(nn.Module):
    def __init__(self, dt=0.05, alpha1=1.0, alpha2=1.0, beta1=0.5, beta2=0.5, D1=0.4, D2=0.5):
        super(dynamics, self).__init__()
        self.dt = dt
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.D1 = D1
        self.D2 = D2

    def forward(self, x, u=None):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]

        dx1 = x2
        dx2 = -self.alpha1 * torch.sin(x1) - self.beta1 * torch.sin(x1 - x3) - self.D1 * x2
        dx3 = x4
        dx4 = -self.alpha2 * torch.sin(x3) - self.beta2 * torch.sin(x3 - x1) - self.D2 * x4
        
        rx1 = x1 + self.dt * dx1
        rx2 = x2 + self.dt * dx2
        rx3 = x3 + self.dt * dx3
        rx4 = x4 + self.dt * dx4

        return torch.cat((rx1, rx2, rx3, rx4), dim=1)
    
    def linearization(self, x, u=None):
        x1 = x[:, 0:1].squeeze(-1) 
        x2 = x[:, 1:2].squeeze(-1)
        x3 = x[:, 2:3].squeeze(-1)
        x4 = x[:, 3:4].squeeze(-1)
        
        batch_size = x.shape[0]
        
        row0 = torch.stack([
            torch.zeros_like(x1),
            torch.ones_like(x1),
            torch.zeros_like(x1),
            torch.zeros_like(x1)
        ], dim=1) 
        
        row1 = torch.stack([
            -self.alpha1 * torch.cos(x1) - self.beta1 * torch.cos(x1 - x3),
            -self.D1 * torch.ones_like(x1),
            self.beta1 * torch.cos(x1 - x3),
            torch.zeros_like(x1)
        ], dim=1)
        
        row2 = torch.stack([
            torch.zeros_like(x1),
            torch.zeros_like(x1),
            torch.zeros_like(x1),
            torch.ones_like(x1)
        ], dim=1)
        
        row3 = torch.stack([
            self.beta2 * torch.cos(x3 - x1),
            torch.zeros_like(x1),
            -self.alpha2 * torch.cos(x3) - self.beta2 * torch.cos(x3 - x1),
            -self.D2 * torch.ones_like(x1)
        ], dim=1)
        
        A = torch.stack([row0, row1, row2, row3], dim=1)  # [batch, 4, 4]
        I = torch.eye(4, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        F = I + self.dt * A
        
        return F
    
    def compute_h_jacobian(self, x, A, u=None):
        Df = self.linearization(x, u)
        A_expanded = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        Dh = Df - A_expanded 

        return Dh
    
class loss_nn(nn.Module):
    def __init__(self, dynamics, lyapunov, quadratic_lyapunov, quad_norm):
        super(loss_nn, self).__init__()
        self.dynamics = dynamics
        self.lyapunov = lyapunov
        self.q = quadratic_lyapunov
        self.quad_norm = quad_norm
        
    def forward(self,x):
        new_x = self.dynamics(x, None)
        V_x = self.lyapunov(x)
        V_fx = self.lyapunov(new_x)
        V_qx = self.q(x)
        V_qfx = self.q(new_x)
        quad_norm = self.quad_norm(x)
        return torch.cat((V_fx - V_x, V_x, V_qx, V_qfx - V_qx, quad_norm, new_x[:, 0:1], new_x[:, 1:2], new_x[:, 2:3], new_x[:, 3:4]),dim=1)
