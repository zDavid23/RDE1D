from __future__ import absolute_import

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import euler_1D_py
import numpy as np
from clawpack import pyclaw
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#!/usr/bin/env python
# encoding: utf-8

def getPrimitive(q,state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']
    
    rho = q[0,:]
    u = q[1,:]/q[0,:]
    z = q[3,:]/q[0,:]
    P = (gamma-1.)*(q[2,:] - 0.5*(q[1,:]**2/q[0,:]))
    T = P/rho
    
    return rho,u,P,T,z

def step_Euler(solver,state,dt):
    # Get parameters:
    gamma = state.problem_data['gamma']
    Da = state.problem_data['Da']
    Ea = state.problem_data['Ea']
    Tign = state.problem_data['Tign']
    Pref = state.problem_data['Pref']
    rhoref = state.problem_data['rhoref']
    AR = state.problem_data['AR']
    hv = state.problem_data['hv']
    s = state.problem_data['s']

    # ODE solved explicitly with 2-stage, 2nd-order Runge-Kutta method.
    dt2 = dt/2.
    q = state.q
    
    # Get state prior to integrating:
    rho,u,P,T,z = getPrimitive(q,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-10.0,0.5) - np.heaviside(state.t-12.0,0.5))*np.heaviside(1.0-xc,0.0) + np.heaviside(state.t-10.0,0.5)*K
    Ic = np.sqrt(gamma)*(2.0/(gamma+1.0))**((gamma + 1.0)/(2.0*(gamma - 1.0)))
    r = (1.0+(gamma-1.0)/2.0)**(-gamma/(gamma-1.0))
    uref = np.sqrt(Pref/rhoref)
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    s = s*np.heaviside(state.t - 10.0,0.0)
    alpha = Ic*np.sqrt(Pref*rhoref)/(rhoref*uref)
    omega = K*rho*(1.0-z)*Da

    qstar = np.empty(q.shape)
    qstar[0,:] = q[0,:] + dt2*alpha*(H*AR - np.sqrt(P*rho))
    qstar[1,:] = q[1,:]
    qstar[2,:] = q[2,:] + dt2*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    qstar[3,:] = q[3,:] + dt2*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

    # Update primitive variables:
    rho,u,P,T,z = getPrimitive(qstar,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-10.0,0.5) - np.heaviside(state.t-12.0,0.5))*np.heaviside(1.0-xc,0.0) + np.heaviside(state.t-10.0,0.5)*K
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    omega = K*rho*(1.0-z)*Da
    
    q[0,:] = q[0,:] + dt*alpha*(H*AR - np.sqrt(P*rho))
    q[1,:] = q[1,:]
    q[2,:] = q[2,:] + dt*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    q[3,:] = q[3,:] + dt*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

def init(state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']
    L = state.problem_data['L']
    xc = state.grid.x.centers
    
    P = 1.0 + 0.0*xc
    T = 1.0 + 0.0*xc
    z = 0.5*np.sin((2.0*np.pi/L)*xc)+0.5
    rho = P/T

    state.q[0,:] = rho
    state.q[1,:] = 0.0 
    state.q[2,:] = P/(gamma-1.)
    state.q[3,:] = rho*z

    
# Specify Riemann Solver and instantiate solver object:
rs_HLL = euler_1D_py.euler_rq1D
rs_HLLC = euler_1D_py.euler_hllc_rq1D_counterProp
solver = pyclaw.ClawSolver1D(rs_HLLC)
solver.kernel_language = 'Python'

# Set Boundary Conditions:
solver.step_source = step_Euler
solver.bc_lower[0]=pyclaw.BC.periodic # custom inlet
solver.bc_upper[0]=pyclaw.BC.periodic # custom outlet
solver.aux_bc_lower[0]=pyclaw.BC.extrap
solver.aux_bc_upper[0]=pyclaw.BC.extrap
solver.max_steps = 100000
solver.cfl_desired = 0.1

# Working Fluid:
pref = 1.0
rhoref = 1.0
gamma = 1.29
R = 1.0
Tref = 1.0

# Injector/mixing:
s = 0.07
AR = 0.2
# Kinetics:
Da = 208.3
Tign = 6.2
hv = 24.577
Ea = 11.0

# Geometry:
L = 24.0
mx = 4800

# 1-D domain specification:
x = pyclaw.Dimension(0,L,mx,name='x')
domain = pyclaw.Domain([x])
state = pyclaw.State(domain,num_eqn=4,num_aux=4)

# Working fluid:
state.problem_data['gamma'] = gamma
state.problem_data['gamma1'] = gamma-1.0
state.problem_data['Pref'] = pref
state.problem_data['rhoref'] = rhoref
state.problem_data['R'] = R
state.problem_data['Tref'] = Tref

# Injector/mixing:
state.problem_data['s'] = s
state.problem_data['AR'] = AR
state.problem_data['L'] = L

# Kinetics and heat release:
state.problem_data['Da'] = Da
state.problem_data['Tign'] = Tign
state.problem_data['hv'] = hv
state.problem_data['Ea'] = Ea

# Initialize domains, including the aux cells:
init(state)
#initDetTube(state)

# Get coordinates of domain:
xc = state.grid.x.centers
# Initialize arrays. These will be added together to form complete area profile.
state.aux[0,:] = np.ones(xc.shape) #area - needs to be ones always
state.aux[1,:] = np.zeros(xc.shape) #tracker for heat release through time 

# Set up PyClaw Controller:
claw = pyclaw.Controller()
claw.tfinal = 100.0
claw.solution = pyclaw.Solution(state,domain)
claw.solver = solver
claw.num_output_times = 1000
claw.outdir = './_output'
claw.keep_copy = True
claw.write_aux_always = False

#q^T = [rho*A, rho*u*A, E*A, rho*Z]
#2 waves for HLL, 3 waves for HLLC
claw.solver.num_eqn = 4   
claw.solver.num_waves = 3 

solver.fwave = False 

# Run the simulation:
claw.run()


import pickle
import numpy as np
data = []
for i in range(1001):
    new_file_num = str(i)
    while len(new_file_num) < 4:
        new_file_num = "0" + new_file_num
    with open(f"/Users/davidzoro/RDE1D/_output/fort.q{new_file_num}", "rb") as f:
        for _ in range(5):
            next(f) 
        temp_data = (np.loadtxt(f))
    data.append(temp_data)
data = np.vstack(data)

rows = data.shape[0]

gaussian_noise = np.random.normal(0, 0.2, data.shape)
train_inputs = data[:, :] + gaussian_noise[:, :]
train_labels = data[:, :]     


# Model
class UNET (nn.Module):
    def __init__(self, in_channels):
        super(UNET, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels= in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=in_channels * 2,out_channels=  in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels * 4, out_channels= in_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose1d(in_channels=in_channels * 8,out_channels=  in_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels * 8, out_channels= in_channels * 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=in_channels * 16, out_channels= in_channels * 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=in_channels * 32, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
        self.conv8= nn.ConvTranspose1d(in_channels=in_channels * 64, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(in_channels=in_channels * 64, out_channels= in_channels * 128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(in_channels=in_channels * 128, out_channels= in_channels * 256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv1d(in_channels=in_channels * 256, out_channels= in_channels * 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_len, features) to (batch_size, features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, features)
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels, d_model, num_heads, num_encoder_layers, num_classes):
        super(Transformer, self).__init__()
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_encoder_layers)
        self.regressor = nn.Linear(2048, num_classes)
        self.the_UNET = UNET(in_channels)
    def forward(self, x):
        x = self.TransformerEncoder(x)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.the_UNET(x)
        x = self.regressor(x)
        return x

#Define hyperparameters
epochs = 500
loss_func = nn.MSELoss()
Model = Transformer(in_channels=4, d_model=4, num_heads=1, num_encoder_layers=2, num_classes=4)
learning_rate = 0.0001
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)
# Identifying tracked values

train_loss = []

# training loop
train_inputs = torch.from_numpy(train_inputs).float()
train_labels = torch.from_numpy(train_labels).float()

for i in range(epochs):
    optimizer.zero_grad()
    predictions = Model(train_inputs)
    loss = loss_func(predictions, train_labels)
    print(f"Epoch {i} Loss: {loss}")
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()


new_predictions = predictions.squeeze().numpy()


# Update q with predictions from the ML model

# Animation code
def animate_simulation():
    # Use len(claw.frames) directly in the loop
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    lines = []

    titles = ['Pressure (P)', 'Temperature (T)', 'Combustion Progress Variable (z)']
    ylabels = ['Pressure', 'Temperature', 'Progress Variable']

    for ax, title, ylabel in zip(axs, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        line, = ax.plot([], [], lw=2)
        lines.append(line)

    for ax in axs:
        ax.set_xlim(xc.min(), xc.max())
        ax.set_ylim(0, 1.5)

    def update(frame_idx):
        frame = claw.frames[frame_idx]
        q = frame.q[:, :4800] # Average over each 48 timesteps
        
        q[:, :] = (new_q[frame_idx]).T  # Ensure correct shape
        rho, u, P, T, z = getPrimitive(q, state)
        # Use rho and u to avoid "not accessed" warnings
        print(rho, u)  # Example usage, can be removed later
        
        lines[0].set_data(xc, P)
        lines[1].set_data(xc, T)
        lines[2].set_data(xc, z)
        return lines

    ani = FuncAnimation(fig, update, frames=100, blit=True)
    ani.save('simulation.mp4', writer='ffmpeg', fps=30)
    plt.show()

# Call the animation function
animate_simulation()
