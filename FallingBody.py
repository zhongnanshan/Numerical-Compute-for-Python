from math import *
from pylab import *
from scipy.integrate._ode import *
from Euler import *
from EulerCromer import *
from EulerRichardson import *
from RungeKutta4 import *
from Predictor_corrector import *
from numpy import arange,vstack,array
from pylab import plot,clf,show,xlabel,ylabel,title,grid
 
# Function defining derivates as a function of the positions and velocities.
# This is the INTERACTION, the inputs to this function are OBSERVABLES.
def sho(t,y,g,m):
# INPUTS:
# t time, only used in non-autonomous systems.
# y[0] the x-position
# y[1] the velocity
# m Mass
# g Gravity
# OUTPUTS:
# dxdt derivative of the position = velocity
# dvdt acceleration, Newton's second law, here for a spring a = -mg/m
    return array([y[1], -g])
 
# Initial Conditions
y0, t0 = [10.,0.], 0.
 
# Model parameters these and the observables define the COMPONENT
m = 1      # Mass
g = 9.8    # gravity
 
# CREATE ODE OBJECT
i = ode(sho)                      # Create an ode object and bind the interaction.
i.set_integrator('Euler',dt=.1)   # Which integrator to use. Set the time step.
i.set_initial_value(y0,t0)        # The initial values
i.set_f_params(g, m)              # Define parameters for the derivatives function.
 
tf = 1.6                 # Final time
dt = .01                    # Output interval
yf= [array(y0)]  
yec= [array(y0)]           # List of arrays for storing the output, use arrays because integrate returns them
yer = [array(y0)]
yrk = [array(y0)]
ypc = [array(y0)]
 
time = arange(t0,tf,dt)    # Times to evaluate a solution. 
 
# Main loop for the integration
for t in time:
    i.integrate(i.t+dt)
    yf.append(i.y)
 
#Euler Cromer
i.set_integrator('EulerCromer',dt=.1)  # Which integrator to use. Set the time step.
i.set_initial_value(y0,t0)            # The initial values
i.set_f_params(g, m)                     # Define parameters for the derivatives function.
 
for t in time:
    i.integrate(i.t+dt)
    yec.append(i.y)
 
#Euler Richardson
i.set_integrator('EulerRichardson',dt=.1)  # Which integrator to use. Set the time step.
i.set_initial_value(y0,t0)            # The initial values
i.set_f_params(g, m)                     # Define parameters for the derivatives function.
 
for t in time:
    i.integrate(i.t+dt)
    yer.append(i.y)
 
#Runge Kutta
i.set_integrator('RungeKutta4',dt=.1)  # Which integrator to use. Set the time step.
i.set_initial_value(y0,t0)            # The initial values
i.set_f_params(g, m)                     # Define parameters for the derivatives function.
 
for t in time:
    i.integrate(i.t+dt)
    yrk.append(i.y)
 
#Predictor Corrector
i.set_integrator('Predictor_corrector',dt=.1)  # Which integrator to use. Set the time step.
i.set_initial_value(y0,t0)            # The initial values
i.set_f_params(g, m)                     # Define parameters for the derivatives function.
 
for t in time:
    i.integrate(i.t+dt)
    ypc.append(i.y)
# Convert return list to array:
yf = array(yf)
yec = array(yec)
yrk = array(yrk)
yer = array(yer)
ypc = array(ypc)
 
#Analytic
ayv2 = y0[0] - 0.5 * g * time**2
 
# Plot the results
clf()
 
plot(time,yf[1:,0], label='Euler')   # The 0 column is position, 1 is velocity
plot(time,yec[1:,0], label='Euler Cromer')
plot(time,yer[1:,0], label= 'Euler Richardson')
plot(time,yrk[1:,0], label='Runge Kutta')
plot(time,ypc[1:,0], label='Predictor-Corrector')
plot(time, ayv2, 'b--', label='Analytical')
 
legend()
xlabel('Time (s)')
ylabel('Position (m)')
title('Falling Body')
#legend(('Euler','Euler Cromer', 'Euler Richardson', 'Runge Kutta', 'Predictor Corrector', 'Analytical'), loc='lower left')
grid()
show()