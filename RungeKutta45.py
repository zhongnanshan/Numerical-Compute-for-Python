from scipy.integrate._ode import IntegratorBase
from numpy import array,arange,isfinite,ceil
from pylab import *
 
class RungeKutta45(IntegratorBase):
    runner = True
 
    def __init__(self,dt=.01):
        self.dt = dt
 
    def reset(self,n,has_jac):
        pass
 
    def Error(self,yn4,yn5,dt,f_params):
        atol = f_params[0]
        rtol = f_params[1]
        safe = f_params[2]
        y = max( max(abs(yn4)), max(abs(yn5)) )
        scale = atol + rtol * y
 
        tot = 0.
        for a in range(len(yn4)):
            tot = tot + abs( yn4[a] - yn5[a] ) / scale
        err = sqrt( ( 1./ len(yn4) ) * tot )
 
 
        dt = dt*safe*(1./err)**(1./5.)
 
        if err <= 1.:
            e = True
        else:
            e = False
 
        return e, dt
 
    def run(self,f,jac,y0,t0,t1,f_params,jac_params):
 
        yo = array(y0) # Initial condition
        t = t0
        dt = self.dt
 
        while t < t1:
 
            if t + dt > t1:
 
                dt = t1 - t
 
                F1 = f(t, yo, *f_params) * dt
                F2 = f(t +dt/5., yo + F1/5., *f_params) * dt
                F3 = f(t + (3./10.)*dt, yo + (3./40.)*F1 + (9./40.)*F2, *f_params) * dt
                F4 = f(t + (4./5.)*dt, yo + (44./45.)*F1 + (-56./15.)*F2 + (32./9.)*F3, *f_params) * dt
                F5 = f(t + (8./9.)*dt, yo + (19372./6561.)*F1 + (-25360./2187.)*F2 + (64448./6561.)*F3 + (-212./729.)*F4, *f_params) * dt
                F6 = f(t + dt, yo + (9017./3168.)*F1 + (-355./33.)*F2 + (46732./5247.)*F3 + (49./176.)*F4 + (-5103./18656.)*F5, *f_params) * dt
                yn = yo + (35./384.)*F1 + (500./1113.)*F3 + (125./192.)*F4 + (-2187./6784.)*F5 + (11./84.)*F6
                F7 = f(t + dt, yn) * dt
                yn5 = yo + (5179./57600.)*F1 + (7571./16695.)*F3 + (393./640.)*F4 + (92097./339200)*F5 + (187./2100.)*F6 + (1./40.)*F7
 
                t += dt
 
            else: 
                e = False
                while e == False:
 
                    F1 = f(t, yo, *f_params) * dt
                    F2 = f(t +dt/5., yo + F1/5., *f_params) * dt
                    F3 = f(t + (3./10.)*dt, yo + (3./40.)*F1 + (9./40.)*F2, *f_params) * dt
                    F4 = f(t + (4./5.)*dt, yo + (44./45.)*F1 + (-56./15.)*F2 + (32./9.)*F3, *f_params) * dt
                    F5 = f(t + (8./9.)*dt, yo + (19372./6561.)*F1 + (-25360./2187.)*F2 + (64448./6561.)*F3 + (-212./729.)*F4, *f_params) * dt
                    F6 = f(t + dt, yo + (9017./3168.)*F1 + (-355./33.)*F2 + (46732./5247.)*F3 + (49./176.)*F4 + (-5103./18656.)*F5, *f_params) * dt
                    yn = yo + (35./384.)*F1 + (500./1113.)*F3 + (125./192.)*F4 + (-2187./6784.)*F5 + (11./84.)*F6
                    F7 = f(t + dt, yn) * dt
                    yn5 = yo + (5179./57600.)*F1 + (7571./16695.)*F3 + (393./640.)*F4 + (92097./339200)*F5 + (187./2100.)*F6 + (1./40.)*F7
 
                    e, dt = self.Error(yn,yn5,dt,f_params)
 
                t += dt
                yo = yn.copy()
 
        if isfinite(yn[-1]): self.success = True # Check for success
        return yn,t
 
 
if RungeKutta45.runner:
    IntegratorBase.integrator_classes.append(RungeKutta45)
