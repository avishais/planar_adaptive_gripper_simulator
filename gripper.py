#!/usr/bin/env python

import math
import numpy as np
from matrices import get_matrices


class AdaptiveGripperEnv():

    # Gripper properties
    m_link = 1 # Links masss
    K = np.array([1, 2]) # Finger spring coefficients
    c = np.array([20, 20]) # Natural damping of the joints
    l_links = np.array([0.6, 0.4]) # Lengths of the two links of a finger
    a = np.array([-0.2+0.0305, 0.2-0.0305]) # First joints position on the base
    z = np.deg2rad([30, -5, -30, 5]) # Springs pre-loading angles of the joints

    # Fail criteria
    u_min = 0.1
    f_max = 18

    # Object properties
    mo = 1 # Object mass
    Io = 1 # Object inertia
    ro = 0.1 # Object radius

    R = np.array([[-1, 0], [-0.5, 0], [0, 1], [0, 0.5]])/10 # Maps tendon forces to joint torques
    Q = np.array([[f_max*1.11115, 0], [0, f_max*1.11115]]) # Maps actuator angles to tendon forces

    n = 14
    dt = 0.01
    x = np.zeros((n,1))
    t = 0

    EulerOrRK = 1

    def __init__(self):

        self.reset()

    def ODEfunc(self, t, x):

        Dt, Ct, Gt, Ft, Gmap, _, Jee, dJG = get_matrices(x, self.l_links, self.m_link, self.mo, self.Io, self.ro, self.c, self.K, self.z, self.a)

        f = np.array([0, 0]).reshape(2,1)
        u = self.R.dot( f )

        ddx = np.linalg.inv(Dt).dot( Ft.dot(u) - Ct.dot(x[self.n-3:self.n].reshape(3,1)) - Gt ) 

        dx = np.zeros((self.n,1))
        dx[:7] = x[7:].reshape((7,1))
        dx[7:11] = dJG.dot(x[self.n-3:self.n].reshape(3,1)) + np.linalg.inv(Jee).dot( np.transpose(Gmap).dot(ddx) )#
        dx[11:] = ddx

        return dx

    def reset(self):
        self.x = np.concatenate( (np.deg2rad(np.array([-0, -10, 0, 10])), np.array([0, 0.9939, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0 ) # [q1 q2 q1 q2 dq1 dq2 dq1 dq2 x y th dx dy dth]
        self.t = 0

    def step(self):

        if not self.EulerOrRK: # Euler method
            df = self.ODEfunc(self.t, self.x)

            self.x = self.x.reshape((14,1)) + df * self.dt

        else: # Runga-Kutta method
            k1 = self.dt*self.ODEfunc(self.t, self.x)
            k2 = self.dt*self.ODEfunc(self.t+self.dt/2, self.x.reshape((14,1))+k1/2)
            k3 = self.dt*self.ODEfunc(self.t+self.dt/2, self.x.reshape((14,1))+k2/2)
            k4 = self.dt*self.ODEfunc(self.t+self.dt, self.x.reshape((14,1))+k3)

            self.x = self.x.reshape((14,1)) + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

        self.t += self.dt

        return self.observe()

    def observe(self):

        return self.x[4:7]


if __name__ == '__main__':
    
    G = AdaptiveGripperEnv()

    G.reset()
    for i in range(100):
        print G.step().reshape((1,3))