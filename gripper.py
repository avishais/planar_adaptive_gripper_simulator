#!/usr/bin/env python

import math
import numpy as np
from matrices import get_matrices
import time
from gym.envs.classic_control import rendering


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
    x = None #np.zeros((n,1))
    t = 0

    ee = None

    EulerOrRK = 1

    def __init__(self):

        self.reset()

        self.viewer = None

    def ODEfunc(self, t, x):

        Dt, Ct, Gt, Ft, Gmap, self.ee, Jee, dJG = get_matrices(x, self.l_links, self.m_link, self.mo, self.Io, self.ro, self.c, self.K, self.z, self.a)

        f = np.array([100, 150]).reshape(2,1)
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
        

    def step(self, action=0):

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

    def render(self):
        screen_width = 600
        screen_height = 600

        c = [screen_width/2.0, 100]
        scale = 300
        jr = 7.0
        linkwidth = 5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # Cyl
            Th = np.linspace(0., 2.*np.pi, num = 60)
            P = []
            for th in Th:
                P.append((scale*self.ro*np.cos(th), scale*self.ro*np.sin(th)))
            cyl = rendering.FilledPolygon(P)
            cyl.set_color(1,0,0)
            self.cyltrans = rendering.Transform()
            cyl.add_attr(self.cyltrans)
            self.viewer.add_geom(cyl)

            # Orientation mark
            patch = rendering.FilledPolygon([(0,0), (scale*self.ro*np.cos(0.3+np.pi/2),scale*self.ro*np.sin(0.3+np.pi/2)), (scale*self.ro*np.cos(-0.3+np.pi/2),scale*self.ro*np.sin(-0.3+np.pi/2))])
            patch.set_color(0,0,1)
            self.patchtrans = rendering.Transform()
            patch.add_attr(self.patchtrans)
            self.viewer.add_geom(patch)
            patch.add_attr(self.cyltrans)

            # Joints
            j0l = rendering.make_circle(jr/2)
            j0l.set_color(0,0,0)
            self.j0ltrans = rendering.Transform()
            self.viewer.add_geom(j0l)
            j0l.add_attr(self.j0ltrans)
            j0r = rendering.make_circle(jr/2)
            j0r.set_color(0,0,0)
            self.j0rtrans = rendering.Transform()
            self.viewer.add_geom(j0r)
            j0r.add_attr(self.j0rtrans)
            
            j1 = rendering.make_circle(jr/2)
            j1.set_color(0,0,0)
            self.j1trans = rendering.Transform()
            self.viewer.add_geom(j1)
            j1.add_attr(self.j1trans)
            j2 = rendering.make_circle(jr/2)
            j2.set_color(0,0,0)
            self.j2trans = rendering.Transform()
            self.viewer.add_geom(j2)
            j2.add_attr(self.j2trans)
            j3 = rendering.make_circle(jr/2)
            j3.set_color(0,0,0)
            self.j3trans = rendering.Transform()
            self.viewer.add_geom(j3)
            j3.add_attr(self.j3trans)
            j4 = rendering.make_circle(jr/2)
            j4.set_color(0,0,0)
            self.j4trans = rendering.Transform()
            self.viewer.add_geom(j4)
            j4.add_attr(self.j4trans)

            # Base
            l0 = rendering.Line((scale*self.a[0]+c[0],c[1]), (scale*self.a[1]+c[0],c[1]))
            l0.set_color(0,0,0)
            self.viewer.add_geom(l0)
            h0 = rendering.Line((c[0],c[1]), (c[0],c[1]-100))
            h0.set_color(0,0,0)
            self.viewer.add_geom(h0)

            # Links
            l1 = rendering.Line((0,0), (0,self.l_links[0]*scale))
            l1.set_color(0,0,0)
            self.l1trans = rendering.Transform(translation=(self.a[0]*scale+c[0], c[1]))
            self.viewer.add_geom(l1)
            l1.add_attr(self.l1trans)
            l2 = rendering.Line((0,0), (0,self.l_links[1]*scale))
            l2.set_color(0,0,0)
            self.l2trans = rendering.Transform(translation=(0, self.l_links[0]*scale))
            self.viewer.add_geom(l2)
            l2.add_attr(self.l2trans)
            l2.add_attr(self.l1trans)
            l3 = rendering.Line((0,0), (0,self.l_links[0]*scale))
            l3.set_color(0,0,0)
            self.l3trans = rendering.Transform(translation=(self.a[1]*scale+c[0], c[1]))
            self.viewer.add_geom(l3)
            l3.add_attr(self.l3trans)
            l4 = rendering.Line((0,0), (0,self.l_links[1]*scale))
            l4.set_color(0,0,0)
            self.l4trans = rendering.Transform(translation=(0, self.l_links[0]*scale))
            self.viewer.add_geom(l4)
            l4.add_attr(self.l4trans)
            l4.add_attr(self.l3trans)

            # axleoffset =cartheight/4.0
            # pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # pole.set_color(.8,.6,.4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(carttrans)
            # self.axle.set_color(.5,.5,.8)
            # self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.x is None: return None

        
        x = self.observe()
        self.cyltrans.set_translation(x[0]*scale+c[0], x[1]*scale+c[1])
        self.patchtrans.set_translation(0,0)
        self.patchtrans.set_rotation(x[2])
        self.j0ltrans.set_translation(self.a[0]*scale+c[0],c[1])
        self.j0rtrans.set_translation(self.a[1]*scale+c[0],c[1])
        self.j1trans.set_translation(self.ee[0][0]*scale+c[0],self.ee[1][0]*scale+c[1])
        self.j2trans.set_translation(self.ee[0][1]*scale+c[0],self.ee[1][1]*scale+c[1])
        self.j3trans.set_translation(self.ee[0][2]*scale+c[0],self.ee[1][2]*scale+c[1])
        self.j4trans.set_translation(self.ee[0][3]*scale+c[0],self.ee[1][3]*scale+c[1])

        self.l1trans.set_rotation(self.x[0])
        self.l2trans.set_rotation(self.x[1])
        self.l3trans.set_rotation(self.x[2])
        self.l4trans.set_rotation(self.x[3])

        
        
        return self.viewer.render(return_rgb_array = 'human'=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


if __name__ == '__main__':
    
    G = AdaptiveGripperEnv()

    G.reset()
        
    for i in range(1000):
        G.step()
        G.render()
        

    G.close()