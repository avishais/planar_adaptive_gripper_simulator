#!/usr/bin/env python

import math
import numpy as np
import time
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
from matrices_full import get_matrices_full
from matrices import get_matrices



class AdaptiveGripperEnv():

    # Gripper properties
    m_link = 100 # Links masss
    K = np.array([10, 20]) # Finger spring coefficients
    c = np.array([25, 25]) # Natural damping of the joints
    l_links = np.array([0.6, 0.4]) # Lengths of the two links of a finger
    a = np.array([-0.2+0.0305, 0.2-0.0305]) # First joints position on the base
    z = np.deg2rad([30, -5, -30, 5]) # Springs pre-loading angles of the joints

    # Object properties
    mo = 100 # Object mass
    Io = 10 # Object inertia
    ro = 0.1 # Object radius

    # Fail criteria
    u_min = 0.1
    u_max = 95
    f_max = 950.
    f_min = -1#5.
    actuator_angle_bound = np.array([0,1]) # Normalized


    R = np.array([[-1, 0], [-0.5, 0], [0, 1], [0, 0.5]])/10 # Maps tendon forces to joint torques
    Q = np.array([[2000, 0], [0, 2000]]) # Maps actuator angles to tendon forces
    A = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1], [0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]])*0.00002 # Set of possible_actions

    n = 14
    dt = 0.01
    x = None #np.zeros((n,1))
    tendon_forces = None # Current forces on the tendons
    thetas = None # Actuators angles
    t = 0
    done = False
    ee = None
    constraint = None

    EulerOrRK = 'Euler'#'RK'
    system_reset = False

    track_state = []
    track_T = []
    track_f = []

    def __init__(self):

        self.reset()

        self.viewer = None

    def reset(self):

        if not self.system_reset:
            # self.x = np.concatenate( (np.deg2rad(np.array([-0, -10, 0, 10])), np.array([0, 0.9939, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0 ) # [q1 q2 q1 q2 x y th dq1 dq2 dq1 dq2 dx dy dth]
            self.x = np.array([0.0448790923493863,	-0.288268315358574,	-0.0448790923493863,	0.288268315358574,	0,	0.987482396155299,	0,	0,	0,	0,	0,	0,	0,	0]) # [q1 q2 q1 q2 x y th dq1 dq2 dq1 dq2 dx dy dth]
            self.tendon_forces = np.array([200.0, 200.0])
            self.thetas = np.array([0.1,0.1])

            self.t = 0
            self.done = False
            self.system_reset = True

            self.track_state = []
            self.track_f = []
            self.track_T = []

            print('Gripper initialized.')

    def ODEfunc(self, t, x, dthetas):

        # Dt, Ct, Gt, Ft, Gmap, self.ee, Jee, dJG = get_matrices_full(x, self.l_links, self.m_link, self.mo, self.Io, self.ro, self.c, self.K, self.z, self.a) # Slow matrices computations
        Dt, Ct, Gt, Ft, Gmap, self.ee, Jee, dJG = get_matrices(x, self.l_links, self.m_link, self.mo, self.Io, self.ro, self.c, self.K, self.z, self.a)

        self.thetas += dthetas
        
        self.tendon_forces = self.Q.dot(self.thetas.reshape((2,1)))
        u = self.R.dot( self.tendon_forces )
        
        ddx = np.linalg.inv(Dt).dot( Ft.dot(u) - Ct.dot(x[self.n-3:self.n].reshape(3,1)) - Gt ) 

        dx = np.zeros((self.n,1))
        dx[:7] = x[7:].reshape((7,1))
        dx[7:11] = dJG.dot(x[self.n-3:self.n].reshape(3,1)) + np.linalg.inv(Jee).dot( np.transpose(Gmap).dot(ddx) )#
        dx[11:] = ddx

        # self.constraint = Jee.dot(dx[7:11].reshape(4,1)) - np.transpose(Gmap).dot(dx[11:].reshape(3,1))

        # print(self.thetas.reshape((-1,2)), self.tendon_forces.reshape(-1,2), u.reshape(-1,4))
        return dx
        

    def step(self, action=0):
        self.system_reset = False

        dthetas = self.A[action]

        for i in range(1): # Each step is n*dt time
            if self.EulerOrRK == 'Euler': # Euler method
                df = self.ODEfunc(self.t, self.x, dthetas)

                self.x = self.x.reshape((14,1)) + df * self.dt

            else: # Runga-Kutta method
                k1 = self.dt*self.ODEfunc(self.t, self.x, dthetas)
                k2 = self.dt*self.ODEfunc(self.t+self.dt/2, self.x.reshape((14,1))+k1/2, dthetas)
                k3 = self.dt*self.ODEfunc(self.t+self.dt/2, self.x.reshape((14,1))+k2/2, dthetas)
                k4 = self.dt*self.ODEfunc(self.t+self.dt, self.x.reshape((14,1))+k3, dthetas)

                self.x = self.x.reshape((self.n,1)) + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

            self.t += self.dt

        d = np.linalg.norm(self.ee[:,1]-self.ee[:,3]) # Distance between finger-tips

        if any( self.tendon_forces > self.f_max ) or any( self.tendon_forces < self.f_min ) or ( d > 2.25*self.ro ):# or any( np.abs(self.constraint) > 0.4 ):
            self.done = True

        self.track_state.append(self.x)
        self.track_T.append(self.t)
        self.track_f.append(self.tendon_forces)

        # print(self.t, self.thetas.reshape((-1,2)), self.observe().reshape(-1,5), d)
        return self.observe(), self.done

    def observe(self):

        return np.concatenate((self.x[4:7], self.tendon_forces), axis=0)

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
        if not self.done:
            self.cyltrans.set_translation(x[0]*scale+c[0], x[1]*scale+c[1])
        else:
            self.cyltrans.set_translation(x[0]*scale*100+c[0], x[1]*scale*100+c[1])
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

    def key_control(self):

        import curses
        screen = curses.initscr()
        curses.cbreak()
        screen.keypad(1)

        screen.addstr("Enter j to quit ")
        screen.refresh()

        k = ''

        while k != ord('j') or not self.done:
            try:
                k = screen.getch()

                if chr(k)=='w':
                    a = 0
                if chr(k)=='x':
                    a = 1
                if chr(k)=='a':
                    a = 2
                if chr(k)=='d':
                    a = 3
                if chr(k)=='s':
                    a = 8

                G.step(a)
                G.render()

                if self.done:
                    print "Torque bound breached"
                    1/0
                
                # screen.addch(k)
                screen.refresh()
            except ValueError:
                curses.endwin()
            
        curses.endwin()

    def plots(self):
        ax1 = plt.subplot(221)
        ax1.plot(np.array(self.track_T).reshape(-1,1),np.array(self.track_f).reshape(-1,2))    
        ax1.set(xlabel='Time', ylabel='Tendon forces')
        ax1.set_ylim([0,400])

        ax2 = plt.subplot(222)
        ax2.plot(np.array(self.track_T).reshape(-1,1),np.array(self.track_state)[:,4:6].reshape(-1,2))    
        ax2.set(xlabel='Time', ylabel='Object position')
        # ax2.set_ylim([0,1.2])

        ax3 = plt.subplot(223)
        ax3.plot(np.array(self.track_T).reshape(-1,1),np.array(self.track_state)[:,12:14].reshape(-1,2))    
        ax3.set(xlabel='Time', ylabel='Object velocity')
        # ax3.set_ylim([0,1.2])

        ax4 = plt.subplot(224)
        ax4.plot(np.array(self.track_state)[:,4],np.array(self.track_state)[:,5])    
        ax4.set(xlabel='x', ylabel='y')
        # ax3.set_ylim([0,1.2])

        plt.show()


if __name__ == '__main__':
    
    G = AdaptiveGripperEnv()

    G.reset()

    # G.key_control()

    st = time.time()
    while G.t < 20:
        if G.t < 10:
            action = 0#np.random.randint(8)
        else:
            if G.t < 20:
                action = 2
            else:
                action = 8

        
        _, done = G.step(action)
        # 
        G.render()

        if done:
            print('Torque bound breached at time %f.' % G.t)
            time.sleep(2)
            break
    
    print("step time: ", time.time()-st)
    G.close()

    G.plots()

    # print G.x.reshape(-1,14)
    

