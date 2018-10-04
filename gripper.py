#!/usr/bin/env python

import math
import numpy as np
from matrices import get_matrices


class AdaptiveGripperEnv():

    # Gripper properties
    m_link = 1 # Links masss
    K = np.array([1, 2]) # Finger spring coefficients
    c = np.array([2, 20]) # Natural damping of the joints
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

    def __init__(self):

        x = np.concatenate( (np.deg2rad(np.array([-0, -10, 0, 10])), np.array([0, 0.9939, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0 ) # [q1 q2 q1 q2 dq1 dq2 dq1 dq2 x y th dx dy dth]
        self.GetMatrices(x)

    def GetMatrices(self, x):

        Dt, Ct, Gt, Ft, Gmap, ee, Jee = get_matrices(x, self.l_links, self.m_link, self.mo, self.Io, self.ro, self.c, self.K, self.z, self.a)

        print(Ct)
        





if __name__ == '__main__':
    
    # try:
    AdaptiveGripperEnv()
    # except:
    #     pass