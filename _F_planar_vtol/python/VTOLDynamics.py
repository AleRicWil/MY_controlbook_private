import numpy as np 
import VTOLParam as P

class VTOLDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],         # z initial position
            [P.h0],         # initial height
            [P.theta0],     # Theta initial orientation
            [P.zdot0],      # zdot initial velocity
            [P.hdot0],      # initial climb rate
            [P.thetadot0],  # Thetadot initial velocity
        ])
        # simulation time step
        self.Ts = P.Ts
        # Mass of the craft, kg
        self.mc = P.mc * (1.+alpha*(2.*np.random.rand()-1.))
        # Mass of the motors, kg
        self.mr = P.mr * (1.+alpha*(2.*np.random.rand()-1.))
        # Rotational intertia of craft, 
        self.Jc = P.Jc * (1.+alpha*(2.*np.random.rand()-1))
        # Length of the rod, m
        self.d = P.d * (1.+alpha*(2.*np.random.rand()-1.))
        # Wind damping, 
        self.mu = P.mu * (1.+alpha*(2.*np.random.rand()-1.))
        # gravity constant is well known, don't change.
        self.g = P.g
        # Wind disturbance, N
        self.F_wind = P.F_wind
        self.force_limit = P.F_max

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        u = saturate(u, self.force_limit)
        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state, u):
        # Return xdot = f(x,u)
        z = state[0][0]
        h = state[1][0]
        theta = state[2][0]
        zdot = state[3][0]
        hdot = state[4][0]
        thetadot = state[5][0]
        F_r = u[0][0]
        F_L = u[1][0]
        # The equations of motion.
        # print("here")
        # print(u.item(0))
        # exit()
        #LHS
        M = np.array([[self.mc + 2*self.mr, 0, 0], 
                      [0, self.mc + 2*self.mr, 0],
                      [0, 0, self.Jc + 2*self.d**2*self.mr]])
        #RHS
        C = np.array([[-self.mu*zdot - (F_r+F_L)*np.sin(theta)],
                      [(F_r+F_L)*np.cos(theta) - (self.mc + 2*self.mr)*self.g],
                      [(F_r-F_L)*self.d]])
        # print(M.shape,C.shape) 
        accels = np.linalg.inv(M) @ C
        zddot = accels[0][0]
        hddot = accels[1][0]
        thetaddot = accels[2][0]
        # build xdot and return
        xdot = np.array([[zdot], [hdot], [thetadot], [zddot], [hddot], [thetaddot]])
        return xdot

    def h(self):
        # return y = h(x)
        z = self.state[0][0]
        h = self.state[1][0]
        theta = self.state[2][0]
        y = np.array([[z], [h], [theta]])
        return y

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state = self.state + self.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)

        
def saturate(u, limit):
    if abs(u[0][0]) > limit:
        u[0][0] = limit*np.sign(u[0][0])
    if abs(u[1][0]) > limit:
        u[1][0] = limit*np.sign(u[1][0])
    return u
