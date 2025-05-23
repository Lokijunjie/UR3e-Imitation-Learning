'''
    Dynamic Movement Primitive Class
    Author: Michail Theofanidis, Joe Cloud, James Brady
'''

import numpy as np
from .utils import psi
import matplotlib.pyplot as plt

class DynamicMovementPrimitive:

    """Create an DMP.
        
        Keyword arguments:
        a -- Gain a of the transformation system
        b -- Gain b of the transformation system
        as_deg -- Degradation of the canonical system
        ng -- Number of Gaussian
        stb -- Stabilization term
    """

    def __init__(self, _a, _ng, _stb):
       
        self.a = _a
        self.b = _a/4
        self.as_deg = _a/3
        self.ng = _ng
        self.stb = _stb

    # Create the phase of the system using the time vector
    def phase(self, time):
        return np.exp((-self.as_deg) * np.linspace(0, 1.0, len(time))).T

    # Generate a gaussian distribution
    def distributions(self, s, h=1):

        # Find the centers of the Gaussian in the s domain
        c = np.linspace(min(s), max(s), self.ng)
        d = c[1] - c[0]
        c /= d
        # Calculate every gaussian
        psv = np.array([[psi(h, _c, _s/d) for _s in s] for _c in c]) 

        return psv

    # Imitation Learning
    def imitate(self, x, dx, ddx, time, s, psv):

        # Initialize variables
        sigma, f_target = np.zeros((2,len(time)))
        g = x[-1]
        x0 = x[0]
        tau = time[-1]

        # Compute ftarget
        for i in range(0, len(time)):

            # Add stabilization term
            if self.stb:
                mod = self.b*(g - x0)*s[i]
                sigma[i] = (g - x0)*s[i]
            else:
                mod = 0
                sigma[i] = s[i]

            # Check again in the future
            f_target[i] = np.power(tau, 2)*ddx[i] - self.a*(self.b*(g - x[i]) - tau*dx[i]) + mod

        # Regression
        w = [sigma.T.dot(np.diag(p)).dot(f_target)/(sigma.T.dot(np.diag(p)).dot(sigma)) for p in psv]

        return f_target, np.array(w)

    # Generate a trajectory
    def generate(self, w, x0, g, time, s, psv):

        # Initialize variables
        ddx, dx, x, sigma, f_rep = np.zeros((5, len(time)))
        tau = time[-1]
        dx_r = 0
        x_r = x0

        for i in range(len(time)):

            p_sum = 0
            p_div = 0

            if i == 0:
                dt = time[i]
            else:
                dt = time[i] - time[i - 1]

            # Add stabilization term
            if self.stb:
                mod = self.b*(g - x0)*s[i]
                sigma[i] = (g - x0)*s[i]
            else:
                mod = 0
                sigma[i] = s[i]

            for j in range(self.ng):

                p_sum += psv[j][i]*w[j]
                p_div += psv[j][i]

            # Calculate the new control input
            f_rep[i] = p_sum/p_div*sigma[i]

            # Calculate the new trajectory
            ddx_r = (self.a*(self.b*(g - x_r) - tau*dx_r) + f_rep[i] + mod)/np.power(tau, 2)
            dx_r += ddx_r*dt
            x_r += dx_r*dt

            ddx[i], dx[i], x[i] = ddx_r, dx_r, x_r

        return ddx, dx, x

    # Adaptation using reinforcement learning
    def adapt(self, w, x0, g, t, s, psv, samples, rate):

        print('Trajectory adapted')

        # Initialize the action variables
        a = w
        tau = t[-1]

        # Flag which acts as a stop condition
        met_threshold = False
        counter = 0
        gain = []

        while not met_threshold:
            exploration = np.array([[np.random.normal(0, np.std(psv[j]*a[j]))
                    for j in range(self.ng)] for i in range(samples)])

            actions = np.array([a + e for e in exploration])

            # Generate new rollouts
            ddx, dx, x = np.transpose([self.generate(act, x0, g, t, s, psv) for act in actions], (1, 2, 0))

            # Estimate the Q values
            Q = [sum([self.reward(g, x[j, i], t[j], tau) for j in range(len(t))]) for i in range(samples)]

            # Sample the highest Q values to adapt the action parameters
            sort_Q = np.argsort(Q)[::-1][:np.floor(samples*rate).astype(int)]

            # Update the action parameter
            sumQ_y = sum([Q[i] for i in sort_Q])
            sumQ_x = sum([exploration[i]*Q[i] for i in sort_Q])

            # Update the policy parameters
            a += sumQ_x/sumQ_y

            gain.append(Q[sort_Q[0]])

            # Stopping condition
            if np.abs(x[-1, sort_Q[0]] - g) < 0.01:
                met_threshold = True

        return ddx[:, sort_Q[0]], dx[:, sort_Q[0]], x[:, sort_Q[0]], actions[sort_Q[0]], np.cumsum(gain)

    # Reward function
    def reward(self, goal, position, time, tau, w=0.9, threshold=0.01):

        dist = goal - position

        if np.abs(time - tau) < threshold:
            rwd = w*np.exp(-np.sqrt(dist*dist.T))
        else:
            rwd = (1-w) * np.exp(-np.sqrt(dist*dist.T))/tau

        return rwd





# import numpy as np
# from pyrdmp.utils import psi
# import matplotlib.pyplot as plt

# class DynamicMovementPrimitive:

#     """Create a DMP.
        
#         Keyword arguments:
#         a -- Gain a of the transformation system
#         b -- Gain b of the transformation system
#         as_deg -- Degradation of the canonical system
#         ng -- Number of Gaussian
#         stb -- Stabilization term
#     """

#     def __init__(self, _a, _ng, _stb):
#         self.a = _a
#         self.b = _a / 4
#         self.as_deg = _a / 3
#         self.ng = _ng
#         self.stb = _stb

#     # Create the phase of the system using the time vector
#     def phase(self, time):
#         return np.exp((-self.as_deg) * np.linspace(0, 1.0, len(time))).T

#     # Generate a gaussian distribution
#     def distributions(self, s, h=1):
#         # Find the centers of the Gaussian in the s domain
#         c = np.linspace(min(s), max(s), self.ng)
#         d = c[1] - c[0]
#         c /= d
#         # Calculate every gaussian
#         psv = np.array([[psi(h, _c, _s/d) for _s in s] for _c in c])
#         return psv

#     # Imitation Learning
#     def imitate(self, x, dx, ddx, time, s, psv):
#         sigma, f_target = np.zeros((2, len(time)))
#         g = x[-1]
#         x0 = x[0]
#         tau = time[-1]

#         # Compute f_target
#         for i in range(0, len(time)):
#             if self.stb:
#                 mod = self.b * (g - x0) * s[i]
#                 sigma[i] = (g - x0) * s[i]
#             else:
#                 mod = 0
#                 sigma[i] = s[i]

#             f_target[i] = np.power(tau, 2) * ddx[i] - self.a * (self.b * (g - x[i]) - tau * dx[i]) + mod

#         # Regression
#         w = [sigma.T.dot(np.diag(p)).dot(f_target) / (sigma.T.dot(np.diag(p)).dot(sigma)) for p in psv]
#         return f_target, np.array(w)

#     # Generate a trajectory
#     def generate(self, w, x0, g, time, s, psv):
#         ddx, dx, x, sigma, f_rep = np.zeros((5, len(time)))
#         tau = time[-1]
#         dx_r = 0
#         x_r = x0

#         for i in range(len(time)):
#             p_sum = 0
#             p_div = 0

#             if i == 0:
#                 dt = time[i]
#             else:
#                 dt = time[i] - time[i - 1]

#             if self.stb:
#                 mod = self.b * (g - x0) * s[i]
#                 sigma[i] = (g - x0) * s[i]
#             else:
#                 mod = 0
#                 sigma[i] = s[i]

#             for j in range(self.ng):
#                 p_sum += psv[j][i] * w[j]
#                 p_div += psv[j][i]

#             f_rep[i] = p_sum / p_div * sigma[i]

#             ddx_r = (self.a * (self.b * (g - x_r) - tau * dx_r) + f_rep[i] + mod) / np.power(tau, 2)
#             dx_r += ddx_r * dt
#             x_r += dx_r * dt

#             ddx[i], dx[i], x[i] = ddx_r, dx_r, x_r

#         return ddx, dx, x

#     # Adaptation using Path Integral Policy Improvement
#     def path_integral_update(self, w, x0, g, t, s, psv, num_samples=100, temperature=0.1):
#         # Initialize exploration noise
#         exploration_noise = np.random.normal(0, 0.1, size=(num_samples, self.ng))

#         # Generate actions (w) for each sample by adding exploration noise
#         actions = np.array([w + noise for noise in exploration_noise])

#         # Generate rollouts (trajectories) for each action
#         ddx, dx, x = np.transpose([self.generate(act, x0, g, t, s, psv) for act in actions], (1, 2, 0))

#         # Compute rewards for each trajectory
#         Q_values = np.array([self.reward(g, x[j, -1], t[j], t[-1]) for j in range(num_samples)])

#         # Path integral: calculate the softmax of rewards
#         exp_rewards = np.exp(Q_values / temperature)
#         norm_rewards = exp_rewards / np.sum(exp_rewards)

#         # Update action weights (policy parameters)
#         action_update = np.dot(norm_rewards, actions)

#         return action_update, ddx, dx, x, norm_rewards

#     # Reward function
#     def reward(self, goal, position, time, tau, w=0.9, threshold=0.01):
#         dist = goal - position
#         reward_value = 0

#         if np.abs(time - tau) < threshold:
#             reward_value = w * np.exp(-np.sqrt(dist * dist.T))
#         else:
#             reward_value = (1 - w) * np.exp(-np.sqrt(dist * dist.T)) / tau

#         return reward_value

#     # Adaptation using path integral and reward maximization
#     def adapt_with_path_integral(self, w, x0, g, t, s, psv, num_samples=100, temperature=0.1, rate=0.1):
#         print('Trajectory adapted with Path Integral')

#         a = w
#         tau = t[-1]

#         met_threshold = False
#         gain = []

#         while not met_threshold:
#             # Perform path integral update to explore new actions
#             action_update, ddx, dx, x, norm_rewards = self.path_integral_update(a, x0, g, t, s, psv, num_samples, temperature)

#             # Evaluate the new trajectory and rewards
#             Q_values = [self.reward(g, x[j, -1], t[j], tau) for j in range(num_samples)]
#             gain.append(np.max(Q_values))

#             # Update the action parameters using path integral update
#             a = action_update

#             if np.abs(x[-1, np.argmax(Q_values)] - g) < 0.01:
#                 met_threshold = True

#         return ddx[:, np.argmax(Q_values)], dx[:, np.argmax(Q_values)], x[:, np.argmax(Q_values)], a, np.cumsum(gain)


# # Example usage of the DynamicMovementPrimitive class
# if __name__ == '__main__':
#     # Define initial parameters for DMP
#     a = 35  # Gain of the transformation system
#     ng = 50  # Number of Gaussian basis functions
#     stb = True  # Stabilization term

#     # Instantiate the DMP class
#     dmp = DynamicMovementPrimitive(a, ng, stb)

#     # Generate time vector and phase space
#     time = np.linspace(0, 1, 1000)
#     s = dmp.phase(time)

#     # Define the goal and initial conditions
#     x0 = 0  # Initial position
#     g = 1  # Goal position

#     # Generate a Gaussian distribution
#     psv = dmp.distributions(s)

#     # Initial weights for DMP
#     w_init = np.zeros(ng)

#     # Adapt the trajectory using path integral policy improvement
#     ddx, dx, x, a, gain = dmp.adapt_with_path_integral(w_init, x0, g, time, s, psv, num_samples=100, temperature=0.1)

#     # Plot the resulting trajectory
#     plt.plot(time, x, label="Generated Trajectory")
#     plt.axhline(y=g, color='r', linestyle='--', label="Goal")
#     plt.xlabel("Time")
#     plt.ylabel("Position")
#     plt.legend()
#     plt.show()
