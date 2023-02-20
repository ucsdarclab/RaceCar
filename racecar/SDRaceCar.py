# -*- coding: utf-8 -*-
"""
This is an Car simulation based off a paper from Chris Gerdes.
@author: michyip
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Force pyplot to be interactive
from racecar.util import line_search_tool, wrap_angle


class SDRaceCar():
    # This is an car example that exposes OpenAI Gym-like commands
    def __init__(self, render_env, track):
        # simuation parameters
        self.dt = 0.05  # seconds

        # Car model
        self.m = 1724  # mass [kg]
        self.I_z = 1300  # moment of inertia [kg/m^2]
        self.l_f = 1.35  # distance from front wheel to CoG [m]
        self.l_r = 1.15  # distance from rear wheel to CoG [m]

        #Fiala Model
        self.C_alpha = np.array([57.5,
                                 92.5])  # tire cornering stiffness [N/rad]
        self.mu = np.array([0.56, 0.5])  # friction between tire and ground
        self.max_steer_flag = False  # flag for max_steer_flaging
        self.thrust_max = 20  # max thrust
        self.thrust_min = 0  # min thrust
        self.steer_max = 1.047  # max steering angle
        self.steer_min = -1.047  # min steering angle
        self.F_z = self.m * 9.81 / 1000  # kN

        self.Horizon = 100

        #parameters from optimization that remain constant
        self.alpha_slip = np.arctan2(3 * self.mu[0] * self.F_z,
                                     self.C_alpha)  #Slip parameter
        self.slip_coeff = [
            -1000 * self.C_alpha, 1000 * self.C_alpha**2 *
            (2 - self.mu[1] / self.mu[0]) / (3 * self.mu * self.F_z),
            -1000 * self.C_alpha**3 * (3 - 2 * self.mu[1] / self.mu[0]) /
            (27 * self.mu**2 * (self.F_z)**2)
        ]
        self.no_slip_coeff = -1000 * self.mu[1] * self.F_z

        #track model
        self.track_len = 5000
        if track == "FigureEight":
            t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, self.track_len)
            self.track = 20 * np.vstack(
                [np.cos(t), np.multiply(np.sin(t), np.cos(t))])
            self.track_boundaries = [-40, 40, -20, 20]
            self.maxStep = 500
        elif track == "Linear":
            self.track = np.vstack([
                10 * np.linspace(0, 100, self.track_len),
                0 * np.ones(self.track_len)
            ])
            self.track_boundaries = [-5, 35, -20, 20]
            self.maxStep = 50
        else:  # default to circle track
            t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, self.track_len)
            self.track = 10 * np.vstack([np.cos(t), np.sin(t) + 1])
            self.track_boundaries = [-30, 30, -20, 40]
            self.maxStep = 350
        self.distance_threshold = 4
        self.stepCount = 0
        self.closest_track_ind = 0  # index to pt on track closest to current position

        # Render settings
        self.render_env = render_env

        # Initalize the environment
        self.reset()

        # Set I/O lengths
        self.n_actions = 2
        self.n_internal_states = 8
        self.n_observations = 12

    def _set_internal_state(self, values):
        # Sets an arbitrary state of the model
        # Generally used as an internal function
        # size can be somewhat arbitrary (since number of internal states are not exposed)

        self.x = values[0]  # self.x pos (in inertial frame)
        self.y = values[1]  # self.y pos (in inertial frame)
        self.psi = values[2]  # yaw
        self.v_x = values[3]  # velocity longitudinal
        self.v_y = values[4]  # velocity lateral
        self.omega = values[5]  # yaw rate
        self.steer = values[6]  # steering angle
        self.thrust = values[7]  # thrust

        #set closest track point
        self.closest_track_pt_x, self.closest_track_pt_y, self.closest_track_ind = line_search_tool(
            self.track, self.x, self.y, 0)

        return self.get_observation()

    def reset(self):
        self.stepCount = 0
        reset_values = np.array([-1.0, 0, 0, 0, 0, 0, 0, 0])
        return self._set_internal_state(reset_values)

    def get_observation(self):
        #returns a vector of observations
        if self.closest_track_ind + self.Horizon >= self.track_len:
            X_at_H = self.track[:, self.closest_track_ind + self.Horizon -
                                self.track_len]
        else:
            X_at_H = self.track[:, self.closest_track_ind + self.Horizon]
        return (np.array(
            [self.x, self.y, self.psi, self.v_x, self.v_y, self.omega,
             X_at_H], dtype=object))

    def step(self, action):
        # Steps the simulation one time unit into the future
        # Args:
        #   action is between [-1,1]
        steer = action[0]
        thrust = action[1]
        self.stepCount +=1

        # check action limits
        if steer >= 1:
            self.steer = 1
            self.max_steer_flag = True
        elif steer <= -1:
            self.steer = -1
            self.max_steer_flag = True
        else:
            self.max_steer_flag = False

        if thrust >= 1:
            self.thrust = 1
        elif thrust <= -1:
            self.thrust = -1

        #rescale inputs
        steer_re = (self.steer + 1) / 2 * (self.steer_max -
                                           self.steer_min) + self.steer_min
        thrust_re = (self.thrust + 1) / 2 * (self.thrust_max -
                                             self.thrust_min) + self.thrust_min

        #perform a step (i.e. do dynamics) <<< EDIT THIS FOR DYNAMICS
        psi = self.psi
        v_x = self.v_x
        v_y = self.v_y
        omega = self.omega

        #slip angles
        alpha = [
            np.arctan2(v_y + self.l_f * omega, v_x) - steer_re,
            np.arctan2(v_y - self.l_r * omega, v_x)
        ]

        #complete Fiala lateral tyre force model
        #based on tire slip angle and applied force to rear tires
        # Tire 1: Front
        # Tire 2: Rear
        F_y = np.array([0, 0])
        for tire in range(0, 2):
            #calculate lateral force
            z = np.tan(alpha[tire])
            if np.abs(alpha[tire]) < self.alpha_slip[tire]:
                #No slipping
                F_y[tire] = self.slip_coeff[0][tire]*z \
                    + np.abs(z)*z*self.slip_coeff[1][tire]  \
                    + z**3*self.slip_coeff[2][tire]
            else:
                #Slipping
                #alpha should never be zero otherwise sign function spits out something that isn't {-1,1}
                F_y[tire] = self.no_slip_coeff * np.sign(alpha[tire] + 0.00001)

        #lagrange eqns
        dx = v_x * np.cos(psi) - v_y * np.sin(psi)
        dy = v_x * np.sin(psi) + v_y * np.cos(psi)
        dpsi = omega
        dv_x = 1 / self.m * (1000 * thrust_re -
                             F_y[0] * np.sin(steer_re)) + v_y * omega
        dv_y = 1 / self.m * (F_y[1] + F_y[0] * np.cos(steer_re)) - v_x * omega
        domega = 1 / self.I_z * (F_y[0] * self.l_f * np.cos(steer_re) -
                                 F_y[1] * self.l_r)

        #update states
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.psi = wrap_angle(self.psi + dpsi * self.dt)
        self.v_x += dv_x * self.dt
        self.v_y += dv_y * self.dt
        self.omega += domega * self.dt

        # find closest point to track (required for control and reward signal)
        # start at previous track_ind and search forward
        self.closest_track_pt_x, self.closest_track_pt_y, self.closest_track_ind = line_search_tool(
            self.track, self.x, self.y, self.closest_track_ind)

        #save actions
        self.steer = steer
        self.thrust = thrust

        done = False
        # Distance of car from the track
        distance_check = np.hypot(self.x - self.closest_track_pt_x,
                                  self.y - self.closest_track_pt_y)
        if distance_check > self.distance_threshold:
            done = True
        if self.stepCount >= self.maxStep:
            done = True
        if (self.x < self.track_boundaries[0]
                or self.x > self.track_boundaries[1]
                or self.y < self.track_boundaries[2]
                or self.y > self.track_boundaries[3]):
            done = True
        return self.get_observation(), self.reward(), done, {}
        
    def reward(self):
        diff = self.closest_track_pt_x - self.x, self.closest_track_pt_y - self.y
        reward1 = -np.sqrt(diff[0]**2 + diff[1]**2)
        reward2 = np.sqrt(self.v_x**2 + self.v_y**2)
        return 0.01 * (reward1 + 10 * reward2)

    def render(self):
        if self.render_env == True:
            plt.figure(1, figsize=(15, 15))
            plt.clf()

            # rescale inputs for display
            steer_re = (self.steer + 1) / 2 * (self.steer_max -
                                               self.steer_min) + self.steer_min

            #draw car
            self._plot_car(x=self.x,
                           y=self.y,
                           psi=self.psi,
                           vx=self.v_x,
                           vy=self.v_y,
                           omega=self.omega,
                           steer=steer_re,
                           color='blue',
                           c_alpha=1)

            # display track
            plt.plot(self.track[0, :],
                     self.track[1, :],
                     'k',
                     linewidth=3,
                     alpha=0.7)  # track outline
            track_ticks = np.arange(0, self.track_len, 50)
            plt.plot(self.track[0, track_ticks],
                     self.track[1, track_ticks],
                     'ok',
                     markersize=8,
                     linewidth=3,
                     alpha=0.7)  # track outline

            #plot car's closest location on track
            plt.plot(self.closest_track_pt_x,
                     self.closest_track_pt_y,
                     marker='o',
                     color='r',
                     markersize=12)

            plt.axis('equal')
            plt.xlim([self.track_boundaries[0], self.track_boundaries[1]])
            plt.ylim([self.track_boundaries[2], self.track_boundaries[3]])
            plt.title('Speed: ' +
                      str(round(2.23 *
                                np.sqrt(self.v_x**2 + self.v_y**2), 1)) +
                      ' (mph)')
            plt.show()
            plt.pause(0.001)

    def _plot_car(self, x, y, psi, vx, vy, omega, steer, color, c_alpha):
        '''Plot a single instance of a car'''
        plt.plot(x, y, marker='o', color=color, markersize=12,
                 alpha=c_alpha)  # plot car CoG
        Rot = np.array([[np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]])
        carOutline = 1.0 * np.array([[-self.l_r, -.6], [self.l_f, -.6],
                                     [self.l_f, .6], [-self.l_r, .6],
                                     [-self.l_r, -.6]])
        carPose = np.dot(Rot, carOutline.transpose())
        plt.plot(carPose[0, :] + x,
                 carPose[1, :] + y,
                 linestyle='dashed',
                 color=color,
                 alpha=c_alpha)  # car

        # display car wheels
        Rot_fw = np.array([[np.cos(steer), -np.sin(steer)],
                           [np.sin(steer), np.cos(steer)]])
        frontWheel = np.dot(
            Rot,
            np.dot(Rot_fw, np.array([[-0.5, 0.5], [0, 0]])) +
            np.array(np.array([[self.l_f, self.l_f], [0, 0]])))

        # display steering angle (and if it is maxed out)
        if (self.max_steer_flag == False):
            plt.plot(frontWheel[0, :] + x,
                     frontWheel[1, :] + y,
                     '-',
                     color=color,
                     linewidth=7,
                     alpha=c_alpha)
        else:
            plt.plot(frontWheel[0, :] + x,
                     frontWheel[1, :] + y,
                     '-',
                     color='red',
                     linewidth=7,
                     alpha=c_alpha)

        # display rear wheel wheel slip
        rearWheel = np.dot(
            Rot, np.array([[-0.5 - self.l_r, 0.5 - self.l_r], [0, 0]]))
        alpha = np.arctan2(vy - self.l_r * omega, vx)

        if np.abs(alpha) >= self.alpha_slip[1]:
            plt.plot(rearWheel[0, :] + x,
                     rearWheel[1, :] + y,
                     '-',
                     color='red',
                     linewidth=5,
                     alpha=c_alpha)
        else:
            plt.plot(rearWheel[0, :] + x,
                     rearWheel[1, :] + y,
                     '-',
                     color=color,
                     linewidth=5,
                     alpha=c_alpha)
