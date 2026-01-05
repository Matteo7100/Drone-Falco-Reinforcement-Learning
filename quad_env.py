import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time

class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()
        

        # Initialize PyBullet
        self.client = p.connect(p.GUI)  
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)  # Disable real-time so we can manually control step timing

        # Load Quadcopter Model (Replace with actual URDF later)
        # Set path to built-in data (includes ground plane, URDFs, etc.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the default ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone = p.loadURDF(r"C:\Users\matte\Desktop\FALCO\Drone_Falco\quadrotor.urdf")

        # Define action space 
        # The action space represents the set of all possible actions that an agent can take at any given time.
        # Here, the action space is being defined using the spaces.Box class from the gym library,
        # which is commonly used to create continuous action spaces.

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Target Position and Orientation (Hover at 2m with no rotation)
        self.target_position = np.array([0, 0, 3])
        self.target_attitude = np.array([0, 0, 0])  # Roll, Pitch, Yaw
        self.target_quaternion = p.getQuaternionFromEuler(self.target_attitude)

        self.Ts = 0.01 # Time step for simulation
        self.timesteps = 0
        self.Done = False

        self.max_force = 30  # Maximum force applied by each rotor
        self.kd = 1.5 * pow(10 , -9)   # Moment coefficient
        self.kf = 6.11 * pow(10 , -8)  # Aerodynamic drag coefficient
        self.g = 9.81
        self.m = 4 # Mass of the quadcopter
        self.l = 0.15 # Distance from the center of mass to the rotor
        self.J = 2/5 * 3 * 0.02**2 + 2 * self.l**2 * 0.25
        self.I = np.diag([self.J, self.J, self.J])  # Inertia matrix for a symmetric quadcopter
        self.config = {
            'm': self.m,
            'g': self.g,
            'l': self.l,
            'I': self.I
        }
        self.Fh = self.m*self.g / 4  # Hover force per rotor

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)        
        self.reward = 0
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        starting_pose = [0, 0, 1.5]  # Start at 1m height
        starting_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        p.resetBasePositionAndOrientation(self.drone, starting_pose, starting_orn)
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])
        
        self.timesteps = 0  # 🔧 Reset timestep counter
        
        # Return normalized initial state
        pos_norm = np.clip(np.array(starting_pose) / 10, -1, 1)
        euler_norm = np.array([0, 0, 0])
        vel_norm = np.array([0, 0, 0])
        ang_vel_norm = np.array([0, 0, 0])
        
        self.state = np.hstack((pos_norm, euler_norm, vel_norm, ang_vel_norm))
        return self.state

    def _get_state(self):
        """Get quadcopter state: position, velocity, attitude, angular velocity"""
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        return np.array(pos + vel + (roll, pitch, yaw) + ang_vel)
    

    def rotation_matrix(self, phi, theta, psi):
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R = np.array([
            [cpsi * ctheta, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
            [cpsi * stheta * sphi + cphi * spsi, cphi * cpsi - sphi * stheta * spsi, -cpsi * sphi + cphi * stheta * spsi],
            [-stheta, sphi * ctheta, cphi * ctheta]
        ])
        return R

    def sys_dynamics(self, y, action, params):
        m, g, l, I = params['m'], params['g'], params['l'], params['I']
        F = action # array of 4 forces [F1, F2, F3, F4]
        # Check force saturation
        F = np.clip(F, -self.max_force, self.max_force)  # Clip forces to max thrust

        M = np.array([f * self.kd / self.kf for f in F])  # Compute torques from forces


        # Unpack state vector
        x, y_pos, z, dx, dy, dz, phi, theta, psi, p, q, r = y

        # Compute rotation matrix
        R = self.rotation_matrix(phi, theta, psi)

        # Translational acceleration
        thrust = np.array([0, 0, sum(F)])
        gravity = np.array([0, 0, -m * g])
        acc = (1 / m) * (gravity + R @ thrust)

        # Rotational acceleration
        torque = np.array([
            l * (F[0] + F[1] - F[2] - F[3]),
            l * (-F[0] + F[1] + F[2] - F[3]),
            -M[0] + M[1] - M[2] + M[3]
        ])
        omega = np.array([p, q, r])
        domega = np.linalg.inv(I) @ (torque - np.cross(omega, I @ omega))

        # Euler angle rates
        T = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        euler_dot = T @ omega

        # Return derivative of state vector
        return np.concatenate([
            [dx, dy, dz],     # dx/dt
            acc,              # dv/dt
            euler_dot,        # dangles/dt
            domega            # domega/dt
        ])


    # Runge-Kutta 4 integrator
    def integration(self, state, dt, action, params):
        k1 = self.sys_dynamics(state, action, params)
        k2 = self.sys_dynamics(state + dt/2 * k1, action, params)
        k3 = self.sys_dynamics(state + dt/2 * k2, action, params)
        k4 = self.sys_dynamics(state + dt * k3, action, params)

        new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # # CORRECT: Return position and orientation properly
        # new_pos = new_state[0:3]      # x, y, z
        # new_orn = new_state[6:9]      # roll, pitch, yaw
        
        # return new_pos, new_orn

        return new_state
    

    def step(self, action):
        # Scale action to proper force range: Fi = Fh + a_i*(Fmax − Fmin)/2
        scaled_action = self.Fh * (1.0 + action)        
        scaled_action = np.clip(scaled_action, 0, self.max_force)
        # Get current state and integrate dynamics
        y = self._get_state()
        new_state = self.integration(y, self.Ts, scaled_action, self.config)
        
        new_pos = new_state[0:3]
        new_orn_euler = new_state[6:9]
        new_vel = new_state[3:6]
        new_ang_vel = new_state[9:12]

        # CRITICAL: Update PyBullet simulation state
        new_orn_quat = p.getQuaternionFromEuler(new_orn_euler)
        p.resetBasePositionAndOrientation(self.drone, new_pos, new_orn_quat)
        
        p.resetBaseVelocity(self.drone, new_vel, new_ang_vel)
        
        # Step simulation
        p.stepSimulation()
        self.timesteps += 1
        
        # Get updated state for observation
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        euler = np.array(p.getEulerFromQuaternion(orn))
        
        # Compute errors
        error_pos = np.linalg.norm(self.target_position - np.array(pos))
        error_orn = np.linalg.norm(self.target_attitude - euler)
        
        #   REWARD FUNCTION  
        # Parameters:     
        alpha_a = 0.025
        alpha_p = 1.0
        alpha_v = 0.05
        alpha_omega = 0.001
        alpha_xi = 0.02
        alpha_r = 0.02
        beta = 2.0

        reward0, reward1, reward2, reward3 = 0, 0, 0, 0
        self.reward = 0
        # reward0: encourage stay alive
        reward0 = beta
        # penalize eccessive actions
        reward1 = -alpha_a * np.sum(np.square(action))
        # penalize position error
        reward2 = -(alpha_p * np.linalg.norm(error_pos) + alpha_v * np.linalg.norm(vel) + alpha_omega * np.linalg.norm(ang_vel))
        # penalize orientation error
        reward3 = -(alpha_xi * np.linalg.norm(error_orn) + alpha_r * np.linalg.norm(euler))
        
        self.reward += reward0 + reward1 + reward2 + reward3

        # Time limit
        done = (self.timesteps >= 1500)  # End after 100000 timesteps


        # Update observation
        pos_norm = np.clip(np.array(pos) / 10, -1, 1)
        vel_norm = np.clip(np.array(vel) / 5, -1, 1)
        ang_vel_norm = np.clip(np.array(ang_vel) / 5, -1, 1)
        euler_norm = np.clip(euler / np.pi, -1, 1)
        
        self.state = np.hstack((pos_norm, euler_norm, vel_norm, ang_vel_norm))
        
        return self.state, self.reward, done, { "reward0": reward0,
                                            "reward1": reward1,
                                            "reward2": reward2,
                                            "reward3": reward3
                                            }
    def close(self):
        p.disconnect()

##### def reset(self, seed=None, options=None):
    #     """Reset the environment to initial state."""
    #     super().reset(seed=seed)  # Correct seeding for Gym >= 0.26

    #     starting_pose = [0,0,0]
    #     starting_orn = p.getQuaternionFromEuler([0, 0, 0])

    #     p.resetBasePositionAndOrientation(self.drone, starting_pose, starting_orn)
    #     p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])   
    #     # state describe the initial condition of the quadcopter (position, orientation and velocity)
    #     self.state = np.hstack((starting_pose, p.getEulerFromQuaternion(starting_orn), np.zeros(6)))

    #     return self.state

    #------------------------------------------------------------------------------------------------

    #### def step(self, action):
    #     """Perform one step in the simulation."""
    #     # Get drone state
    #     pos, orn = p.getBasePositionAndOrientation(self.drone)
    #     euler = np.array(p.getEulerFromQuaternion(orn))
    #     vel, ang_vel = p.getBaseVelocity(self.drone)

    #     y = self._get_state()
    #     new_pose, new_orn = self.integration(y, self.Ts, action, self.config)
        
        
    #     # Compute position errors
    #     error_pos = self.target_position - np.array(new_pose)
    #     error_orn = np.array(self.target_attitude) - np.array(new_orn)

    #     # Update simulation
    #     p.stepSimulation()
    #     self.timesteps += 1
    #     # print(f"Step: {self.timesteps}, Position: {pos}, Orientation: {euler}, Velocity: {vel}, Angular Velocity: {ang_vel} \n")

    #     # Normalize observation
    #     pos = np.array(pos)  # Convert tuple to NumPy array
    #     vel = np.array(vel)
    #     ang_vel = np.array(ang_vel)
    #     euler_norm = np.array(euler)

    #     pos_norm = np.clip(pos / 10, -1, 1)  # Normalize position
    #     vel_norm = np.clip(vel / 5, -1, 1)  # Normalize velocity
    #     ang_vel_norm = np.clip(ang_vel / 5, -1, 1)  # Normalize angular velocity
    #     euler_norm = np.clip(euler / np.pi, -1, 1)

    #     self.state = np.hstack((pos_norm, euler_norm, vel_norm, ang_vel_norm))

    #     # Reward shaping
    #     reward =  np.linalg.norm(error_pos)  # Penalize position error with 
    #     reward += 0.3 * np.linalg.norm(error_orn)  # Penalize orientation error
    #     reward += 0.05 * np.linalg.norm(vel)  # Encourage smooth flight
    #     reward += 0.05 * np.linalg.norm(ang_vel)  # Reduce excessive rotations

    #     if self.timesteps == 100000:
    #         self.Done = True



    #     return self.state, reward, self.Done, False
