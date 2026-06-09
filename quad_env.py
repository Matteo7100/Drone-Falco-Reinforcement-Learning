import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces


# =============================================================================
#  STATE VECTOR CONVENTION (used consistently everywhere):
#  index  0:3  -> position      [x, y, z]
#  index  3:6  -> euler angles  [phi (roll), theta (pitch), psi (yaw)]
#  index  6:9  -> linear vel    [dx, dy, dz]
#  index 9:12  -> angular vel   [p, q, r]
# =============================================================================

class QuadcopterEnv(gym.Env):

    def __init__(self, render=False):
        super(QuadcopterEnv, self).__init__()

        # ── PyBullet setup ──────────────────────────────────────────────────
        mode = p.GUI if render else p.DIRECT   # use p.DIRECT for training (much faster)
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)          # gravity handled by custom RK4 dynamics
        p.setRealTimeSimulation(0)

        self.plane_id = p.loadURDF("plane.urdf")
        self.drone    = p.loadURDF(r"C:\Users\matte\Desktop\FALCO\RL_DRONE_NEW\quadrotor.urdf")

        # ── Spaces ──────────────────────────────────────────────────────────
        # Observation: [pos(3), euler(3), vel(3), ang_vel(3)] — all normalised to [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # Action: delta thrust per rotor in [-1, 1]; scaled to [0, Fmax] inside step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # ── Physical parameters ─────────────────────────────────────────────
        self.g         = 9.81
        self.m         = 4.0                          # [kg]
        self.l         = 0.15                         # arm length [m]
        self.kd        = 1.5e-9                       # moment coefficient
        self.kf        = 6.11e-8                      # drag coefficient
        self.max_force = 30.0                         # max thrust per rotor [N]
        self.Ts        = 0.01                         # simulation timestep [s]

        # Inertia — symmetric quadcopter
        J         = 2/5 * 3 * 0.02**2 + 2 * self.l**2 * 0.25
        self.I    = np.diag([J, J, J])
        self.Fh   = self.m * self.g / 4              # hover force per rotor [N]

        self.config = {'m': self.m, 'g': self.g, 'l': self.l, 'I': self.I}

        # ── Task ────────────────────────────────────────────────────────────
        self.target_position = np.array([0.0, 0.0, 3.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])

        # ── Episode state ───────────────────────────────────────────────────
        self.timesteps   = 0
        self.max_steps   = 1500
        self.state       = np.zeros(12, dtype=np.float32)
        self.prev_dist   = None       # used for shaping

    # =========================================================================
    #  RESET
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise start slightly to improve generalisation
        rng = np.random.default_rng(seed)
        start_pos = np.array([
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            rng.uniform(1.2, 1.8)
        ])
        start_orn_q = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        p.resetBasePositionAndOrientation(self.drone, start_pos.tolist(), start_orn_q)
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        self.timesteps = 0
        self.prev_dist = float(np.linalg.norm(self.target_position - start_pos))

        self.state = self._make_obs(
            pos     = start_pos,
            euler   = np.zeros(3),
            vel     = np.zeros(3),
            ang_vel = np.zeros(3)
        )
        return self.state.copy()

    # =========================================================================
    #  STEP
    # =========================================================================
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Map [-1,1] → [0, 2*Fh] so that action=0 ≡ exact hover thrust
        scaled_action = self.Fh * (1.0 + action)
        scaled_action = np.clip(scaled_action, 0.0, self.max_force)

        # Integrate dynamics with RK4
        raw_state = self._get_raw_state()      # [pos, euler, vel, ang_vel]
        new_raw   = self._rk4(raw_state, self.Ts, scaled_action)

        new_pos     = new_raw[0:3]
        new_euler   = new_raw[3:6]
        new_vel     = new_raw[6:9]
        new_ang_vel = new_raw[9:12]

        # Push updated state into PyBullet (for rendering only — physics bypass)
        new_orn_q = p.getQuaternionFromEuler(new_euler.tolist())
        p.resetBasePositionAndOrientation(self.drone, new_pos.tolist(), new_orn_q)
        p.resetBaseVelocity(self.drone, new_vel.tolist(), new_ang_vel.tolist())
        p.stepSimulation()

        self.timesteps += 1

        # ── Reward ──────────────────────────────────────────────────────────
        reward, done, info = self._compute_reward(
            new_pos, new_euler, new_vel, new_ang_vel, action
        )

        # ── Observation ─────────────────────────────────────────────────────
        self.state = self._make_obs(new_pos, new_euler, new_vel, new_ang_vel)

        # Time limit
        if self.timesteps >= self.max_steps:
            done = True

        return self.state.copy(), reward, done, info

    # =========================================================================
    #  DYNAMICS
    # =========================================================================
    def _get_raw_state(self):
        """Return [pos(3), euler(3), vel(3), ang_vel(3)] directly from PyBullet."""
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(list(pos) + list(euler) + list(vel) + list(ang_vel), dtype=np.float64)

    def _sys_dynamics(self, y, forces):
        """
        Equations of motion.
        State convention: [x, y, z, phi, theta, psi, dx, dy, dz, p, q, r]
        """
        m, g, l, I = self.config['m'], self.config['g'], self.config['l'], self.config['I']

        F = np.clip(forces, 0.0, self.max_force)
        M = F * (self.kd / self.kf)      # reaction torques

        x,  y_p, z  = y[0], y[1], y[2]
        phi, theta, psi = y[3], y[4], y[5]
        dx, dy, dz  = y[6], y[7], y[8]
        p_r, q_r, r_r = y[9], y[10], y[11]

        # Rotation matrix (body → world)
        R = self._rotation_matrix(phi, theta, psi)

        # Translational acceleration
        thrust_body = np.array([0.0, 0.0, float(np.sum(F))])
        acc = (1.0 / m) * (np.array([0.0, 0.0, -m * g]) + R @ thrust_body)

        # Rotational acceleration
        torque = np.array([
            l * ( F[0] + F[1] - F[2] - F[3]),
            l * (-F[0] + F[1] + F[2] - F[3]),
            -M[0] + M[1] - M[2] + M[3]
        ])
        omega   = np.array([p_r, q_r, r_r])
        domega  = np.linalg.solve(I, torque - np.cross(omega, I @ omega))

        # Euler-angle kinematics (body rates → Euler rates)
        # Guard against gimbal lock (theta ≈ ±90°)
        cos_theta = np.cos(theta)
        if np.abs(cos_theta) < 1e-6:
            cos_theta = np.sign(cos_theta) * 1e-6

        T_euler = np.array([
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi),                 -np.sin(phi)                ],
            [0.0, np.sin(phi) / cos_theta,      np.cos(phi) / cos_theta   ]
        ])
        euler_dot = T_euler @ omega

        return np.concatenate([[dx, dy, dz], euler_dot, acc, domega])

    def _rotation_matrix(self, phi, theta, psi):
        cphi,  sphi  = np.cos(phi),   np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi,  spsi  = np.cos(psi),   np.sin(psi)
        return np.array([
            [cpsi*ctheta,
             sphi*stheta*cpsi - cphi*spsi,
             cphi*stheta*cpsi + sphi*spsi],
            [cpsi*stheta*sphi + cphi*spsi,
             cphi*cpsi - sphi*stheta*spsi,
            -cpsi*sphi + cphi*stheta*spsi],
            [-stheta, sphi*ctheta, cphi*ctheta]
        ])

    def _rk4(self, state, dt, forces):
        k1 = self._sys_dynamics(state,           forces)
        k2 = self._sys_dynamics(state + dt/2*k1, forces)
        k3 = self._sys_dynamics(state + dt/2*k2, forces)
        k4 = self._sys_dynamics(state + dt  *k3, forces)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # =========================================================================
    #  REWARD
    # =========================================================================
    def _compute_reward(self, pos, euler, vel, ang_vel, raw_action):
        """
        Reward design rationale:
          - Dense shaping: reward progress toward target (delta-distance).
          - Exponential proximity bonus: large signal near the goal.
          - Attitude penalty: keep the drone level.
          - Velocity penalty: gentle, only for excessive speed (not for approach).
          - Action regularisation: stay near hover.
          - Survival bonus: small constant to encourage staying alive.
          - Early-termination penalties: flip or out-of-bounds.
        """
        dist = float(np.linalg.norm(self.target_position - pos))
        done = False
        info = {}

        # ── 1. Progress shaping (encourage getting closer) ──────────────────
        delta_dist = self.prev_dist - dist          # positive = got closer
        r_progress = 2.0 * delta_dist
        self.prev_dist = dist

        # ── 2. Proximity bonus (exponential — peaks at goal) ─────────────────
        r_proximity = 3.0 * np.exp(-1.5 * dist)

        # ── 3. Attitude penalty ───────────────────────────────────────────────
        att_error = np.linalg.norm(euler)           # should be near [0,0,0]
        r_attitude = -0.5 * att_error

        # ── 4. Velocity penalty (only excessive speed) ────────────────────────
        speed = float(np.linalg.norm(vel))
        r_velocity = -0.02 * max(0.0, speed - 1.0)  # free up to 1 m/s

        # ── 5. Angular velocity penalty ───────────────────────────────────────
        r_ang_vel = -0.005 * float(np.linalg.norm(ang_vel))

        # ── 6. Action regularisation (stay near hover) ────────────────────────
        r_action = -0.01 * float(np.sum(np.square(raw_action)))

        # ── 7. Survival bonus ─────────────────────────────────────────────────
        r_survive = 0.5

        reward = r_progress + r_proximity + r_attitude + r_velocity + r_ang_vel + r_action + r_survive

        # ── Early termination ─────────────────────────────────────────────────
        # Flip (more than 60° tilt)
        if np.abs(euler[0]) > np.pi / 3 or np.abs(euler[1]) > np.pi / 3:
            reward -= 30.0
            done    = True
            info['termination'] = 'flip'

        # Out of bounds
        elif np.any(np.abs(pos[:2]) > 8.0) or pos[2] < 0.05 or pos[2] > 15.0:
            reward -= 30.0
            done    = True
            info['termination'] = 'out_of_bounds'

        info.update({
            'r_progress':  r_progress,
            'r_proximity': r_proximity,
            'r_attitude':  r_attitude,
            'r_velocity':  r_velocity,
            'r_action':    r_action,
            'dist':        dist,
        })
        return float(reward), done, info

    # =========================================================================
    #  OBSERVATION BUILDER
    # =========================================================================
    def _make_obs(self, pos, euler, vel, ang_vel):
        """Normalise and stack into a [-1,1] observation vector."""
        pos_n     = np.clip(np.array(pos)     / 10.0, -1.0, 1.0)
        euler_n   = np.clip(np.array(euler)   / np.pi, -1.0, 1.0)
        vel_n     = np.clip(np.array(vel)     / 5.0,  -1.0, 1.0)
        ang_vel_n = np.clip(np.array(ang_vel) / 5.0,  -1.0, 1.0)
        return np.hstack((pos_n, euler_n, vel_n, ang_vel_n)).astype(np.float32)

    # =========================================================================
    def close(self):
        p.disconnect(self.client)
