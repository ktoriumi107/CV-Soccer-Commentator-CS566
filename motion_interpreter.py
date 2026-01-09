import math
from collections import defaultdict, deque
import numpy as np

class MotionInterpreter:
    def __init__(self):
        # params
        self.history = 15 # frames of history to store
        self.min_speed = 1 # minimum speed to not be stationary (since there is camera movement, could be more robust)
        self.direction_thresh = .5 # threshold for moving towards a certain position

        # deque[(frame_idx, (x,y))]
        self.pos = defaultdict(lambda: deque(maxlen=self.history))

        # (vx, vy)
        self.vel = {}

        # interpreted state
        self.state = {}

    def update(self, tid, player_pos, ball_pos, frame_idx):
        if ball_pos is None:
            return
        
        self.pos[tid].append((frame_idx, player_pos))

        # at least 2 samples to compute velocity
        if len(self.pos[tid]) < 2:
            self.state[tid] = "only 1 sample"
            return

        # get positions and change in 
        (_, p_prev), (_, p_curr) = self.pos[tid][-2], self.pos[tid][-1]
        dt = max(1e-5, self.pos[tid][-1][0] - self.pos[tid][-2][0])

        # velocity as change in position over change in time
        # actually frames but assume frames come at consistent frequency
        vx = (p_curr[0] - p_prev[0]) / dt
        vy = (p_curr[1] - p_prev[1]) / dt
        self.vel[tid] = (vx, vy)

        speed = math.hypot(vx, vy)

        if speed < self.min_speed:
            self.state[tid] = "stationary"
            return

        # get direction
        direction = np.array([vx, vy]) / (speed + 1e-6)

        # distance to ball
        to_ball = np.array([ball_pos[0] - p_curr[0],
                            ball_pos[1] - p_curr[1]])
        
        # distance to ball
        dist_ball = np.linalg.norm(to_ball)
        
        if dist_ball < 1e-6:
            self.state[tid] = "at_ball"
            return
        
        direction_to_ball = to_ball / dist_ball

        # dot product for direction
        dot = np.dot(direction, direction_to_ball)

        if dot > self.direction_thresh:
            self.state[tid] = "moving towards ball"
        elif dot < -self.direction_thresh:
            self.state[tid] = "moving away from ball"
        else:
            self.state[tid] = "moving perpendicular to ball"

    def get_state(self, tid):
        vx, vy = self.vel.get(tid, (0,0))
        return {
            "state": self.state.get(tid, "unknown"),
            "velocity": (vx, vy),
            "speed": math.hypot(vx, vy),
        }
