"""
gp_ac_mpc_ros2.py
================================
Fixes:
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

import numpy as np
import math
import time
import os
import random
import warnings
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import casadi as ca
import do_mpc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', module='sklearn')

# ============================================================
#  Global Parameters
# ============================================================

MASS        = 20.133
IZ          = 5.768
IY          = 10.58
B_ANGLE     = 1.47
OMEGA       = 0.15
SPEED_SCALE = 0.75
LOS_L       = 1.0
DEPTH_TARGET= -3.5
DT          = 0.1


# ============================================================
#  Nominal Dynamics Integration (core fix for GP residuals)
# ============================================================

def nominal_step(state, controls):
    """
    One-step nominal dynamics integration (no ocean current, no disturbance).
    Used as the correct baseline for computing GP residuals.
    state    : [x, y, u, r, pitch_rate, pitch, yaw, depth]
    controls : [F_LF, F_LB, F_RF, F_RB, F_angle]
    """
    if state is None or controls is None:
        return list(state) if state else [0]*8
    x, y, u, r, pr, pt, yw, d = [float(v) for v in state[:8]]
    F_LF, F_LB, F_RF, F_RB, ang = [float(v) for v in controls[:5]]
    B = B_ANGLE
    v_est = r * 0.5
    x_new = x + DT*(u*math.cos(pt)*math.cos(yw) - v_est*math.sin(yw))
    y_new = y + DT*(u*math.cos(pt)*math.sin(yw) + v_est*math.cos(yw))
    u_dot = (1/MASS)*(-24.3748*u*abs(u)+3.255*u+0.1*r*abs(r)
        -0.5*u*u*0.15*math.cos(B)*0.2
        -F_LF*math.sin(ang)-F_LB*math.sin(B)
        -F_RF*math.sin(ang)-F_RB*math.sin(B))
    u_new = u + DT*u_dot
    d_new = d + DT*(-math.sin(pt)*u)
    r_dot = (1/IZ)*(-44.0*r*abs(r)-4.88*r-2.0*r*abs(u*u)
        +F_LF*0.613*math.sin(ang)+F_LB*0.613*math.sin(B)
        -F_RF*0.613*math.sin(ang)-F_RB*0.613*math.sin(B))
    r_new = r + DT*r_dot
    pr_new = pr + DT*(1/IY)*(-0.5*1000*u*u*0.15*math.cos(ang)*0.3*0.5
        +pt*abs(pt)+pt+(F_LF+F_RF)*0.3*math.cos(ang))
    pt_new = pt + DT*pr
    yw_new = yw + DT*r
    return [x_new, y_new, u_new, r_new, pr_new, pt_new, yw_new, d_new]


# ============================================================
#  Utility Functions
# ============================================================

def normalize_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return float(a)

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau*sp.data + (1-tau)*tp.data)

def save_training_log(filename, x_hist, y_hist, depth_hist,
                      xr_hist, yr_hist, zr_hist, t_hist,
                      roll_hist, pitch_hist, heading_hist,
                      lf_hist, lb_hist, rf_hist, rb_hist,
                      front_angle_hist,
                      actor_losses, critic_losses,
                      Q_x_hist, Q_y_hist, Q_depth_hist,
                      Q_yaw_hist, R_thr_hist,
                      gp_mean_x_hist, gp_mean_y_hist,gp_mean_z_hist,
                      gp_std_x_hist, gp_std_y_hist, gp_std_z_hist,
                      reward_hist,
                      buffer_size):
    with open(filename, 'w') as f:
        f.write("# GP-AC-MPC Training Log\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Buffer: {buffer_size}  Steps: {len(x_hist)}\n\n")
        f.write("# t x y depth xref yref zref "
                "roll pitch heading lf lb rf rb angle\n")
        n = min(len(x_hist), len(xr_hist))
        for i in range(n):
            f.write(f"{t_hist[i]:.3f} "
                    f"{x_hist[i]:.4f} {y_hist[i]:.4f} "
                    f"{depth_hist[i]:.4f} "
                    f"{xr_hist[i]:.4f} {yr_hist[i]:.4f} "
                    f"{zr_hist[i]:.4f} "
                    f"{roll_hist[i]:.4f} {pitch_hist[i]:.4f} "
                    f"{heading_hist[i]:.4f} "
                    f"{lf_hist[i]:.4f} {lb_hist[i]:.4f} "
                    f"{rf_hist[i]:.4f} {rb_hist[i]:.4f} "
                    f"{front_angle_hist[i]:.4f}\n")
        f.write("\n# actor_loss critic_loss\n")
        nl = min(len(actor_losses), len(critic_losses))
        for i in range(nl):
            f.write(f"{actor_losses[i]:.6f} "
                    f"{critic_losses[i]:.6f}\n")
        f.write("\n# t Q_x Q_y Q_depth Q_yaw R_thr "
                "gp_mean_x gp_mean_y gp_mean_z gp_std_x gp_std_y gp_std_z reward\n")
        nw = min(len(Q_x_hist), len(gp_mean_x_hist))
        for i in range(nw):
            f.write(f"{i*0.1:.3f} "
                    f"{Q_x_hist[i]:.4f} {Q_y_hist[i]:.4f} "
                    f"{Q_depth_hist[i]:.4f} {Q_yaw_hist[i]:.4f} "
                    f"{R_thr_hist[i]:.6f} "
                    f"{gp_mean_x_hist[i]:+.4f} "
                    f"{gp_mean_y_hist[i]:+.4f} "
                    f"{gp_mean_z_hist[i]:+.4f} "
                    f"{gp_std_x_hist[i]:.4f} "
                    f"{gp_std_y_hist[i]:.4f} "
                    f"{gp_std_z_hist[i]:+.4f} "
                    f"{reward_hist[i]:+.4f} \n")


# ============================================================
#  GP Residual Model
# ============================================================

class GPResidualModel:
    def __init__(self, max_data=300, update_interval=20,
                 std_threshold=0.3):
        self.max_data        = max_data
        self.update_interval = update_interval
        self.std_threshold   = std_threshold
        self.X_buf    = deque(maxlen=max_data)
        self.Y_buf    = deque(maxlen=max_data)
        self.x_scaler = StandardScaler()
        self.gps = []
        for _ in range(3):
            k = RBF(1.0,(0.1,10.0)) + WhiteKernel(1e-4,(1e-6,0.1))
            self.gps.append(GaussianProcessRegressor(
                kernel=k, n_restarts_optimizer=0,
                normalize_y=False, alpha=1e-4))
        self.is_fitted  = False
        self.fit_count  = 0
        self.step_count = 0
        self._accum_steps = 5
        self._accum_real  = np.zeros(3)
        self._accum_nom   = np.zeros(3)
        self._accum_s0    = None
        self._accum_ctrl  = None
        self._accum_cnt   = 0

    def _feat(self, state, controls):
        u=float(state[2]); r=float(state[3])
        p=float(state[5]); y=float(state[6])
        Fs=float(np.clip(sum(controls[:4]),-400,400))
        Fa=float(controls[4])
        return np.array([u,r,p,math.sin(y),math.cos(y),
                         Fs,Fa,r*u,u*abs(u)], dtype=np.float64)

    def push_step(self, s_before, controls, s_real, s_nom):
        """
        s_real : actual measured state (includes ocean current disturbance)
        s_nom  : nominal integration prediction (nominal_step output, no ocean current)
        residual: s_real - s_nom ≈ drift caused by ocean current
        """
        if self._accum_s0 is None:
            self._accum_s0   = list(s_before)
            self._accum_ctrl = list(controls)
            self._accum_real = np.zeros(3)
            self._accum_nom  = np.zeros(3)
            self._accum_cnt  = 0
        self._accum_real += np.array([
            float(s_real[0])-float(s_before[0]),
            float(s_real[1])-float(s_before[1]),
            float(s_real[7])-float(s_before[7])])
        self._accum_nom  += np.array([
            float(s_nom[0])-float(s_before[0]),
            float(s_nom[1])-float(s_before[1]),
            float(s_nom[7])-float(s_before[7])])
        self._accum_cnt += 1
        if self._accum_cnt >= self._accum_steps:
            res = self._accum_real - self._accum_nom
            if np.all(np.abs(res)<1.0) and not np.any(np.isnan(res)):
                feat = self._feat(self._accum_s0, self._accum_ctrl)
                self.X_buf.append(feat)
                self.Y_buf.append(res)
                self.step_count += 1
            self._accum_s0  = None
            self._accum_cnt = 0
            if (self.step_count % self.update_interval == 0
                    and len(self.X_buf) >= 20):
                self._fit()

    def _fit(self):
        X = np.stack(self.X_buf); Y = np.array(self.Y_buf)
        if np.any(Y.std(axis=0) < 1e-6): return
        try: Xs = self.x_scaler.fit_transform(X)
        except: return
        ok = True
        for i,gp in enumerate(self.gps):
            try: gp.fit(Xs, Y[:,i])
            except: ok=False; break
        if ok:
            self.is_fitted = True
            self.fit_count += 1
            Y_mean = Y.mean(axis=0)
            Y_std  = Y.std(axis=0)
            print(f"  [GP] fit#{self.fit_count}  n={len(X)}  "
                  f"Y_mean=({Y_mean[0]:+.4f},{Y_mean[1]:+.4f})  "
                  f"Y_std=({Y_std[0]:.4f},{Y_std[1]:.4f})")

    def predict(self, state, controls):
        if not self.is_fitted:
            return np.zeros(3), np.ones(3)*0.5
        feat = self._feat(state, controls).reshape(1,-1)
        try: fs = self.x_scaler.transform(feat)
        except: return np.zeros(3), np.ones(3)*0.5
        ms, ss = [], []
        for gp in self.gps:
            try:
                mu,sigma = gp.predict(fs, return_std=True)
                ms.append(float(mu[0]))
                ss.append(float(np.clip(sigma[0],0,1.0)))
            except: ms.append(0.0); ss.append(0.5)
        mean = np.array(ms); std = np.array(ss)
        # Truncation threshold: single-step ocean current residual ~0.075m, set to 0.15m
        if np.any(np.abs(mean) > 0.25):
            mean = np.zeros(3)
        return mean, std

    def should_compensate(self):
        return self.is_fitted and self.fit_count >= 2


# ============================================================
#  Actor / Critic Networks
# ============================================================


class ActorNet(nn.Module):
    def __init__(self, state_dim=12, weight_dim=7, hidden=128):
        super().__init__()
        bounds = list(WEIGHT_BOUNDS.values())
        self.register_buffer('w_lb',
            torch.tensor([b[0] for b in bounds], dtype=torch.float32))
        self.register_buffer('w_ub',
            torch.tensor([b[1] for b in bounds], dtype=torch.float32))
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,hidden),    nn.LayerNorm(hidden), nn.ReLU())
        self.w_head = nn.Sequential(
            nn.Linear(hidden,hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, weight_dim))
        with torch.no_grad():
            nn.init.zeros_(self.w_head[-1].bias)
            nn.init.uniform_(self.w_head[-1].weight,-0.01,0.01)

    def forward(self, s):
        f = self.backbone(s)
        return (self.w_lb +
                (self.w_ub-self.w_lb)*torch.sigmoid(self.w_head(f)))


class CriticNet(nn.Module):
    def __init__(self, state_dim=12, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,hidden),    nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,hidden//2), nn.ReLU(),
            nn.Linear(hidden//2,1))
    def forward(self, s): return self.net(s)


class ReplayBuffer:
    def __init__(self, cap=50000): self.buf = deque(maxlen=cap)
    def push(self, s, w, r, ns, done=False):
        self.buf.append((np.array(s,dtype=np.float32),
                         np.array(w,dtype=np.float32),
                         float(r), np.array(ns,dtype=np.float32),
                         float(done)))
    def sample(self, n):
        b = random.sample(self.buf, n)
        s,w,r,ns,d = zip(*b)
        return (np.stack(s), np.stack(w),
                np.array(r,dtype=np.float32),
                np.stack(ns), np.array(d,dtype=np.float32))
    def __len__(self): return len(self.buf)


def compute_reward(curr_state, gp_std):
    xe    = float(curr_state[8])
    ye    = float(curr_state[9])
    ze    = float(curr_state[10])
    yaw_e = float(curr_state[11])
    depth = float(curr_state[7])
    u_vel = float(curr_state[2])

    err = math.sqrt(xe**2 + ye**2 + ze**2)

    reward  = -2.0 * abs(xe)
    reward -= 0.5 * abs(ye)
    reward -= 2.0 * abs(ze)

    if err > 1.0:
        reward -= 3.0 * (err - 1.0)

    reward -= 0.3 * abs(yaw_e)

    if abs(u_vel) > 2.0:
        reward -= 0.3 * (abs(u_vel) - 2.0)

    if depth < -12.0 or depth > 0.3:
        reward -= 3.0

    reward -= 0.1 * float(np.mean(gp_std[:3]))

    return float(reward)


# ============================================================
#  GP-AC-MPC Node
# ============================================================

class GPACMPCNode(Node):

    def __init__(self):
        super().__init__('gp_ac_mpc_node')

        self.x=self.y=self.depth=0.0
        self.u_vel=self.v_vel=self.w_vel=0.0
        self.r_vel=self.pitch_rate=0.0
        self.roll=self.pitch=self.yaw=0.0
        self.time_log=0.0
        self.x_ref=2.0; self.y_ref=1.0; self.z_ref=-1.0

        self.Q_x=5.0; self.Q_y=5.0; self.Q_depth=10.0
        self.Q_yaw=1.0; self.Q_pitch=0.5
        self.R_thr=0.001; self.R_ang=0.1

        self.gp_mean = np.zeros(3)
        self.gp_std  = np.ones(3)*0.5
        self._prev_state_for_gp    = None
        self._prev_controls_for_gp = None
        self._prev_angle           = 1.47
        self.reward  = 0

        self.step_count      = 0
        self.warmup_steps    = 100
        self.enable_training = False
        self.prev_weights    = None
        self.last_state      = None
        self.last_weights    = None
        self.solve_times     = []

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.gp = GPResidualModel(
            max_data=300, update_interval=20, std_threshold=0.3)

        self._init_ac()
        self._init_ros2()
        self._init_logging()
        self._build_mpc()

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            f"GP-AC-MPC ready  device={self.device}")

    # ----------------------------------------------------------
    def _init_ac(self):
        self.actor      = ActorNet(12,7,128).to(self.device)
        self.critic     = CriticNet(12,128).to(self.device)
        self.tgt_critic = CriticNet(12,128).to(self.device)
        self.tgt_critic.load_state_dict(self.critic.state_dict())
        for p in self.tgt_critic.parameters():
            p.requires_grad = False
        self.a_opt = optim.Adam(self.actor.parameters(),  lr=1e-4)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.buffer     = ReplayBuffer(50000)
        self.batch_size = 128
        self.gamma      = 0.99
        self.tau        = 0.005
        self.a_loss_hist = []
        self.c_loss_hist = []

    # ----------------------------------------------------------
    def _init_ros2(self):
        self.sub = self.create_subscription(
            Odometry, '/model/qingzhuan/odometry',
            self.pose_callback, 10)
        self.traj_pub   = self.create_publisher(
            Marker, '/trajectory_marker', 10)
        self.lb_pub     = self.create_publisher(
            Float64, '/qingzhuan/lb_trust_joint/cmd', 10)
        self.lf_pub     = self.create_publisher(
            Float64, '/qingzhuan/lf_trust_joint/cmd', 10)
        self.rb_pub     = self.create_publisher(
            Float64, '/qingzhuan/rb_trust_joint/cmd', 10)
        self.rf_pub     = self.create_publisher(
            Float64, '/qingzhuan/rf_trust_joint/cmd', 10)
        self.behind_pub = self.create_publisher(
            Float64, '/qingzhuan/behind_joint/cmd', 10)
        self.front_pub  = self.create_publisher(
            Float64, '/qingzhuan/front_joint/cmd', 10)
        self.behind_pub.publish(Float64(data=1.47))
        self.front_pub.publish(Float64(data=2.07))

    # ----------------------------------------------------------
    def _init_logging(self):
        save_dir = ('/home/zhiteng/ami_ws/src/fishbot_description'
                    '/launch/3D_Trajectory_Track/results')
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_file = os.path.join(
            save_dir, f'gp_ac_mpc_{ts}.txt')
        self.t_hist=[]; self.x_hist=[]; self.y_hist=[]
        self.depth_hist=[]; self.xr_hist=[]; self.yr_hist=[]
        self.zr_hist=[]; self.roll_hist=[]
        self.pitch_hist=[]; self.heading_hist=[]
        self.lf_hist=[]; self.lb_hist=[]
        self.rf_hist=[]; self.rb_hist=[]
        self.front_angle_hist=[]
        self.err_hist=[]
        self.reward_hist=[]
        self.Q_x_hist=[]; self.Q_y_hist=[]
        self.Q_depth_hist=[]; self.Q_yaw_hist=[]
        self.R_thr_hist=[]
        self.gp_mean_x_hist=[]; self.gp_mean_y_hist=[]
        self.gp_std_x_hist=[]; self.gp_std_y_hist=[]
        self.gp_mean_z_hist=[]; self.gp_std_z_hist=[]
        self.get_logger().info(f"Log -> {self.data_file}")

    # ----------------------------------------------------------
    def _get_ocean_current(self, t):
        """
        Ocean current model: steady-state + periodic + random turbulence.
        The MPC nominal model excludes this term, generating unmodeled
        residuals for the GP to learn and compensate.
        """
        if not hasattr(self, '_oc_rng'):
            self._oc_rng   = np.random.default_rng(42)
            self._oc_noise = np.zeros(3)
            self._oc_nt    = -1.0
        if t - self._oc_nt >= 0.5:
            self._oc_noise = self._oc_rng.normal(
                0, [0.05, 0.05, 0.02])
            self._oc_nt = t
        cx = 0.15 + 0.10*math.sin(2*math.pi*t/30.0) + self._oc_noise[0]
        cy = 0.10 + 0.05*math.sin(2*math.pi*t/40.0) + self._oc_noise[1]
        cz = 0.05 + 0.04*math.sin(2*math.pi*t/50.0) + self._oc_noise[2]
        return float(cx), float(cy), float(cz)

    # ----------------------------------------------------------
    def _build_mpc(self):
        model = do_mpc.model.Model('continuous')
        xp =model.set_variable('_x','x_position')
        yp =model.set_variable('_x','y_position')
        uv =model.set_variable('_x','u_vel')
        rv =model.set_variable('_x','r_vel')
        pr =model.set_variable('_x','pitch_rate')
        pt =model.set_variable('_x','pitch')
        yw =model.set_variable('_x','heading')
        dp =model.set_variable('_x','depth')
        F_LF   =model.set_variable('_u','F_LF')
        F_LB   =model.set_variable('_u','F_LB')
        F_RF   =model.set_variable('_u','F_RF')
        F_RB   =model.set_variable('_u','F_RB')
        F_angle=model.set_variable('_u','F_angle')
        v_tvp  =model.set_variable('_tvp','v_vel')
        w_tvp  =model.set_variable('_tvp','w_vel')
        tgt_x  =model.set_variable('_tvp','tgt_x')
        tgt_y  =model.set_variable('_tvp','tgt_y')
        tgt_z  =model.set_variable('_tvp','tgt_z')
        tgt_yaw=model.set_variable('_tvp','tgt_yaw')
        tgt_pt =model.set_variable('_tvp','tgt_pitch')
        Q_x_t  =model.set_variable('_tvp','Q_x')
        Q_y_t  =model.set_variable('_tvp','Q_y')
        Q_z_t  =model.set_variable('_tvp','Q_depth')
        Q_yaw_t=model.set_variable('_tvp','Q_yaw')
        Q_pt_t =model.set_variable('_tvp','Q_pitch')
        R_thr_t=model.set_variable('_tvp','R_thr')
        R_ang_t=model.set_variable('_tvp','R_ang')
        B = B_ANGLE
        model.set_rhs('x_position',
            uv*ca.cos(pt)*ca.cos(yw)-v_tvp*ca.sin(yw))
        model.set_rhs('y_position',
            uv*ca.cos(pt)*ca.sin(yw)+v_tvp*ca.cos(yw))
        model.set_rhs('u_vel',
            (1/MASS)*(-24.3748*uv*ca.fabs(uv)+3.255*uv
                +0.1*rv*ca.fabs(rv)
                -0.5*uv*uv*0.15*float(np.cos(B))*0.2
                -F_LF*ca.sin(F_angle)-F_LB*ca.sin(B)
                -F_RF*ca.sin(F_angle)-F_RB*ca.sin(B)))
        model.set_rhs('r_vel',
            (1/IZ)*(-44.0*rv*ca.fabs(rv)-4.88*rv
                -2.0*rv*ca.fabs(uv*uv)
                +F_LF*0.613*ca.sin(F_angle)+F_LB*0.613*ca.sin(B)
                -F_RF*0.613*ca.sin(F_angle)-F_RB*0.613*ca.sin(B)))
        model.set_rhs('pitch_rate',
            (1/IY)*(-0.5*1000*uv*uv*0.15*ca.cos(F_angle)*0.3*0.5
                +pt*ca.fabs(pt)+pt+(F_LF+F_RF)*0.3*ca.cos(F_angle)))
        model.set_rhs('pitch',   pr)
        model.set_rhs('heading', rv)
        model.set_rhs('depth',  -ca.sin(pt)*uv-w_tvp*ca.sin(pt))
        model.setup()
        self.model = model

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=13, t_step=0.1,
            state_discretization='collocation',
            collocation_type='radau', collocation_deg=2,
            collocation_ni=2, store_full_solution=True)
        for nm in ['F_LF','F_LB','F_RF','F_RB','F_angle']:
            mpc.scaling['_u',nm]=10.0
        for nm in ['u_vel','r_vel','heading','depth',
                   'pitch','pitch_rate','x_position','y_position']:
            mpc.scaling['_x',nm]=10.0

        xe=xp-tgt_x; ye=yp-tgt_y; ze=dp-tgt_z
        psi_e=ca.atan2(ca.sin(yw-tgt_yaw),ca.cos(yw-tgt_yaw))
        the_e=pt-tgt_pt; k=2.0
        sig = lambda e: 1/(1+ca.exp(-k*ca.fabs(e)))
        lterm=(Q_x_t*xe**2+Q_y_t*ye**2+Q_z_t*ze**2
               +Q_yaw_t*psi_e**2+Q_pt_t*the_e**2
               +R_thr_t*(F_LF**2+F_LB**2+F_RF**2+F_RB**2)
               +R_ang_t*F_angle**2)
        mterm=(20*Q_x_t*sig(xe)*xe**2+20*Q_y_t*sig(ye)*ye**2
               +2*Q_z_t*sig(ze)*ze**2
               +2*Q_yaw_t*sig(psi_e)*psi_e**2)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.bounds['lower','_x','u_vel']=-2.0
        mpc.bounds['upper','_x','u_vel']= 2.0
        for nm in ['F_LF','F_LB','F_RF','F_RB']:
            mpc.bounds['lower','_u',nm]=-100.0
            mpc.bounds['upper','_u',nm]= 100.0
        mpc.bounds['lower','_u','F_angle']=0.80
        mpc.bounds['upper','_u','F_angle']=2.20
        mpc.set_param(nlpsol_opts={
            'ipopt.print_level':0,'ipopt.sb':'yes',
            'print_time':0,
            'ipopt.warm_start_init_point':'yes',
            'ipopt.tol':1e-4,'ipopt.max_iter':30})

        tvp_tmpl = mpc.get_tvp_template()

        def tvp_fun(t_now):
            t_now = float(np.array(t_now).flat[0])
            if self.enable_training:
                self._actor_update_weights()
            ctrl_est = self._prev_controls_for_gp or [0,0,0,0,1.47]
            if self.gp.is_fitted:
                self.gp_mean, self.gp_std = self.gp.predict(
                    self._get_state_for_gp(), ctrl_est)
            else:
                self.gp_mean = np.zeros(3)
                self.gp_std  = np.ones(3)*0.5
            do_comp = (self.gp.should_compensate() and
                       float(np.mean(self.gp_std[:2]))
                       < self.gp.std_threshold)
            for k in range(14):
                t = float(t_now+k*0.1)
                xr = -6.0+8.0*math.cos(OMEGA*t)
                yr =  1.0+8.0*math.sin(OMEGA*t)
                zr = max(-1.0-0.15*SPEED_SCALE*t,-12.0)
                t_los = t+LOS_L/(abs(self.u_vel)+1e-3)
                x_los = -6.0+8.0*math.cos(OMEGA*t_los)
                y_los =  1.0+8.0*math.sin(OMEGA*t_los)
                beta  = math.atan2(self.v_vel,self.u_vel+1e-5)
                psi_k = math.atan2(y_los-self.y,x_los-self.x)-beta
                depth_err_k = self.depth-zr
                pt_ref = float(np.clip(
                    math.atan2(-depth_err_k,1.0),-0.4,0.4))
                if k==0 and do_comp:
                    xr_c = xr-float(self.gp_mean[0])
                    yr_c = yr-float(self.gp_mean[1])
                    zr_c = zr-float(self.gp_mean[2])
                else:
                    xr_c,yr_c,zr_c = xr,yr,zr
                tvp_tmpl['_tvp',k,'tgt_x']    = float(xr_c)
                tvp_tmpl['_tvp',k,'tgt_y']    = float(yr_c)
                tvp_tmpl['_tvp',k,'tgt_z']    = float(zr_c)
                tvp_tmpl['_tvp',k,'tgt_yaw']  = float(psi_k)
                tvp_tmpl['_tvp',k,'tgt_pitch']= pt_ref
                tvp_tmpl['_tvp',k,'v_vel']    = float(self.v_vel)
                tvp_tmpl['_tvp',k,'w_vel']    = float(self.w_vel)
                tvp_tmpl['_tvp',k,'Q_x']      = float(self.Q_x)
                tvp_tmpl['_tvp',k,'Q_y']      = float(self.Q_y)
                tvp_tmpl['_tvp',k,'Q_depth']  = float(self.Q_depth)
                tvp_tmpl['_tvp',k,'Q_yaw']    = float(self.Q_yaw)
                tvp_tmpl['_tvp',k,'Q_pitch']  = float(self.Q_pitch)
                tvp_tmpl['_tvp',k,'R_thr']    = float(self.R_thr)
                tvp_tmpl['_tvp',k,'R_ang']    = float(self.R_ang)
                if k==1: self.time_log = float(t_now)
            return tvp_tmpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        self.mpc = mpc
        self.estimator = do_mpc.estimator.StateFeedback(model)
        self.x0 = np.zeros((8,1)); self.x0[7,0]=-1.0
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()

    # ----------------------------------------------------------
    def _get_state(self):
        t_los = self.time_log+LOS_L/(abs(self.u_vel)+1e-3)
        x_los = -6.0+8.0*math.cos(OMEGA*t_los)
        y_los =  1.0+8.0*math.sin(OMEGA*t_los)
        beta  = math.atan2(self.v_vel,self.u_vel+1e-5)
        tgt_yaw = math.atan2(y_los-self.y,x_los-self.x)-beta
        return np.array([
            self.x,self.y,self.u_vel,self.r_vel,
            self.pitch_rate,self.pitch,self.yaw,self.depth,
            self.x-self.x_ref, self.y-self.y_ref,
            self.depth-self.z_ref,
            normalize_angle(self.yaw-tgt_yaw)],
            dtype=np.float32)

    def _get_state_for_gp(self):
        return [self.x,self.y,self.u_vel,self.r_vel,
                self.pitch_rate,self.pitch,self.yaw,self.depth]

    # ----------------------------------------------------------
    def _actor_update_weights(self):
        state = self._get_state()
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            w = self.actor(st).cpu().numpy()[0]

        noise_scale = max(0.02, 0.1*np.exp(-self.step_count/500.0))
        w += np.random.normal(0, noise_scale, size=w.shape)

        steps_sw = max(0, self.step_count - self.warmup_steps)

        alpha_xy    = min(0.15, 0.02 + steps_sw * 0.0005)
        alpha_depth = min(0.30, 0.05 + steps_sw * 0.001)
        alpha_other = min(0.10, 0.02 + steps_sw * 0.0003)

        self.Q_x     = (1-alpha_xy)    * self.Q_x     + alpha_xy    * w[0]
        self.Q_y     = (1-alpha_xy)    * self.Q_y     + alpha_xy    * w[1]
        self.Q_depth = (1-alpha_depth) * self.Q_depth + alpha_depth * w[2]
        self.Q_yaw   = (1-alpha_other) * self.Q_yaw   + alpha_other * w[3]
        self.Q_pitch = (1-alpha_other) * self.Q_pitch + alpha_other * w[4]
        self.R_thr   = (1-alpha_other) * self.R_thr   + alpha_other * w[5]
        self.R_ang   = (1-alpha_other) * self.R_ang   + alpha_other * w[6]

        self.Q_x     = float(np.clip(self.Q_x,     0.5,  10.0))
        self.Q_y     = float(np.clip(self.Q_y,     0.5,  10.0))
        self.Q_depth = float(np.clip(self.Q_depth, 8.0,  20.0))
        self.Q_yaw   = float(np.clip(self.Q_yaw,   0.5,   5.0))
        self.Q_pitch = float(np.clip(self.Q_pitch, 0.1,   2.0))
        self.R_thr   = float(np.clip(self.R_thr,  0.0001, 0.01))
        self.R_ang   = float(np.clip(self.R_ang,   0.01,  0.5))

        self.prev_weights = np.array([
            self.Q_x, self.Q_y, self.Q_depth,
            self.Q_yaw, self.Q_pitch,
            self.R_thr, self.R_ang], dtype=np.float32)

    # ----------------------------------------------------------
    def _train(self):
        if len(self.buffer) < self.batch_size: return
        s,w,r,ns,d = self.buffer.sample(self.batch_size)
        S  = torch.FloatTensor(s).to(self.device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        NS = torch.FloatTensor(ns).to(self.device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        V_curr = self.critic(S)
        with torch.no_grad():
            V_tgt = R+self.gamma*(1-D)*self.tgt_critic(NS)
        c_loss = nn.MSELoss()(V_curr,V_tgt)
        self.c_opt.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(),1.0)
        self.c_opt.step()
        with torch.no_grad():
            td_error=(R+self.gamma*(1-D)*self.tgt_critic(NS)
                      -self.critic(S))
            adv = td_error/(td_error.std()+1e-6)
            adv = adv.clamp(-3.0,3.0)
        w_out = self.actor(S)
        w_norm=((w_out-self.actor.w_lb)/
                (self.actor.w_ub-self.actor.w_lb+1e-8)
                ).clamp(1e-3,1-1e-3)
        entropy=-(w_norm*torch.log(w_norm+1e-8)
                  +(1-w_norm)*torch.log(1-w_norm+1e-8)).mean()
        a_loss=-(adv*w_norm).mean()-0.05*entropy
        self.a_opt.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)
        self.a_opt.step()
        soft_update(self.tgt_critic,self.critic,self.tau)
        self.a_loss_hist.append(float(a_loss.item()))
        self.c_loss_hist.append(float(c_loss.item()))
        if self.step_count % 50 == 0:
            total_grad = sum(
                p.grad.abs().sum().item()
                for p in self.actor.parameters()
                if p.grad is not None)
            self.get_logger().info(
                f"  grad={total_grad:.6f}  "
                f"a_loss={a_loss.item():.4f}  "
                f"V={V_curr.mean().item():.3f}  "
                f"adv_std={adv.std().item():.4f}")

    # ----------------------------------------------------------
    def pose_callback(self, msg):
        if not hasattr(self,'estimator'): return

        p = msg.pose.pose.position
        self.x, self.y, self.depth = p.x, p.y, p.z

        t = msg.twist.twist
        self.u_vel=t.linear.x; self.v_vel=t.linear.y
        self.w_vel=t.linear.z; self.r_vel=t.angular.z
        self.pitch_rate=t.angular.y

        q = msg.pose.pose.orientation
        self.roll,self.pitch,self.yaw = euler_from_quaternion(
            [q.x,q.y,q.z,q.w])

        # Ocean current injection (single-step displacement, non-cumulative).
        # MPC nominal model excludes ocean current -> produces residuals -> GP learns to compensate.
        t_now = self.time_log
        cx, cy, cz = self._get_ocean_current(t_now)
        self.x     += cx * DT
        self.y     += cy * DT
        self.depth += cz * DT

        self.x_ref = -6.0+8.0*math.cos(OMEGA*t_now)
        self.y_ref =  1.0+8.0*math.sin(OMEGA*t_now)
        self.z_ref = max(-1.0-0.15*SPEED_SCALE*t_now,-12.0)

        err = math.sqrt((self.x-self.x_ref)**2+
                        (self.y-self.y_ref)**2+
                        (self.depth-self.z_ref)**2)
        self.err_hist.append(err)

        if (self.enable_training and
                self.last_state is not None):
            curr = self._get_state()
            rwd  = compute_reward(curr,self.gp_std)
            self.buffer.push(
                self.last_state,
                self.last_weights if self.last_weights is not None
                else np.zeros(7),
                rwd, curr, False)
            self.reward = rwd

        y_meas = np.array([
            self.x,self.y,self.u_vel,self.r_vel,
            self.pitch_rate,self.pitch,self.yaw,self.depth
        ]).reshape(-1,1)
        self.x0 = self.estimator.make_step(y_meas)

    # ----------------------------------------------------------
    def timer_callback(self):
        self.step_count += 1
        self.enable_training=(self.step_count>=self.warmup_steps)
        self.last_state = self._get_state()
        self.last_weights = np.array([
            self.Q_x,self.Q_y,self.Q_depth,self.Q_yaw,
            self.Q_pitch,self.R_thr,self.R_ang],dtype=np.float32)

        t0 = time.time()
        try: u0 = self.mpc.make_step(self.x0)
        except Exception as e:
            self.get_logger().warn(f"MPC failed: {e}"); return
        self.solve_times.append((time.time()-t0)*1000)
        if u0 is None: return

        F_LF=float(u0[0,0]); F_LB=float(u0[1,0])
        F_RF=float(u0[2,0]); F_RB=float(u0[3,0])
        ang =float(u0[4,0])

        def tm(f):
            fc=float(np.clip(f,-100,100))*5.0
            return float(np.sign(fc)*math.sqrt(abs(fc)*7.5))
        lf=tm(F_LF); lb=tm(F_LB)
        rf=tm(F_RF); rb=tm(F_RB)
        ang=float(np.clip(ang,0.80,2.20))

        self.lf_pub.publish(Float64(data=lf))
        self.lb_pub.publish(Float64(data=lb))
        self.rf_pub.publish(Float64(data=rf))
        self.rb_pub.publish(Float64(data=rb))
        self.front_pub.publish(Float64(data=ang))
        self.behind_pub.publish(Float64(data=1.47))

        # GP data collection (fixed: use nominal_step as baseline)
        s_now = self._get_state_for_gp()
        if self._prev_state_for_gp is not None:
            s_nom = nominal_step(
                self._prev_state_for_gp,
                self._prev_controls_for_gp or [0,0,0,0,1.47])
            self.gp.push_step(
                self._prev_state_for_gp,
                self._prev_controls_for_gp or [0,0,0,0,1.47],
                s_now,   # actual state (includes ocean current)
                s_nom)   # nominal prediction (no ocean current)

        self._prev_state_for_gp    = s_now
        self._prev_controls_for_gp = [F_LF,F_LB,F_RF,F_RB,ang]
        self._prev_angle           = ang

        self.x0 = np.array([
            self.x,self.y,self.u_vel,self.r_vel,
            self.pitch_rate,self.pitch,self.yaw,self.depth
        ]).reshape(-1,1)

        self.t_hist.append(time.time())
        self.x_hist.append(self.x); self.y_hist.append(self.y)
        self.depth_hist.append(self.depth)
        self.xr_hist.append(self.x_ref)
        self.yr_hist.append(self.y_ref)
        self.zr_hist.append(self.z_ref)
        self.roll_hist.append(self.roll)
        self.pitch_hist.append(self.pitch)
        self.heading_hist.append(self.yaw)
        self.lf_hist.append(F_LF); self.lb_hist.append(F_LB)
        self.rf_hist.append(F_RF); self.rb_hist.append(F_RB)
        self.front_angle_hist.append(ang)
        self.Q_x_hist.append(self.Q_x); self.Q_y_hist.append(self.Q_y)
        self.Q_depth_hist.append(self.Q_depth)
        self.Q_yaw_hist.append(self.Q_yaw)
        self.R_thr_hist.append(self.R_thr)
        self.gp_mean_x_hist.append(float(self.gp_mean[0]))
        self.gp_mean_y_hist.append(float(self.gp_mean[1]))
        self.gp_mean_z_hist.append(float(self.gp_mean[2]))
        self.gp_std_x_hist.append(float(self.gp_std[0]))
        self.gp_std_y_hist.append(float(self.gp_std[1]))
        self.gp_std_z_hist.append(float(self.gp_std[2]))
        self.reward_hist.append(float(self.reward))

        if (self.enable_training and self.step_count%2==0):
            for _ in range(3): self._train()

        if self.step_count%10==0 and self.err_hist:
            avg_t = np.mean(self.solve_times[-10:])
            mode  = 'TRAIN' if self.enable_training else 'WARMUP'
            self.get_logger().info(
                f"Step {self.step_count:5d} | "
                f"err={self.err_hist[-1]:.3f}m | "
                f"Qx={self.Q_x:.2f} Qd={self.Q_depth:.2f} | "
                f"GP_std=({self.gp_std[0]:.3f},"
                f"{self.gp_std[1]:.3f}) "
                f"GP_mu=({self.gp_mean[0]:+.3f},"
                f"{self.gp_mean[1]:+.3f}) "
                f"fit={self.gp.fit_count} | "
                f"solve={avg_t:.1f}ms | [{mode}]")

        if self.step_count%50==0 and len(self.solve_times)>=50:
            rt = self.solve_times[-50:]
            self.get_logger().info(
                f"MPC avg={np.mean(rt):.1f}ms "
                f"max={np.max(rt):.1f}ms")

    # ----------------------------------------------------------
    def save_all(self):
        err = np.array(self.err_hist) if self.err_hist else np.array([0])
        n = len(err)
        e1=err[:n//3]; e2=err[n//3:2*n//3]; e3=err[2*n//3:]
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  GP-AC-MPC  Final Statistics\n"
            f"{'='*60}\n"
            f"  Total steps : {n}\n"
            f"  Mean error  : {np.mean(err):.4f} m\n"
            f"  RMSE        : {np.sqrt(np.mean(err**2)):.4f} m\n"
            f"  Max error   : {np.max(err):.4f} m\n"
            f"  Early  (0~1/3)  : {np.mean(e1):.4f} m\n"
            f"  Middle (1/3~2/3): {np.mean(e2):.4f} m\n"
            f"  Late   (2/3~end): {np.mean(e3):.4f} m\n"
            f"  GP fit count: {self.gp.fit_count}\n"
            f"  GP data size: {len(self.gp.X_buf)}\n"
            f"  Solve avg: {np.mean(self.solve_times):.1f}ms\n"
            f"{'='*60}")
        self.get_logger().info("Saving...")
        save_training_log(
            self.data_file,
            self.x_hist,self.y_hist,self.depth_hist,
            self.xr_hist,self.yr_hist,self.zr_hist,
            self.t_hist,self.roll_hist,self.pitch_hist,
            self.heading_hist,self.lf_hist,self.lb_hist,
            self.rf_hist,self.rb_hist,self.front_angle_hist,
            self.a_loss_hist,self.c_loss_hist,
            self.Q_x_hist,self.Q_y_hist,
            self.Q_depth_hist,self.Q_yaw_hist,
            self.R_thr_hist,
            self.gp_mean_x_hist,self.gp_mean_y_hist,self.gp_mean_z_hist,
            self.gp_std_x_hist,self.gp_std_y_hist,self.gp_std_z_hist,
            self.reward_hist,
            len(self.buffer))
        model_file = self.data_file.replace('.txt','.pth')
        torch.save({
            'actor':self.actor.state_dict(),
            'critic':self.critic.state_dict(),
            'step':self.step_count}, model_file)
        self.get_logger().info(f"Saved -> {model_file}")

    # ----------------------------------------------------------
    def plot_3d(self):
        if len(self.x_hist) < 2: return
        try:
            x=np.array(self.x_hist); y=np.array(self.y_hist)
            z=np.array(self.depth_hist)
            xr=np.array(self.xr_hist); yr=np.array(self.yr_hist)
            zr=np.array(self.zr_hist)
            fig=plt.figure(figsize=(12,9))
            ax=fig.add_subplot(111,projection='3d')
            ax.plot3D(xr,yr,zr,'r--',lw=2.5,label='Reference')
            ax.plot3D(x,y,z,'b-',lw=2.0,label='GP-AC-MPC')
            ax.set_xlabel('X (m)',fontsize=13)
            ax.set_ylabel('Y (m)',fontsize=13)
            ax.set_zlabel('Depth (m)',fontsize=13)
            ax.set_title('3D Trajectory (GP-AC-MPC)',fontsize=15)
            ax.legend(fontsize=13)
            ax.view_init(elev=35,azim=135)
            plt.tight_layout()
            out=self.data_file.replace('.txt','_3D.png')
            plt.savefig(out,dpi=150)
            self.get_logger().info(f"3D -> {out}")
            plt.close()

            if len(self.err_hist) < 10: return
            fig,axes=plt.subplots(2,2,figsize=(14,10))
            t=np.arange(len(self.err_hist))*0.1
            axes[0,0].plot(t,self.err_hist,'b-',lw=1)
            axes[0,0].axhline(np.mean(self.err_hist),
                color='r',lw=1,linestyle='--',
                label=f'Mean={np.mean(self.err_hist):.3f}m')
            axes[0,0].set(xlabel='Time(s)',ylabel='Error(m)',
                title='Tracking Error')
            axes[0,0].legend(); axes[0,0].grid(True)
            n=len(self.Q_x_hist); tq=np.arange(n)*0.1
            axes[0,1].plot(tq,self.Q_x_hist,label='Q_x')
            axes[0,1].plot(tq,self.Q_y_hist,label='Q_y')
            axes[0,1].plot(tq,self.Q_depth_hist,label='Q_depth')
            axes[0,1].set(xlabel='Time(s)',ylabel='Weight',
                title='AC Weights')
            axes[0,1].legend(); axes[0,1].grid(True)
            ng=len(self.gp_mean_x_hist); tg=np.arange(ng)*0.1
            axes[1,0].plot(tg,self.gp_mean_x_hist,label='mu_x')
            axes[1,0].plot(tg,self.gp_mean_y_hist,label='mu_y')
            axes[1,0].axhline(0,color='gray',lw=0.6,linestyle=':')
            axes[1,0].set(xlabel='Time(s)',ylabel='GP Mean(m)',
                title='GP Feedforward Compensation')
            axes[1,0].legend(); axes[1,0].grid(True)
            axes[1,1].plot(tg,self.gp_std_x_hist,label='sigma_x')
            axes[1,1].plot(tg,self.gp_std_y_hist,label='sigma_y')
            axes[1,1].axhline(self.gp.std_threshold,
                color='r',lw=1,linestyle='--',label='Threshold')
            axes[1,1].set(xlabel='Time(s)',ylabel='GP Std(m)',
                title='GP Uncertainty')
            axes[1,1].legend(); axes[1,1].grid(True)
            plt.suptitle('GP-AC-MPC Summary',fontsize=15)
            plt.tight_layout()
            out2=self.data_file.replace('.txt','_summary.png')
            plt.savefig(out2,dpi=150)
            self.get_logger().info(f"Summary -> {out2}")
            plt.close()
        except Exception as e:
            self.get_logger().error(f"Plot failed: {e}")
            plt.close('all')


# ============================================================
#  Entry Point
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = GPACMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try: node.save_all()
        except Exception as e:
            node.get_logger().error(f"Save failed: {e}")
        try: node.plot_3d()
        except Exception as e:
            node.get_logger().error(f"Plot failed: {e}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()