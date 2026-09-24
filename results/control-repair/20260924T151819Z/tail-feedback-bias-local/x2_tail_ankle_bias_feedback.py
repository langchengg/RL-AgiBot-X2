"""Experimental observation-only torso-bias ankle feedback; no environment/runner."""
import numpy as np

def rotation_x(q):
    s,c=np.sin(q),np.cos(q);return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
def rotation_y(q):
    s,c=np.sin(q),np.cos(q);return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
def rotation_z(q):
    s,c=np.sin(q),np.cos(q);return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def sagittal_tilt_and_rate(gravity,gravity_rate):
    x,_,z=gravity;dx,_,dz=gravity_rate;den=x*x+z*z
    if den<=1e-10:raise ValueError('Sagittal gravity projection is undefined')
    return float(np.arctan2(x,-z)),float((-z*dx+x*dz)/den)

class TailAnkleFeedback:
    """Predict bounded 17D actions, using only the existing149D observation.

    The runner must explicitly set both ankle-pitch full residual scales to
    cap_rad and the configured residual gate to start_s with a0.2s ramp.
    This actor is analytical feedback, not PPO and not a weight transfer.
    """
    def __init__(self,q_lower,q_upper,waist_indices,kp,kd,*,cap_rad=.08,start_s=3.5,
                 episode_timeout_s=20.,joint_velocity_scale=5.,base_angular_scale=2.,theta_des_rad=0.):
        self.lower=np.asarray(q_lower,float);self.upper=np.asarray(q_upper,float)
        self.waist=np.asarray(waist_indices,int);self.kp=float(kp);self.kd=float(kd)
        self.cap=float(cap_rad);self.start=float(start_s);self.timeout=float(episode_timeout_s)
        self.theta_des=float(theta_des_rad)
        if not np.isfinite(self.theta_des) or not -.15<=self.theta_des<=0.:raise ValueError('Invalid explicit torso tilt bias')
        self.joint_scale=float(joint_velocity_scale);self.angular_scale=float(base_angular_scale)
        if self.lower.shape!=(31,) or self.upper.shape!=(31,) or np.any(self.lower>=self.upper):raise ValueError('Expected31 joint bounds')
        if self.waist.shape!=(3,) or len(set(self.waist))!=3 or np.any(self.waist<0) or np.any(self.waist>=31):raise ValueError('Expected waist yaw,pitch,roll indices')
        if not np.isfinite([self.kp,self.kd,self.cap,self.start,self.timeout,self.joint_scale,self.angular_scale]).all() or min(self.kp,self.kd)<0 or self.cap<=0:raise ValueError('Invalid feedback settings')
    def signals(self,observation):
        obs=np.asarray(observation,dtype=float)
        if obs.shape!=(149,) or not np.isfinite(obs).all():raise ValueError('Expected finite149D observation')
        q=(self.lower+self.upper)/2+obs[:31]*(self.upper-self.lower)/2
        dy,dp,dr=(obs[31:62]*self.joint_scale)[self.waist]
        yaw,pitch,roll=q[self.waist];Z=rotation_z(yaw);Y=rotation_y(pitch);Q=Z@Y@rotation_x(roll)
        axes=np.column_stack([np.array([0.,0.,1.]),Z@np.array([0.,1.,0.]),Z@Y@np.array([1.,0.,0.])])
        pelvis_gravity=obs[62:65];torso_gravity=Q.T@pelvis_gravity
        torso_omega=Q.T@(obs[68:71]*self.angular_scale+axes@np.array([dy,dp,dr]))
        gravity_rate=-np.cross(torso_omega,torso_gravity)
        tilt,rate=sagittal_tilt_and_rate(torso_gravity,gravity_rate)
        desired_pelvis_gravity=-Q[:,2]
        return dict(elapsed_s=float(obs[148]*self.timeout),torso_tilt_rad=tilt,
                    torso_tilt_rate_rad_s=rate,torso_gravity=torso_gravity,
                    pelvis_tilt_rad=float(np.arctan2(pelvis_gravity[0],-pelvis_gravity[2])),
                    desired_pelvis_tilt_rad=float(np.arctan2(desired_pelvis_gravity[0],-desired_pelvis_gravity[2])))
    def predict(self,observation,deterministic=True):
        obs=np.asarray(observation,dtype=float)
        if obs.shape!=(149,) or not np.isfinite(obs).all():raise ValueError('Expected finite149D observation')
        action=np.zeros(17,dtype=np.float32)
        if (self.kp==0 and self.kd==0) or obs[148]*self.timeout<self.start:return action,None
        s=self.signals(obs);requested=float(np.clip(self.kp*(s['torso_tilt_rad']-self.theta_des)+self.kd*s['torso_tilt_rate_rad_s'],-self.cap,self.cap))
        action[[4,10]]=requested/self.cap
        return action,None
