import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import time
from .models.model import ED_LSTM_2,ED_LSTM_3
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise



class Flyer():
    def __init__(self, data_form):
        self.data_form = data_form
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.dat_min = np.load(cur_dir + '/data/' + self.data_form + '/npy/dat_min.npy')
        self.dat_mm = np.load(cur_dir + '/data/' + self.data_form + '/npy/dat_mm.npy')
        self.set_model(f"{cur_dir}/output/{self.data_form}.pkl")


    def set_model(self, model_dir):
        model = ED_LSTM_3()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_dir))
        model.to(self.device)
        model.eval()
        self.model = model
    
    @classmethod
    def calculate_v(cls,position,time_step):
        vel = position.copy()
        vel[0] = (position[1] - position[0])/(time_step[1] - time_step[0])
        vel[1] = (position[2] - position[0]) / (time_step[2] - time_step[0])
        vel[2] = (position[4] - position[0]) / (time_step[4] - time_step[0])
        for i in range(3,position.shape[0]-3):
            vel[i] = (position[i+3] - position[i-3])/(time_step[i+3] - time_step[i-3])
        vel[-3] = (position[-1] - position[-5]) / (time_step[-1] - time_step[-5])
        vel[-2] = (position[-1] - position[-3]) / (time_step[-1] - time_step[-3])
        vel[-1] = (position[-1] - position[-2]) / (time_step[-1] - time_step[-2])
        return vel

    @classmethod
    def calculate_a(cls,vel,time_step):
        acc = vel.copy()
        acc[0] = (vel[1] - vel[0])/(time_step[1] - time_step[0])
        acc[1] = (vel[2] - vel[0]) / (time_step[2] - time_step[0])
        acc[2] = (vel[4] - vel[0]) / (time_step[4] - time_step[0])
        for i in range(3,vel.shape[0]-3):
            acc[i] = (vel[i+3] - vel[i-3])/(time_step[i+3] - time_step[i-3])
        acc[-3] = (vel[-1] - vel[-5]) / (time_step[-1] - time_step[-5])
        acc[-2] = (vel[-1] - vel[-3]) / (time_step[-1] - time_step[-3])
        acc[-1] = (vel[-1] - vel[-2]) / (time_step[-1] - time_step[-2])
        return acc

    @classmethod
    def calculate_ang(cls,quater):
        r = R.from_quat(quater)
        r_euler = r.as_euler('zyx', degrees=False)
        for k in range(3):
            bias = 0
            for i in range(1,len(r_euler)):
                r_euler[i][k] += bias
                if (r_euler[i-1][k]*r_euler[i][k])<0:
                    if abs(r_euler[i - 1][k] - r_euler[i][k]) > np.pi:
                        if r_euler[i-1][k]-r_euler[i][k]>0:
                            bias += np.pi * 2
                        else:
                            bias -= np.pi * 2
                        r_euler[i][k] += bias
        return r_euler

    @classmethod
    def calculate_angv(cls,quater,time_step):
        angv = quater.copy()
        angv[0] = (quater[1] - quater[0])/(time_step[1] - time_step[0])
        angv[1] = (quater[2] - quater[0]) / (time_step[2] - time_step[0])
        angv[2] = (quater[4] - quater[0]) / (time_step[4] - time_step[0])
        for i in range(3,quater.shape[0]-3):
            angv[i] = (quater[i+3] - quater[i-3])/(time_step[i+3] - time_step[i-3])
        angv[-3] = (quater[-1] - quater[-5]) / (time_step[-1] - time_step[-5])
        angv[-2] = (quater[-1] - quater[-3]) / (time_step[-1] - time_step[-3])
        angv[-1] = (quater[-1] - quater[-2]) / (time_step[-1] - time_step[-2])
        return angv

    @classmethod
    def calculate_anga(cls,angv,time_step):
        anga = angv.copy()
        anga[0] = (angv[1] - angv[0])/(time_step[1] - time_step[0])
        anga[1] = (angv[2] - angv[0]) / (time_step[2] - time_step[0])
        anga[2] = (angv[4] - angv[0]) / (time_step[4] - time_step[0])
        for i in range(3,angv.shape[0]-3):
            anga[i] = (angv[i+3] - angv[i-3])/(time_step[i+3] - time_step[i-3])
        anga[-3] = (angv[-1] - angv[-5]) / (time_step[-1] - time_step[-5])
        anga[-2] = (angv[-1] - angv[-3]) / (time_step[-1] - time_step[-3])
        anga[-1] = (angv[-1] - angv[-2]) / (time_step[-1] - time_step[-2])
        return anga

    @classmethod
    def calculate_dq(cls,quat,time_step):
        dquat = quat.copy()
        dquat[0] = (quat[1] - quat[0])/(time_step[1] - time_step[0])
        dquat[1] = (quat[2] - quat[0]) / (time_step[2] - time_step[0])
        dquat[2] = (quat[4] - quat[0]) / (time_step[4] - time_step[0])
        for i in range(3,quat.shape[0]-3):
            dquat[i] = (quat[i+3] - quat[i-3])/(time_step[i+3] - time_step[i-3])
        dquat[-3] = (quat[-1] - quat[-5]) / (time_step[-1] - time_step[-5])
        dquat[-2] = (quat[-1] - quat[-3]) / (time_step[-1] - time_step[-3])
        dquat[-1] = (quat[-1] - quat[-2]) / (time_step[-1] - time_step[-2])
        return dquat

    @classmethod
    def spine(cls,pos,quat,t):
        x = []
        v = []
        a = []
        for i in range(3):
            p = pos[:, i]
            raw_f = UnivariateSpline(t, p, k=3)
            vec_f = raw_f.derivative(n=1)
            acc_f = raw_f.derivative(n=2)
            x.append(raw_f(t))
            v.append(vec_f(t))
            a.append(acc_f(t))

        qua = []
        dqua = []
        for i in range(4):
            q = quat[:, i]
            raw_f = UnivariateSpline(t, q, k=3)
            vec_f = raw_f.derivative(n=1)
            # acc_f = raw_f.derivative(n=2)
            qua.append(raw_f(t))
            dqua.append(vec_f(t))
            # a.append(acc_f(t))


        return np.vstack((x[0],x[1],x[2])).transpose(),\
            np.vstack((v[0],v[1],v[2])).transpose(),\
            np.vstack((a[0],a[1],a[2])).transpose(),\
            np.vstack((qua[0],qua[1],qua[2],qua[3])).transpose(),\
            np.vstack((dqua[0],dqua[1],dqua[2],dqua[3])).transpose()

    @classmethod
    def pos_vel_filter(cls,x, P, R, Q=0., dt=1.0):
        """ Returns a KalmanFilter which implements a
        constant velocity self.model for a state [x dx].T
        """
        dt = dt
        dt_2 = 0.5 * dt * dt
        kf = KalmanFilter(dim_x=9, dim_z=3)
        kf.x = x  # location and velocity
        kf.F = np.array([[1., 0., 0., dt, 0., 0., dt_2, 0., 0.],
                        [0., 1., 0., 0., dt, 0., 0., dt_2, 0.],
                        [0., 0., 1., 0., 0., dt, 0., 0., dt_2],
                        [0., 0., 0., 1., 0., 0., dt, 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0., dt, 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., dt],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1.]])  # state transition matrix
        kf.H = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1.]]) # Measurement function
        kf.R *= R  # measurement uncertainty
        if np.isscalar(P):
            kf.P *= P  # covariance matrix
        else:
            kf.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        else:
            kf.Q[:] = Q
        return kf

    @classmethod
    def load_data(cls, npz_name) -> np.ndarray:
        npz_data = np.load(npz_name)
        leng = len(npz_data['position'])
        if leng <= 40:
            # print(npz_name)
            print("{} is not a valid data, plz retry with another.".format(npz_name))
            return ValueError
        data_append = np.zeros((leng,18))
        x, v, a, q, dq  = cls.spine(npz_data['position'],npz_data['quaternion'], npz_data['time_step'])
        data_append[:, 17] = npz_data['time_step']
        data_append[:,:3] = x
        data_append[:,3:6] = v
        data_append[:,6:9] = a
        data_append[:,9:13] = q
        data_append[:,13:17] = dq

        # data_append_total[row_num:row_num+leng] = data_append
        # row_num += leng
        # # flight_data.append(data_append[4:-4])
        # flight_data.append(data_append)
        # flight_name.append(npz_name)

        return data_append #, flight_name

    def init_lstm_pre(self, flight_data):
        self.flight_data = flight_data
        x0=flight_data[-1,:9].copy()
        # create the Kalman filter
        R = 0.00001
        P = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
        Q = np.diag(np.ones((9))*0.5)
        dt = 1.0 / 120
        self.kf = self.pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)


    def get_next_pre_data(self) -> np.ndarray:    

        lstm_start = time.time()


        traj_long = (len(self.flight_data) > 30) and 30 or len(self.flight_data)
        flight_pred_inputs = (self.flight_data[traj_long-30:traj_long, :9] - self.dat_min[:9])/self.dat_mm[:9]
        # flight_pred_inputs = scaler.transform(flight_pred[i-30:i, :9])
        seq = torch.FloatTensor(flight_pred_inputs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = (torch.zeros(self.model.nlayer, 1, self.model.hidden_layer_size).to(self.device),
                    torch.zeros(self.model.nlayer, 1, self.model.hidden_layer_size).to(self.device))

            outVal, hidden_, outcov = self.model.forward(seq,hidden)
            outcov = outcov[:,-1,:].view(1,3,3)
            outcovT = torch.transpose(outcov,1,2)
            R = torch.matmul(outcovT,outcov)
            R = R.cpu().numpy()[0]
            xva = outVal.cpu().numpy()[0]
            # xva = scaler.inverse_transform(xva)[-1]
            xva = xva[-1] * self.dat_mm[:9] + self.dat_min[:9]

            self.kf.predict()
            self.kf.update(xva[6:9],R=R)

        lstm_end = time.time()

        # print("LSTM pre time is", lstm_end-lstm_start)
        self.flight_data = np.row_stack((self.flight_data, np.r_[self.kf.x, np.zeros(9)]))
        return self.kf.x

    def get_whole_pre_data(self):
        return self.flight_data
