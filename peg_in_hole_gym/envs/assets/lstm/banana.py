import torch
import torch.nn as nn
import datetime
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import time
from .models.model import ED_LSTM_2,ED_LSTM_3
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# object_name = 'Banana'
# augdata_path = './data/npz_aug_1113/'+object_name+'/'

# npy_path = augdata_path+'npy/'
# dat_max = np.load(npy_path+'dat_max.npy')
# dat_min = np.load(npy_path+'dat_min.npy')
# dat_mm = np.load(npy_path+'dat_mm.npy')

# npz_train = augdata_path + 'train/'
# train_list = os.listdir(npz_train)

# npz_test = augdata_path + 'test/'
# test_list = os.listdir(npz_test)
# npz_list = train_list + test_list

# test_list.sort()


def calculate_v(position,time_step):
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

def calculate_a(vel,time_step):
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

def calculate_ang(quater):
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

def calculate_angv(quater,time_step):
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

def calculate_anga(angv,time_step):
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

def calculate_dq(quat,time_step):
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

def spine(pos,quat,t):
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

# test 25728 & train 227376
# Bag       train  98760 & test 12860 = 111620   i=1120
# Banana    train 134900 & test 16160 = 151060   i=1560
# Bottle13  train 379920 & test 41920 = 421840   i=4040
# Bottle23  train 177100 & test 21680 = 198780   i=2060

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    dt = dt
    dt_2 = 0.5 * dt * dt
    kf = KalmanFilter(dim_x=9, dim_z=3)
    #     print(kf)
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
    # kf.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                  [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #                  [0., 0., 1., 0., 0., 0., 0., 0., 0.]]) # Measurement function
    kf.H = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1.]]) # Measurement function
    # kf.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                  [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #                  [0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #                  [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    #                  [0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #                  [0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #                  [0., 0., 0., 0., 0., 0., 1., 0., 0.],
    #                  [0., 0., 0., 0., 0., 0., 0., 1., 0.],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 1.]])  # Measurement function
    kf.R *= R  # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P  # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    #     print(kf)
    return kf

def load_data(npz_name):
    print("Loading data {}...".format(npz_name))
    # flight_data = []
    # flight_name = []
    # # test 25728 & train 227376
    # row_num = 0
    # test_list = os.listdir(test_dir)
    # data_append_total = np.zeros((41920,18))
    # for i,npz_name in enumerate(test_list):
    npz_data = np.load(npz_name)
    leng = len(npz_data['position'])
    if leng <= 40:
        # print(npz_name)
        print("{} is not a valid data, plz retry with another.".format(npz_name))
        return ValueError
    data_append = np.zeros((leng,18))
    x, v, a, q, dq  = spine(npz_data['position'],npz_data['quaternion'], npz_data['time_step'])
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


def get_lstm_pre_data(flight_data, flight_name, dat_min, dat_mm, pred_start=40, verbose=False):
    model = ED_LSTM_3()
    device = torch.device("cuda:0")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # model = torch.load(cur_dir + '/output/model_best0.0003_0.052.pkl',map_location=device)
    model.load_state_dict(torch.load(cur_dir + "/output/banana.pkl"))
    model.to(device)
    model.eval()

    # pred_error = []
    # cal_error = []
    # pred_time = []
        # pred_length = 70
    # for k in range(len(flight_data)):
        # num = 458
        # k = num
    # if verbose:
    #     print('-------------------------')
    #     print('Traj:{}'.format(k))
    #     print('Name:{}'.format(flight_name[k]))
    flight_origin = flight_data
    flight_cal = flight_origin.copy()
    flight_pred = flight_origin.copy()

    # if len(flight_origin)-30 <= pred_length:
    #     continue
    # pred_start = len(flight_origin) - pred_length
    # pred_start = 40
    flight_pred[pred_start:,:9] = 0
    # t_step = (flight_cal[-1,17]-flight_cal[0,17])/(len(flight_cal)-1)
    t_step = 1/120

    if verbose:
        print('pred_len:{}'.format(len(flight_cal)-pred_start))
    for i in range(pred_start,len(flight_cal)):
        flight_cal[i][0:3] = flight_cal[i-1][0:3] + flight_cal[i-1][3:6] *t_step
        flight_cal[i][3:6] = flight_cal[i-1][3:6] + np.array([0,-9.8,0]) *t_step
        flight_cal[i][6:9] = [0, -9.8, 0]
    dist_x = np.linalg.norm(flight_cal[len(flight_cal) - 1][0:3] - flight_origin[len(flight_cal) - 1][0:3])
    dist_v = np.linalg.norm(flight_cal[len(flight_cal) - 1][3:6] - flight_origin[len(flight_cal) - 1][3:6])
    dist_a = np.linalg.norm(flight_cal[len(flight_cal) - 1][6:9] - flight_origin[len(flight_cal) - 1][6:9])
    # cal_error.append(np.array([dist_x,dist_v,dist_a]))
    
    if verbose:
        print('cal_error:{}'.format(dist_x))

    lstm_start = time.time()

    x0=[]
    try:
        x0 = flight_pred[pred_start-1,:9]
    except:
        print(flight_name)
    # create the Kalman filter
    R = 0.00001
    P = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
    Q = np.diag(np.ones((9))*0.5)
    dt = 1.0 / 120
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)
    # print(kf)
    # run the kalman filter and store the results
    xs, cov = [], []

    for i in range(pred_start,len(flight_pred)):

        # flight_pred_inputs = scaler_xvq.transform(\
        #     np.hstack((flight_pred[i-20:i, :6], flight_pred[i-20:i, 9:13])))
        # flight_pred_inputs = scaler_xvq.transform(\
        #     np.hstack((flight_pred[:i, :6], flight_pred[:i, 9:13])))
        flight_pred_inputs = (flight_pred[i-30:i, :9] - dat_min[:9])/dat_mm[:9]
        # flight_pred_inputs = scaler.transform(flight_pred[i-30:i, :9])
        seq = torch.FloatTensor(flight_pred_inputs).unsqueeze(0).to(device)
        with torch.no_grad():
            hidden = (torch.zeros(model.nlayer, 1, model.hidden_layer_size).to(device),
                    torch.zeros(model.nlayer, 1, model.hidden_layer_size).to(device))

            outVal, hidden_, outcov = model.forward(seq,hidden)
            outcov = outcov[:,-1,:].view(1,3,3)
            outcovT = torch.transpose(outcov,1,2)
            R = torch.matmul(outcovT,outcov)
            R = R.cpu().numpy()[0]
            xva = outVal.cpu().numpy()[0]
            # xva = scaler.inverse_transform(xva)[-1]
            xva = xva[-1] * dat_mm[:9] + dat_min[:9]

            # kf.predict()
            # kf.update(xva[0:9])
            # flight_pred[i,:9] = kf.x

            kf.predict()
            kf.update(xva[6:9],R=R)
            # kf.update(xva[6:9])
            flight_pred[i,:9] = kf.x

            # kf.predict()
            # kf.update(xva[6:9])
            # flight_pred[i,:9] = kf.x

            # flight_pred[i][0:3] = flight_pred[i - 1][0:3] + flight_pred[i - 1][3:6] * t_step# + 0.5 * flight_pred[i - 1][6:9] * t_step**2
            # flight_pred[i][3:6] = flight_pred[i - 1][3:6] + flight_pred[i - 1][6:9] * t_step
            # flight_pred[i][6:9] = xva[6:9]

            # flight_pred[i][0:3] = xva[0:3]
            # flight_pred[i][3:6] = xva[3:6]
            # flight_pred[i][6:9] = xva[6:9]

        if i ==len(flight_pred)-1:
            # print(flight_pred[i][0:3])
            # print(flight_origin[i][0:3])
            # dist = np.linalg.norm(flight_pred[i][0:3]-flight_origin[i][0:3])
            dist_x = np.linalg.norm(flight_pred[i][0:3] - flight_origin[i][0:3])
            dist_y = np.linalg.norm(flight_pred[i][3:6] - flight_origin[i][3:6])
            dist_z = np.linalg.norm(flight_pred[i][6:9] - flight_origin[i][6:9])
            # pred_error.append(np.array([dist_x, dist_y, dist_z]))
            if verbose:
                print('pred_error:{}'.format(dist_x))
    lstm_end = time.time()


    k_time = lstm_end-lstm_start
    # pred_time.append(k_time)
    if verbose:
        print('time:{}'.format(k_time))

    # if k == num:
    #     ax = plt.subplot(projection='3d')
    #     ax.set_title('Test Data '+str(k))
    #     ax.set_xlabel('X')  # 设置x坐标轴
    #     ax.set_ylabel('Y')  # 设置y坐标轴
    #     ax.set_zlabel('Z')  # 设置z坐标轴
    #     ax.grid(True)
    #     ax.autoscale(axis='x',tight=True)
    #     flight_cal = np.array(flight_cal)
    #
    #     for i in range(flight_cal.shape[0]):
    #         ax.scatter(flight_pred[i, 0], flight_pred[i, 2], flight_pred[i, 1],c='g')
    #         ax.scatter(flight_cal[i, 0], flight_cal[i, 2], flight_cal[i, 1],c='b')
    #         ax.scatter(flight_origin[i, 0], flight_origin[i, 2], flight_origin[i, 1],c='r')
    #         # plt.ion()
    #         # plt.pause(0.3)
    #     # ax.scatter(flight_origin[:,0], flight_origin[:,2], flight_origin[:,1],c='r')
    #     plt.ioff()
    #     # plt.xlim(-0.5, 0.5)
    #     plt.ylim(-2, 2)
    #     plt.show()
    #     break
    
    # if verbose:
    #     print('Average pred time:{}'.format(np.mean(pred_time)))
    #     print('Average pred error:{}'.format(np.mean(pred_error,axis=0)))
    #     print('Average cal error:{}'.format(np.mean(cal_error,axis=0)))

    return flight_pred

# def gen_data_from_model():
