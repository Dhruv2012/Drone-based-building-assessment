import numpy as np
import pandas as pd
import geopy.distance
from scipy.linalg import inv
import matplotlib.pyplot as plt
from utils import Quaternion

import utils

class KalmanFilter():

    def GetScale(self, combined_logs):
        """ Input: prev_coords = (prev_lat, prev_lon) current_cords = (current_lat, current_lon)
            Output: scale = distance traveled in meters/ local distance"""

        for index, i in combined_logs.iterrows():
            if index!=0:    
                
                current_coords = (combined_logs['latitude'][index], combined_logs['longitude'][index])
                prev_coords = (combined_logs['latitude'][index-1], combined_logs['longitude'][index-1])
                global_distance = abs(geopy.distance.distance(prev_coords, current_coords).m)

                prev_translation = [combined_logs['x_orb'][index-1], combined_logs['y_orb'][index-1], combined_logs['z_orb'][index-1]]
                translation = [combined_logs['x_orb'][index], combined_logs['y_orb'][index], combined_logs['z_orb'][index]]
                local_distance = np.linalg.norm(np.array(translation)-np.array(prev_translation))

                scale = global_distance/local_distance
        return scale
    
    def get_drone_data(self, df):
        """ Input: combined dataframe of vo and flight log
            Output: numpy array of IMU readings"""
        imu_readings = []
        # imu = pd.DataFrame(columns=('xspeed', 'yspeed', 'zspeed', 'q_x', 'q_y', 'q_z', 'q_w'))
        for index, row in df.iterrows():
            rpy = Quaternion(euler=[row['roll'], row['pitch'], row['yaw']])
            # rpy_imu = rpy
            gimbal_ot = Quaternion(euler=[0, row['gimbal.yaw'], row['gimbal.pitch']])
            gimbal_otp = Quaternion(w = gimbal_ot.w, x=-gimbal_ot.x, y=-gimbal_ot.y, z=-gimbal_ot.z )
            rpy_imu = (rpy.quat_mult_left(gimbal_ot, out='Quaternion')).quat_mult_right(gimbal_otp, out='Quaternion')
            i = [row['xspeed'], row['yspeed'], row['zspeed'], rpy_imu.x, rpy_imu.y, rpy_imu.z, rpy_imu.w ]
            imu_readings.append(i)

        return np.array(imu_readings)

    def get_visual_odom(self, df):
        """ Input: combined dataframe of vo and flight log
            Output: numpy array of Visual Odom"""
        columns = ['time', 'x_orb', 'y_orb', 'z_orb', 'qx_orb', 'qy_orb', 'qz_orb', 'qw_orb']
        zs = []

        for i in range(len(df)):
            newrow = df.loc[i, columns].tolist()
            zs.append(newrow)

        zs = np.array(zs)
        zs = np.delete(zs, 0, 1)
        return zs

    def plot_points(self, x_filtered, y_filtered, z_filtered, x_noisy, y_noisy, z_noisy):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.plot3D(x_noisy, y_noisy, z_noisy, 'black', label="Actual")
        ax.plot3D(x_filtered, y_filtered, z_filtered, 'red', label="Filtered")
        plt.title("3D Data")
        plt.legend()
        plt.show()
    
    def compute_error(self, xs, zs):
        error = (zs - xs)/(xs+1e-5)
        return error.sum(axis = 0)*100/xs.shape[0]

    def convert_csv(self, xs, t, df):
        xs = np.c_[ t, xs ]
        # print(xs)
        df_kf = pd.DataFrame(xs, columns = ['time', 'x_orb','y_orb','z_orb', 'qx_orb', 'qy_orb', 'qz_orb', 'qw_orb'])
        df_kf['image_name'] = df.loc[:,"image_name"]
        df_kf['height'] = df.loc[:,"height"]
        df_kf['latitude'] = df.loc[:,"latitude"]
        df_kf['longitude'] = df.loc[:,"longitude"]
        df_kf.to_csv(r'/Users/ekanshgupta/Downloads/IIIT-H/dataset/kalman.csv', index=False)

    def kalman_filter(self, imus, zs, df):
        """ Multivariate kalman filter to estimate the drone position based on visual odometry and imu readings
        """
        t = np.array(df.iloc[: , :1])
        
        scale = self.GetScale(df)
        
        # zs[:3] = zs[:3]/scale
        imus[:3] = imus[:3]* scale
        mean = imus.sum(axis=0)/imus.shape[0]
        variance = ((imus - mean)**2/imus.shape[0]).sum(axis=0)
        print(variance)
        # variance = np.array([1.26207540e-01, 1.36853832e-01, 0.00000000e+00, 5.00581157e-07,
        #                      1.33706297e-04, 8.90977548e-05, 1.07200128e-06])   # with scale

        variance = np.array([3.05978499e-06, 5.96475416e-07, 0.0000000e+00, 2.18177561e-07,
                             5.82756928e-05, 3.88331254e-06, 4.67230180e-07]) 
        # variance = np.array([3.05978499e-05, 5.96475416e-03, 0.00000000e+00, 2.18177561e-07,
        #                      5.82756928e-05, 3.88331254e-07, 4.67230180e-07]) # without scale ogx#5.50073707e-02 ogy#5.96475416e-02 ogqz =3.88331254e-05
        # [7.87901638e-01 7.63359700e-01 0.00000000e+00 5.70169998e-01
        # 5.00602536e-04 4.11275110e+00 1.20627494e+00]

        F = np.eye(7)
        F[3:,:] = 0  
        H = np.eye(7)
        P = np.eye(7) # 50
        R = np.eye(7)
        np.fill_diagonal(R, variance)
        # Q = np.random.normal(0, 1, size=(7,7))
        # Q = Q - min(Q.reshape(-1,1))
        # Q = np.maximum(Q, Q.transpose())
        Q = np.array([[1.25929999, 3.00890151, 2.60473338, 2.80596225, 1.98680951, 1.39690084, 1.08464718],
                    [3.00890151, 1.78758193, 3.5811534,  4.02832337, 2.4681936,  2.82868457, 1.88122508], 
                    [2.60473338, 3.5811534,  0.72131517, 3.00484506, 1.44670514, 2.67314721, 3.30774253], 
                    [2.80596225, 4.02832337, 3.00484506, 3.27673267, 4.72531989, 2.17656588,  2.54925095],
                    [1.98680951, 2.4681936,  1.44670514, 4.72531989, 1.39381748, 2.54321307,  2.46377053],
                    [1.39690084, 2.82868457, 2.67314721, 2.17656588, 2.54321307, 2.70526814,  1.66446777],
                    [1.08464718, 1.88122508, 3.30774253, 2.54925095, 2.46377053, 1.66446777,  1.44993373]])
        
        x = imus[0,:]    # initial guess of state
        x = x.reshape(7,1)
        xs, cov = [], []
        xs.append(x)
        cov.append(P)
        for i, z in enumerate(zs):
            if i == 0 :
                continue
            v = np.eye(7)
            np.fill_diagonal(v, imus[i])
            z = z.reshape(7,1)
            # predict
            dt = t[i]-t[i-1]
            # t_sample = [d, d, d, 1., 1., 1., 1.]
            A = np.eye(7)
            np.fill_diagonal(A,[dt,dt,dt,1,1,1,1])
            B = v @ A
            B = np.diag(B).reshape(7,1)

            x = x
            x = F @ x + B # x = Fx + (v)Δt  v--> imu update
            P = F @ P @ F.T + Q     

            #update 
            S = H @ P @ H.T + R     # system uncertainity
            K = P @ H.T @ inv(S)    # Kalman gain
            y = z - H @ x           # residual
            x += K @ y              # state update
            P = P - K @ H @ P       # covariance update

            xs.append(x)
            cov.append(P)

        xs, cov = np.array(xs), np.array(cov)
        xs = xs.reshape(-1,7)
        self.convert_csv(xs, t, df)
        print("Error Values")
        print(self.compute_error(xs, zs))  ## x, y, z, q_x, q_y, q_z, q_w
        self.plot_points(xs[:,0], xs[:,1], xs[:,2], zs[:,0], zs[:,1], zs[:,2])

if __name__ == "__main__":
    # (utils.Combinelogs(pd.read_csv('/Users/ekanshgupta/Downloads/IIIT-H/data.csv'), pd.read_csv('/Users/ekanshgupta/Downloads/IIIT-H/odom.csv'))).to_csv(r'/Users/ekanshgupta/Downloads/IIIT-H/combined.csv', index=False)
    df = pd.read_csv('/Users/ekanshgupta/Downloads/IIIT-H/dataset/combined.csv')
    kf = KalmanFilter()
    imus = kf.get_drone_data(df)
    zs = kf.get_visual_odom(df)

    kf.kalman_filter(imus, zs, df)