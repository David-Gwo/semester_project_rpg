import gflags
import yaml
import sys

import numpy as np
import pandas as pd

from utils.directories import safe_mkdir_recursive
from utils.algebra import correct_quaternion_flip
from data.config.simulated_ds_flags import FLAGS
from data.inertial_ABCs import IMU, GT, InertialDataset


class GenIMU(IMU):
    def __init__(self):
        super(GenIMU, self).__init__()
        self.ts_indx = 0
        self.gyro_indx = [4, 5, 6]
        self.acc_indx = [1, 2, 3]

    def read(self, data):
        self.timestamp = data[self.ts_indx].astype(np.float) * 10e6
        self.gyro = data[self.gyro_indx].astype(np.float)
        self.acc = data[self.acc_indx].astype(np.float)


class GenGT(GT):
    def __init__(self):
        super(GenGT, self).__init__()

    def read(self, data):
        ts_indx = 0
        pos_indx = [1, 2, 3]
        att_indx = [7, 4, 5, 6]  # w,x,y,z
        vel_indx = [8, 9, 10]

        data = data.astype(np.float)
        self.timestamp = data[ts_indx] * 10e6
        self.pos = data[pos_indx]
        self.att = data[att_indx]
        self.vel = data[vel_indx]


class GenDSManager(InertialDataset):
    def __init__(self, *args):
        super(GenDSManager, self).__init__()
        self.ds_flags = FLAGS
        self.simulation_dir = "./catkin_ws/src/rpg_vi_simulation/"

        self.ds_gen_config_dict = None
        self.sampling_freq = None
        self.g_value = None
        self.dataset_name = None
        self.get_ds_params()

        try:
            _ = FLAGS(args)  # parse flags
        except gflags.FlagsError:
            print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
            sys.exit(1)

        # The dataset data is found at the simulator directory
        self.ds_dir = "{0}vi_sim_interface/sim_datasets/{1}".format(self.simulation_dir, self.dataset_name)
        self.gt_file = "{0}interpolated_gt.txt".format(self.ds_dir)
        self.imu_file = "{0}imu_meas.txt".format(self.ds_dir)

        # But all the necessary files for training will be stored in the local datasets path
        self.ds_local_dir = './data/dataset/simulated_datasets/{0}'.format(self.dataset_name)
        safe_mkdir_recursive(self.ds_local_dir, overwrite=False)

    def get_ds_params(self):
        """
        Reads the needed parameters from the synthetic dataset generation configuration files
        """

        config = "{0}vi_sim_interface/exp_configs/{1}.yaml".format(self.simulation_dir, self.ds_flags.dataset_version)

        with open(config, 'r') as stream:
            try:
                self.ds_gen_config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        config_type = self.ds_gen_config_dict['vi_param']
        self.dataset_name = "{0}/{1}/{2}/".format(
            self.ds_gen_config_dict['traj_dir'], self.ds_flags.flight, self.ds_flags.number)

        imu_params = "{0}ze_vi_simulation/vi_params/{1}/imu_params.txt".format(self.simulation_dir, config_type)

        f = open(imu_params, "r")
        for i, x in enumerate(f):
            if i == 0:
                self.sampling_freq = float(x)
            elif i == 5:
                self.g_value = x

    def read_synthetic_data(self):

        raw_imu_data = []
        ground_truth_data = []

        for row in pd.read_csv(self.imu_file, header=None, delimiter=r"\s+").values:
            imu = GenIMU()
            imu.read(row)
            raw_imu_data.append(imu)

        gt_old = GenGT()
        for row in pd.read_csv(self.gt_file, header=None, delimiter=r"\s+").values:
            gt = GenGT()
            gt.read(row)
            ground_truth_data.append(gt)

            gt.integrate(gt_old, int_pos=False)
            gt_old = gt

        self.imu_data = raw_imu_data
        self.gt_data = ground_truth_data

    def get_raw_ds(self):

        self.read_synthetic_data()
        self.interpolate_ground_truth()

        return self.imu_data, self.gt_data

    def pre_process_data(self, gyro_scale_file, acc_scale_file):
        self.basic_preprocessing(gyro_scale_file, acc_scale_file, 10)

        corrected_quaternion = correct_quaternion_flip(np.stack(self.gt_data[:, 2]))
        for i in range(len(self.gt_data)):
            self.gt_data[i, 2] = corrected_quaternion[i, :]

        return self.imu_data, self.gt_data
