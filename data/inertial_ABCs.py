import numpy as np
from abc import ABC, abstractmethod


class IMU(ABC):
    @abstractmethod
    def __init__(self):
        self.timestamp = 0.0
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

    @abstractmethod
    def read(self, data):
        ...

    def unroll(self):
        return self.gyro, self.acc, self.timestamp


class GT(ABC):
    @abstractmethod
    def __init__(self):
        self.timestamp = 0.0
        self.pos = np.array([0.0, 0.0, 0.0])
        self.att = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.ang_vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

    @abstractmethod
    def read(self, data):
        ...

    def read_from_tuple(self, data):
        self.pos = data[0]
        self.vel = data[1]
        self.att = data[2]
        self.ang_vel = data[3]
        self.acc = data[4]
        self.timestamp = data[5]
        return self

    def unroll(self):
        return self.pos, self.vel, self.att, self.ang_vel, self.acc, self.timestamp
