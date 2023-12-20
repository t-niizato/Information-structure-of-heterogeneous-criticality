import numpy as np
import os
import pathlib
import sys

# 現在のパスの取得
cwd = os.getcwd()
p_file = pathlib.Path(cwd)
sys.path.append(str(p_file) + "/python_files")

import trajectory as tj
import Phi_data as phi
from drawer import Drawer, BaseModel

class ayus(BaseModel):

    def __init__(self, data_num, hist, type):
        # parameters
        self.N = 10
        self.hist = hist

        X, school = phi.data_pair_generate(data_num, hist)
        time = np.arange(hist-1, school.data.xy.shape[0])

        self.P = school.data.xy[time, :]
        self.V = school.data.direction()[time]
        self.S = school.data.speed()[time]
        self.positions = np.zeros((self.N, 2))
        self.velocities = np.zeros((self.N, 2))
        self.t = 0
        self.tmax = len(time)
        self.typename = type
        self.trajectory = school.data.xy
        self.direction_history = school.output_phi_all(hist, hist, "d_shita")


        # colors
        self.colors = np.zeros((self.N,3))

        # phi : direction
        self.phi_dir = X[0]
        self.max_phi_dir = np.max(self.phi_dir)
        self.min_phi_dir = np.min(self.phi_dir)
        self.mean_phi_dir = np.mean(self.phi_dir)

        # phi : speed
        self.phi_sp = X[2]

        # main complex : direction
        self.main_complex_dir = X[1]

        # main complex : speed
        self.main_complex_sp = X[3]

        # lifetime core
        self.core = np.zeros(self.N)

        for i in range(self.N):
            self.colors[i][2] = 1.0

        self.colors[0][0] = 1.0
        self.colors[0][1] = 0.0
        self.colors[0][2] = 0.0

    def data_at_t(self, t):
        self.center_x = (np.mean(self.P[t, 0:self.N], axis=0)-1250)/25
        self.center_y = (np.mean(self.P[t, self.N:self.N*2], axis=0)-1000)/20

        self.positions[:, 0] = (self.P[t, 0:self.N] - 1250)/25
        self.positions[:, 1] = (self.P[t, self.N:2*self.N]-1000)/20
        self.velocities[:, 0] = np.cos(self.V[t, :])
        self.velocities[:, 1] = np.sin(self.V[t, :])
        self.history = self.trajectory[t:self.hist+t]


        return tj.trajectory_difference(self.N, self.history)

    def dshita_hist(self, t):
        dire = self.direction_history[t]
        x = np.cos(dire)
        y = np.sin(dire)
        hist_x = np.zeros((self.N, self.hist))
        hist_y = np.zeros((self.N, self.hist))

        for i in range(self.hist):
            hist_x[:, i] = np.sum(x[:, 0:i], axis=1)
            hist_y[:, i] = np.sum(y[:, 0:i], axis=1)

        MC = self.main_complex_dir[t]

        return hist_x, hist_y, MC

    def pattern(self):
        MC1 = self.main_complex_dir[self.t//10 + self.hist//10]
        MC2 = self.main_complex_sp[self.t//10 + self.hist//10]
        if self.typename == "normal":
            for i in range(self.N):
                if MC1[i] == 1 and MC2[i] == 1:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.6
                    self.colors[i][2] = 0.0
                elif MC1[i] == 1 and MC2[i] == 0:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 1.0
                elif MC1[i] == 0 and MC2[i] == 1:
                    self.colors[i][0] = 1.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 0.0
                else:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 0.0

        if self.typename == "core_dir":
            # core_dulation
            for i in range(self.N):
                if MC1[i] == 1:
                    self.core[i] = self.core[i] + 1
                else:
                    self.core[i] = 0


            for i in range(self.N):
                if self.core[i] > 0:
                    self.colors[i][0] = 1.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 0.0
                else:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 0.0

        if self.typename == "core_sp":
            # core_dulation
            for i in range(self.N):
                if MC2[i] == 1:
                    self.core[i] = self.core[i] + 1
                else:
                    self.core[i] = 0

            for i in range(self.N):
                if self.core[i] > 0:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 1.0
                else:
                    self.colors[i][0] = 0.0
                    self.colors[i][1] = 0.0
                    self.colors[i][2] = 0.0




    def update(self):
        self.data_at_t(self.t)
        # self.draw_text("letter",[x, y], font size)
        self.draw_text("PHI_dir:"+str(self.phi_dir[int(self.t/10) + self.hist//10])[0:6], [400, 650], 12)
        self.draw_text("PHI_sp:"+str(self.phi_sp[int(self.t/10) + self.hist//10])[0:6], [400, 620], 12)

        self.t = self.t + 1
        self.pattern()

        if self.t == self.tmax:
            exit()

if __name__ == '__main__':
    draw_name = ["core_dir", "core_sp"]
    model = ayus(data_num=0, hist=600, type="core_dir")
    drawer = Drawer(model, coding_style='others', dimension=2, trail_length=200, trail_max_distance=300, color_enable=True)

    drawer.run()
