# coding: UTF-8
import pandas as pd
import numpy as np
import Data as Dt
import Class_Fish as CF
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

class Schools():

    def __init__(self, fish_number, data_number, scale):
        self.N = fish_number
        data = Dt.fish_data(fish_number)
        pos = Dt.trajectory(data, data_number)
        self.data = CF.Fish_Data(pos, fish_number, scale)
        self.CM = self.data.Material()
        self.data.create_data()
        self.scale = scale

        # ラベルづけ
        self.labels = tuple([chr(i) for i in range(97, 97+fish_number)])
        self.nodes = tuple([i for i in range(0, fish_number)])

    def output_for_phi(self, t=120, past=120, distance=100, deg= 360, name1="Full", name2="d_shita"):
        return self.network(name1, distance, deg, t, past), self.series(name2, t, past)


    def output_phi_all(self, t=120, past=120, name2="d_shita"):
        ranges = past + np.arange(self.data.turn_rate().shape[0] - t)
        slices = [1, 5, 3, 3, 2, 2, 1, 1]
        index = slices[self.scale - 1]
        ranges = ranges[::index]
        return np.array([self.series(name2, i, past).T for i in ranges])

    def phi_time(self, t=120, past=120, name2="d_shita"):
        ranges = past + np.arange(self.data.turn_rate().shape[0] - t)
        slices = [1, 5, 3, 3, 2, 2, 1, 1]
        index = slices[self.scale - 1]
        ranges = ranges[::index]
        return ranges

    def network(self, name, distance=0, deg=0, t=0, past=120):
        #　全結合ネットワーク
        if name == "Full":
            return np.ones((self.N, self.N), dtype=np.int16)
        # 　全結合ネットワーク without self-loop
        elif name == "Without_Loop":
            return np.ones((self.N, self.N), dtype=np.int16) - np.eye(self.N, dtype=np.int16)
        #   接触ネットワーク
        elif name == "Contact":
            C_matrix = np.zeros((self.N, self.N))
            for k in range(past):
                C_matrix = C_matrix + (self.data.Contact_Network(distance, t-k) & self.data.V_field(deg, t-k))
            C_matrix[C_matrix>=1] = 1
            return C_matrix
        #   トポロジカル・ネットワーク
        elif name == "Topological":
            T_matrix = np.zeros((self.N, self.N))
            neigbor = 3
            for k in range(past):
                T_matrix = T_matrix + (self.data.topological_sort(neigbor, t-k) * self.data.V_field(deg, t-k))
            T_matrix[T_matrix>=1] = 1
            return T_matrix
        #   ドロネー・ネットワーク
        elif name == "Delaunay":
            D_matrix = np.zeros((self.N, self.N))
            for k in range(past):
                D_matrix = D_matrix + (self.data.Delaunay_Matrix(t-k) * self.data.V_field(deg, t-k))
            D_matrix[D_matrix>=1] = 1
            return D_matrix

    def series(self, name="d_shita", t=120, past=120):
        #エラーの処理
        if t < past:
            #print("エラー：t<pastとなっていて計算不能!")
            return np.nan
        else:
            pass

        # 角度変化の時系列
        if name == "d_shita":
            return self.data.turn[t-past:t, :]
        # 速度の時系列
        elif name == "speed":
            return self.data.speed()[t-past:t, :]
        # internal viewも同じように使えるかも
        elif name == "entangle":
            pass

    def draw_netwrok(self, R, deg, t, name="Contact"):
        list = np.arange(self.N)
        pos_current = self.data.xy[t, :].reshape(2, self.N).T
        center = pos_current.mean(axis=0)
        connect = self.network(name, R, deg, t, 120)
        print(connect)

        # 相対位置ベクトル
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        Rg = 500.0
        plt.xlim(center[0]-Rg, center[0]+Rg)
        plt.ylim(center[1]-Rg, center[1]+Rg)

        # 近傍で接触したのみをリストアップ
        for i in range(self.N):
            for j in range(self.N):
                if connect[i, j] > 0:
                    plt.plot(pos_current[[i, j], 0], pos_current[[i, j], 1], color='red')

        # 過去120stepの軌跡
        plt.plot(self.data.xy[t - 120 : t , list], self.data.xy[t - 120:t, list + self.N])

        # 時刻tの個体の向き
        plt.quiver(pos_current[list, 0], pos_current[list, 1], np.cos(self.data.dir[t, list]), np.sin(self.data.dir[t, list]), color='black', angles='xy')
        for i in range(10):
            ax.text(pos_current[i, 0], pos_current[i, 1], i)
        #plt.text(pos_current[list, 0], pos_current[list, 1], list)


        plt.title("time : " + str(t))
        plt.show()

school = Schools(10, 0, 1)
times = school.phi_time()
#print(times)
#print(len(school.data.Polarity()))
#print(school.data.Torus())

#print(school.output_phi_all().shape)

#print(np.where(school.data.Torus() > 0.7)[0])
#plt.plot(school.data.Polarity())
#plt.plot(school.data.Torus())
#plt.show()

# draw torus --------------
#t = 9299
#school.data.draw_relation(1000, 360, t)
# t =[1988 1989 1990 1991 1992 1993 1994 1995 1996 2848 2849 2850 2851 2852
#  2853 2854 2855 2856 2857 2858 2859 2860 2861 2862 2863 2864 2872 2873
#  2874 4505 4506 4507 4508 6522 7948 7949 7950 7951 7952 7953 7954 9292
#  9293 9294 9295 9296 9299 9300 9301 9302 9303]
#---------------------------
#print(school.output_phi_all().shape)
#school.draw_netwrok(100, 360, 2260, "Delaunay")
#N, M = school.output_for_phi(300, 120, 100, 360,  "Topological", "d_shita")
#print(M.shape)
#print(N)
#plt.plot(school.series("speed",1300, 1300))
#plt.show()
#school.network("Delaunay", t=20)
