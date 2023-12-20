import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Data as Dt
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.spatial import Delaunay

#from joblib import Parallel, delayed


sns.set()

# arg(x1 - x2)の計算 （メソッドの中で使う）
def def_arg(x1, x2):
    return np.arctan2(np.sin(x1-x2), np.cos(x1-x2))

def rotate(x1, x2, arg):
    z1 = x1 * np.cos(-arg) + x2 * np.sin(-arg)
    z2 = - x1 * np.sin(-arg) + x2 * np.cos(-arg)
    return z1, z2


class Fish_Data():

    def __init__(self, position, fish_number, scale):
        self.xy = position
        self.N = fish_number
        self.scale = scale


    # 与えられたデータから解析に必要なデータに変形
    # 距離行列・角度行列はt=0を採用
    def create_data(self):
        self.xy = self.smoothing()
        self.dir = self.direction()
        self.turn = self.turn_rate()
        self.dis_matrix = self.distance_matrix(0)
        self.rel_matrix = self.relative_matrix(0)

    # 時刻tにおける状態
    def STEP(self, t):
        self.dis_matrix = self.distance_matrix(t)
        self.rel_matrix = self.relative_matrix(t)


    def smoothing(self):
        # データの抜き出し
        # ステップ幅は1-5引の場合, 統一のためstep=6とする.
        # 4-5匹の場合は, すでに調整しなくても6/120sec.
        delta_t = self.scale * 5
        print("delta_t = ", delta_t/120)
        if self.N == 1 or self.N == 2 or self.N == 3:
            self.xy = self.xy[::delta_t, :]
        elif self.N == 4 and self.xy.shape[0] > 50000:
            self.xy = self.xy[::delta_t, :]
        else:
            #pass
            self.xy = self.xy[::self.scale, :]
        # スムージング（前後１ステップ）
        return np.array([np.convolve(self.xy[:, i], np.ones(3) / float(3), 'valid')\
                         for i in range(0, 2 * self.N)]).T

    # スムージング済みの軌跡を代入して時刻tにおける角度を求める：smoothの後に行うこと
    def direction(self):
        bft = np.roll(self.xy, 1, axis=0)  # 一つ前の位置
        df = self.xy - bft
        return np.array([np.arctan2(df[:, self.N + i], df[:, 0 + i])\
                         for i in range(0, self.N)]).T
    # 速度の計算
    def speed(self):
        bft = np.roll(self.xy, 1, axis=0)  # 一つ前の位置
        df = self.xy - bft
        K = df[:, 0:self.N] * df[:, 0:self.N] + df[:, self.N:2*self.N] * df[:, self.N:2*self.N]
        return np.sqrt(K)

    # 重心の計算
    def center_of_mass(self):
        self.cx = np.mean(self.xy[:, 0:self.N], axis=1)
        self.cy = np.mean(self.xy[:, self.N:2*self.N], axis=1)

    # 重心からの距離
    def center_distance(self):
        self.center_of_mass()
        time = self.xy.shape[0]
        X = self.xy[:, 0:self.N] - self.cx.reshape((time, 1))
        Y = self.xy[:, self.N:2*self.N] - self.cy.reshape((time, 1))
        return np.sqrt(np.power(X, 2) + np.power(Y, 2))

    # 重心からの距離Rによる発火
    def distance_fire(self, R):
        return self.center_distance() < R

   # 平均角度の計算
    def average_direction(self):
        dx = np.mean(np.cos(self.dir), axis=1)
        dy = np.mean(np.sin(self.dir), axis=1)
        self.mean_direction = np.arctan2(dy, dx)

    # 平均角度からの偏差
    def dif_average(self):
        self.average_direction()
        mean = self.mean_direction
        time = mean.shape[0]
        mean = mean.reshape((time, 1))

        dx = np.mean(np.cos(self.dir), axis=1)
        dy = np.mean(np.sin(self.dir), axis=1)

        Results = np.abs(def_arg(self.dir, mean))

        for i in range(len(dx)):
            if (dx[i]*dx[i] + dy[i]*dy[i]) < 1e-15:
                if self.N == 2:
                    Results[i][0] = np.pi/2
                    Results[i][1] = np.pi/2
                else:
                    pass

        return Results

    # 平均向きからの距離Rによる発火
    def direction_fire(self, deg):
        return self.dif_average() <= (deg / 360 * np.pi)

    # トポロジカル近傍の計算：ソーティング
    def topological_sort(self, neighbor, t):
        M = np.argsort(self.distance_matrix(t), axis=1)[:, 1:1+neighbor]
        N = np.zeros((self.N, self.N))
        for i in range(self.N):
            N[i, M[i, :]] = 1
        return N

    def find_neighbors(self, pindex, triang):
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]

    # Delaunay近傍の計算
    def Delaunay_Matrix(self, t):
        N = np.zeros((self.N, self.N))
        Tri = Delaunay(self.xy[t].reshape(2, self.N).T)
        for i in range(self.N):
            N[i, self.find_neighbors(i, Tri)] = 1
        return N



    # 角度変化率
    def turn_rate(self):
        past_dishita = np.roll(self.dir, 1, axis=0)
        return def_arg(self.dir, past_dishita)

    # 距離行列（時刻t）
    def distance_matrix(self, t):
        xy_now = self.xy[t, :].reshape(2, self.N).T
        return squareform(pdist(xy_now))

    # 相対的位置関係を角度で表した行列
    def relative_matrix(self, t):
        xy_now = self.xy[t, :].reshape(2, self.N).T
        return np.array([np.arctan2(xy_now[:, 1] - xy_now[i, 1],\
                                    xy_now[:, 0] - xy_now[i, 0])\
                         for i in range(0, self.N)])

    # 視野による隣接角度行列の生成
    def V_field(self, deg, t):
        M = self.relative_matrix(t)
        return np.array([np.absolute(def_arg(M[i, :],\
                self.dir[t, i])) for i in range(0, self.N)]) <= (deg / 360 * np.pi)

    # 半径R以内を接触領域とみなしたネットワーク：自らを含めない
    def Contact_Network(self, R, t):
        return (self.distance_matrix(t) < R) & (self.distance_matrix(t) != 0)

    # 時刻tにおける同条件のネットワーク(結合)状態
    def Network(self, deg, R, t):
        return self.V_field(deg, t) & self.Contact_Network(R, t)

    # 状態遷移
    def State(self, deg, R):
        time = self.direction().shape[0]
        state = np.array([np.where(np.sum(self.Network(deg, R, t), axis=1) > 0, 1, 0)\
                      for t in range(time)])
        next = np.roll(state, -1, axis=0)
        M = np.dot(np.dot(np.linalg.inv(np.dot(state.T, state)), state.T), next)

        return M

    # Φ計算のための行列の生成
    def State_by_Node(self, deg, R):
        total_state = np.power(2, self.N)
        bits = np.power(2, range(0, self.N))
        time = self.direction().shape[0]

        SbN = np.zeros((total_state, self.N + 1))
        state = np.array([np.where(np.sum(self.Network(deg, R, t), axis=1) > 0, 1, 0)\
                      for t in range(time)])
        # 計算のための調整
        next = np.roll(state, -1, axis=0)

        for i in range(0, time):
            num = np.dot(state[i, :], bits.T)
            SbN[num, 0:self.N] = SbN[num, 0:self.N] + next[i, :]
            SbN[num, self.N] = SbN[num, self.N]+1

        Total = SbN[:, self.N]
        SbN = np.delete(SbN, self.N, 1) / Total.T[:, None]
        SbN[np.isnan(SbN)] = 0

        return SbN

    # Φ計算のための行列の生成
    def State_Series(self, deg, R):

        time = self.direction().shape[0]
        state = np.array([np.where(np.sum(self.Network(deg, R, t), axis=1) > 0, 1, 0)\
                      for t in range(time)])

        return state
    # Φ計算のための行列の生成
    def State_Series_2(self, deg, R):
        time = self.direction().shape[0]
        center_dis = self.distance_fire(R)
        average_dir = self.direction_fire(deg)

        state = np.array([center_dis[t, :] & average_dir[t, :] for t in range(time)])

        return state


    # Φ計算のための行列の生成：３パラメータに一般化＋角度変化の度合いを追加 
    #  th =0 で State_by_Nodeと同じ（確認済）

    def Generalized_State_by_Node(self, deg, R, th):
        total_state = np.power(2, self.N)
        bits = np.power(2, range(0, self.N))
        time = self.direction().shape[0]

        SbN = np.zeros((total_state, self.N + 1))
        state = np.array([(np.where(np.sum(self.Network(deg, R, t), axis=1) > 0, 1, 0)\
                           & (np.absolute(self.turn[t][:]) >= th)) for t in range(3, time)])

        # 計算のための調整 
        next = np.roll(state, -1, axis=0)

        for i in range(0, time-3):
            num = np.dot(state[i, :], bits.T)
            SbN[num, 0:self.N] = SbN[num, 0:self.N] + next[i, :]
            SbN[num, self.N] = SbN[num, self.N]+1

        Total = SbN[:, self.N]
        SbN = np.delete(SbN, self.N, 1) / Total.T[:, None]
        SbN[np.isnan(SbN)] = 0

        return SbN

    # 全体情報と個体情報の関係ネットワーク

    def Whole_State_by_Node(self, R, th):
        total_state = np.power(2, self.N)
        bits = np.power(2, range(0, self.N))
        time = self.direction().shape[0]

        center_dis = self.distance_fire(R)
        average_dir = self.direction_fire(th)

        SbN = np.zeros((total_state, self.N + 1))
        state = np.array([center_dis[t, :] & average_dir[t, :] for t in range(3, time)])
        X = self.dif_average()

        # 計算のための調整 
        next = np.roll(state, -1, axis=0)

        for i in range(0, time - 3):
            num = np.dot(state[i, :], bits.T)

            SbN[num, 0:self.N] = SbN[num, 0:self.N] + next[i, :]
            SbN[num, self.N] = SbN[num, self.N] + 1

        Total = SbN[:, self.N]
        SbN = np.delete(SbN, self.N, 1) / Total.T[:, None]
        SbN[np.isnan(SbN)] = 0

        return SbN


    # ALL Connected Including Self-Loop
    def Material(self):
        return np.ones((self.N, self.N))


    def show_data(self, deg, R, t):
        print("Direction : ", fish_data.dir[t, :] / np.pi * 180)
        print("Relative_Matrix :", fish_data.relative_matrix(t) / np.pi * 180)
        print("EyeSight:", fish_data.V_field(deg, t))
        print("Result:", fish_data.Network(deg, R, t))

    # Poralityを計算
    def Polarity(self):
        dx = np.sum(np.cos(self.dir), axis=1)
        dy = np.sum(np.sin(self.dir), axis=1)
        return np.sqrt(dx*dx+dy*dy)/self.N

    # Torusを計算
    def Torus(self):

        center_x = np.mean(self.xy[:, 0:self.N], axis=1)
        center_y = np.mean(self.xy[:, self.N:2*self.N], axis=1)
        time = len(center_x)

        dr_x = [self.xy[t, 0:self.N] - center_x[t] for t in range(time)]
        dr_y = [self.xy[t, self.N:2*self.N] - center_y[t] for t in range(time)]
        norm = np.sqrt(np.power(dr_x, 2) + np.power(dr_y, 2))

        print(dr_x[0:5])
        print(dr_y[0:5])
        print(norm)


        return 0

    # 時刻t付近の位置関係と向きを描写
    def draw_relation(self, R, deg, t):
        list = np.arange(self.N)
        pos_current = self.xy[t, :].reshape(2, self.N).T
        self.center_of_mass()
        self.average_direction()
        x = self.relative_matrix(t)
        center = pos_current.mean(axis=0)

        # 相対位置ベクトル
        plt.figure(figsize=(12, 12))
        Rg = 200.0
        plt.xlim(center[0]-Rg, center[0]+Rg)
        plt.ylim(center[1]-Rg, center[1]+Rg)

        for i in range(0, self.N):
            list2 = np.arange(self.N)
            list2 = np.delete(list2, i)
            plt.quiver(pos_current[i, 0], pos_current[i, 1], \
                       20 * np.cos(x[i, list2]), 20 * np.sin(x[i, list2]), angles='xy')

        # 時刻tの個体の向き
        plt.quiver(pos_current[list, 0], pos_current[list, 1], \
                   np.cos(self.dir[t, list]), np.sin(self.dir[t, list]), color='red', angles='xy')
        # 死角でない部分の色を変える
        M = self.Network(deg, R, t)
        for i in range(0, self.N):
            list3 = np.where(M[i ,:] == True)
            if list3[0].size > 0:
                plt.quiver(pos_current[i, 0], pos_current[i, 1],\
                  30 * np.cos(x[i, list3]), 30 * np.sin(x[i, list3]), color='green', angles='xy')

        plt.quiver(self.cx[t], self.cy[t], \
                       30 * np.cos(self.mean_direction[t]), 30 * np.sin(self.mean_direction[t]), color='blue', angles='xy')

        # 時刻tの軌跡
        plt.plot(self.xy[t - 1 : t + 15, list], self.xy[t - 1:t + 15, list + self.N])
        plt.legend(['1', '2', '3', '4'])
        plt.title("time : " + str(t))
        plt.show()
        #plt.pause(2)
        plt.close()

        # 時刻t付近の位置関係と向きを描写


    def pick_leader(self):
        self.average_direction()
        time = self.direction().shape[0]

        leader = np.zeros(time)
        for t in range(0, time):
            xx, yy = rotate(self.xy[t, 0:self.N], self.xy[t, self.N:2 * self.N], -self.mean_direction[t])
            leader[t] = np.argmax(xx)
        return leader

    def pick_phi_leader(self, deg, R):
        time = self.direction().shape[0]
        state = np.array([np.where(np.sum(self.Network(deg, R, t), axis=1) > 0, 1, 0)\
                      for t in range(time)])
        sum_state = np.sum(state, axis=1)
        leader = np.zeros(time)

        for t in range(0, time):
            if sum_state[t] == self.N-1:
                leader[t] = np.argmin(state[t, :])
            else:
                leader[t] = -1

        return leader





if __name__ == '__main__':

    n_fish = 3
    data = Dt.fish_data(n_fish)
    pos = Dt.trajectory(data, 1)
    fish_data = Fish_Data(pos, n_fish)
    fish_data.create_data()

    time = 2287

    #fish_data.draw_relation(800, 270, time)
    L = fish_data.pick_leader()
    L2 = fish_data.pick_phi_leader(360, 1000)
    print(L)
    print(L2)

    index = np.where(L2>=0)[0]
    print(index.shape)
    #L = L[index]
    #L2 = L2[index]
    print(L.shape[0], L2.shape[0])
    print(np.sum(L == L2)/index.shape[0])



    #print(fish_data.xy[time, 0:fish_data.N])
    #print(fish_data.xy[time, fish_data.N:2 * fish_data.N])
    #print(fish_data.average_direction())
    #print(fish_data.mean_direction[time]/(2*np.pi)*360)
    #print(L[time])

    #fish_data.draw_relation(800, 270, time)
    #xx, yy = rotate(fish_data.xy[time, 0:fish_data.N], fish_data.xy[time, fish_data.N:2 * fish_data.N], -fish_data.mean_direction[time])
    #print(np.argmax(xx))
    #for i in range(0, fish_data.N):
    #    plt.plot(fish_data.xy[time, i],  fish_data.xy[time, fish_data.N + i], "o")
    #plt.show()

    #for i in range(0, fish_data.N):
    #    plt.plot(xx[i], yy[i], "o")
    #plt.show()













