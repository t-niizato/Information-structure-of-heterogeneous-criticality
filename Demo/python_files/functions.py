import numpy as np
from numba import jit

""""-----------------------------------------
#
#               群れパラメータ
#
--------------------------------------------"""

def polarity(vx, vy, vz):
    cx = np.mean(vx)
    cy = np.mean(vy)
    cz = np.mean(vz)

    return np.sqrt(cx * cx + cy * cy + cz * cz)


def torus(px, py, pz, vx, vy, vz):
    center_x = np.mean(px)
    center_y = np.mean(py)
    center_z = np.mean(pz)

    cx = px - center_x
    cy = py - center_y
    cz = pz - center_z

    c_norm = np.sqrt(np.sum(cx * cx + cy * cy + cz * cz))

    cx = cx / c_norm
    cy = cy / c_norm
    cz = cz / c_norm

    rx = cy * vz - cz * vy
    ry = cz * vx - cx * vz
    rz = cx * vy - cy * vx

    number = px.shape[0]

    T = np.sum(np.sqrt(rx * rx + ry * ry + rz * rz)) / number

    return T

""""-----------------------------------------
#
#               境界条件
#
--------------------------------------------"""

@jit('Tuple((f8[:, :], f8[:, :]))(f8, f8, f8, f8[:, :], f8[:, :], i8)', nopython=True)
def boundary(bx, by, bz, P, V, n):
    for i in range(n):
        if P[0, i] > bx:
            P[0, i] = 2 * bx - P[0, i]
            V[0, i] = -V[0, i]
        elif P[0, i] < 0:
            P[0, i] = -P[0, i]
            V[0, i] = -V[0, i]
        if P[1, i] > by:
            P[1, i] = 2 * by - P[1, i]
            V[1, i] = -V[1, i]
        elif P[1, i] < 0:
            P[1, i] = -P[1, i]
            V[1, i] = -V[1, i]
        if P[2, i] > bz:
            P[2, i] = 2 * bz - P[2, i]
            V[2, i] = -V[2, i]
        elif P[2, i] < 0:
            P[2, i] = -P[2, i]
            V[2, i] = -V[2, i]
    return P ,V

""""-----------------------------------------
#
#           Alignment Rule の適用
#
--------------------------------------------"""

@jit('i8[:](i8, i4[:], i8[:])', nopython=True)
def add_index(pindex, neighbors, add_neighbors):
    for i in range(len(neighbors)):
        add_neighbors[i] = neighbors[i]
    add_neighbors[len(neighbors)] = pindex
    return add_neighbors


@jit('f8[:, :](i8, f8, i8[:], f8[:, :], f8[:, :, :], f8[:, :])', nopython=True)
def alignment_vectors(pindex, radius, neighbors, X, Y, R):
    for i in range(6):
        count = 0
        for j in range(len(neighbors)):
            R[0, j] = X[0,  neighbors[j]] + radius/3 * (i+1) * Y[i, 0, neighbors[j]] - X[0, pindex]
            R[1, j] = X[1,  neighbors[j]] + radius/3 * (i+1) * Y[i, 1, neighbors[j]] - X[1, pindex]
            R[2, j] = X[2,  neighbors[j]] + radius/3 * (i+1) * Y[i, 2, neighbors[j]] - X[2, pindex]
            distance = np.sqrt(R[0, j]*R[0, j] + R[1, j]*R[1, j] + R[2, j]*R[2, j])
            if distance < radius:
                break
            else:
                count = count + 1
        if count == len(neighbors):
            break
    return R


@jit('Tuple((f8[:, :], i8))(i8, f8, f8, i8[:], f8[:, :], f8[:, :, :], f8[:, :])', nopython=True)
def alignment_vectors_ver_2(pindex, radius, V,  neighbors, X, Y, R):
    for i in range(20):
        count = 0
        for j in range(len(neighbors)):
            R[0, j] = X[0,  neighbors[j]] + V/3 * (i + 1) * Y[i, 0, neighbors[j]] - X[0, pindex]
            R[1, j] = X[1,  neighbors[j]] + V/3 * (i + 1) * Y[i, 1, neighbors[j]] - X[1, pindex]
            R[2, j] = X[2,  neighbors[j]] + V/3 * (i + 1) * Y[i, 2, neighbors[j]] - X[2, pindex]
            distance = np.sqrt(R[0, j]*R[0, j] + R[1, j]*R[1, j] + R[2, j]*R[2, j])

            if distance < radius:
                break
            else:
                count = count + 1
        if count == len(neighbors):
            break

    return R, i

@jit('Tuple((f8[:], f8[:, :]))(i8, i8[:], f8[:,:], f8[:,:], f8[:], f8[:, :])',nopython=True)
def vector_distribution(pindex, neighbors, P, Out, C, V):

    n = len(neighbors)

    for i in range(n):
        V[0, i] = Out[0, i] + P[0, neighbors[i]] - P[0, pindex]
        V[1, i] = Out[1, i] + P[1, neighbors[i]] - P[1, pindex]
        V[2, i] = Out[2, i] + P[2, neighbors[i]] - P[2, pindex]
        d = np.sqrt(V[0, i]*V[0, i] + V[1, i]*V[1, i] + V[2, i]*V[2, i])
        V[0, i] = V[0, i]/d
        V[1, i] = V[1, i]/d
        V[2, i] = V[2, i]/d

        #平均の計算
        C[0] = C[0] + V[0, i]
        C[1] = C[1] + V[1, i]
        C[2] = C[2] + V[2, i]

    d = np.sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2])
    C[0] = C[0]/d
    C[1] = C[1]/d
    C[2] = C[2]/d

    return C, V



""""-----------------------------------------
#
#           Rotation Matrixの適用
#
--------------------------------------------"""

#@jit('f8[:](f8[:], i8)')
def nb_roll(X, N):
    return np.roll(X, N)


@jit('f8[:, :](f8[:, :])', nopython=True)
def matrix_unite_vector(X):
    for i in range(X.shape[1]):
        l = np.sqrt(X[0, i] * X[0, i] + X[1, i] * X[1, i] + X[2, i] * X[2, i])
        X[0, i] = X[0, i]/l
        X[1, i] = X[1, i]/l
        X[2, i] = X[2, i]/l
    return X


@jit('f8[:, :](f8[:, :],f8[:, :], f8[:, :], f8[:], f8[:, :], f8[:, :])', nopython=True)
def rotation_matrix(X, Y, Z, c, R, T):
    # 単位ベクトル化
    X = matrix_unite_vector(X)
    Y = matrix_unite_vector(Y)

    #Cross Product
    for i in range(X.shape[1]):
        Z[0, i] = X[1, i] * Y[2, i] - X[2, i] * Y[1, i]
        Z[1, i] = X[2, i] * Y[0, i] - X[0, i] * Y[2, i]
        Z[2, i] = X[0, i] * Y[1, i] - X[1, i] * Y[0, i]

    # Dot product
    for i in range(X.shape[1]):
        c[i] = X[0, i] * Y[0, i] + X[1, i] * Y[1, i] + X[2, i] * Y[2, i]

    # parameter
    p = 1/(1+c)

    for i in range(X.shape[1]):
        if c[i] != -1:
            R[0, 0] = p[i] * (- Z[1, i]*Z[1, i] - Z[2, i]*Z[2, i]) + 1
            R[0, 1] = p[i] * Z[0, i] * Z[1, i] - Z[2, i]
            R[0, 2] = p[i] * Z[0, i] * Z[2, i] + Z[1, i]
            R[1, 0] = p[i] * Z[0, i] * Z[1, i] + Z[2, i]
            R[1, 1] = p[i] * (- Z[0, i]*Z[0, i] - Z[2, i]*Z[2, i]) + 1
            R[1, 2] = p[i] * Z[1, i] * Z[2, i] - Z[0, i]
            R[2, 0] = p[i] * Z[0, i] * Z[2, i] - Z[1, i]
            R[2, 1] = p[i] * Z[1, i] * Z[2, i] + Z[0, i]
            R[2, 2] = p[i] * (- Z[0, i] * Z[0, i] - Z[1, i] * Z[1, i]) + 1

            T[0, i] = R[0, 0] * Y[0, i] + R[0, 1] * Y[1, i] + R[0, 2] * Y[2, i]
            T[1, i] = R[1, 0] * Y[0, i] + R[1, 1] * Y[1, i] + R[1, 2] * Y[2, i]
            T[2, i] = R[2, 0] * Y[0, i] + R[2, 1] * Y[1, i] + R[2, 2] * Y[2, i]
        else:
            T[0, i] = -Y[0, i]
            T[1, i] = -Y[1, i]
            T[2, i] = -Y[2, i]

    return T


""""-----------------------------------------
#
#           Attraction Ruleの関数
#
--------------------------------------------"""


@jit('UniTuple(f8, 2)(f8, f8, f8)',nopython=True)
def compute_shita_phi_single(x, y, z):
    return np.arccos(z / np.sqrt(x * x + y * y + z * z)), np.arctan2(y, x)

@jit('UniTuple(f8[:], 2)(f8[:], f8[:], f8[:])',nopython=True)
def compute_shita_phi_vector(x, y, z):
    return np.arccos(z / np.sqrt(x * x + y * y + z * z)), np.arctan2(y, x)

@jit('UniTuple(f8, 3)(f8, f8)',nopython=True)
def compute_xyz_single(shita, phi):
    return np.cos(shita) * np.sin(phi), np.sin(shita) * np.sin(phi), np.cos(phi)

@jit('UniTuple(f8[:], 3)(f8[:], f8[:])',nopython=True)
def compute_xyz_vector(shita, phi):
    return np.cos(shita) * np.sin(phi), np.sin(shita) * np.sin(phi), np.cos(phi)

@jit('UniTuple(f8[:], 3)(f8[:], f8[:], f8[:], f8, f8, f8, f8)',nopython=True)
def relative_position(X, Y, Z, x, y, z, m_dis):
    return (X-x)/m_dis, (Y-y)/m_dis, (Z-z)/m_dis

@jit('UniTuple(f8[:], 3)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])',nopython=True)
def intersections(sx, sy, sz, dx, dy, dz):
    # 交点のスケールdを計算
    b = sx * dx + sy * dy + sz * dz
    c = sx * sx + sy * sy + sz * sz - 1
    d_p = (- b + np.sqrt(b * b - c))
    return sx + d_p * dx, sy + d_p * dy, sz + d_p * dz

#@jit('f8(f8)', nopython=True)
def discrete_noise(arg, noise):
    return int(arg/noise)*noise + noise

@jit('f8(f8[:])', nopython=True)
def namba_max(x):
    return np.max(x)

@jit('i8[:](i8[:],i8[:])', nopython=True)
def intersect(x, y):
    # xを現在ピックアップしてきたlist
    # yをこれまでの履歴リスト（こっちの方が長い）
    for i in range(x.shape[0]):
        if np.where(y == x[i])[0].size == 0:
            x[i] = -1
    return x[x>=0]


@jit('i8(f8[:, :], f8, i8, i8[:])', nopython=True)
def avoidance_index(matrix, threshold, n, update_list):
    list = np.sum((matrix < threshold) - np.eye(n), axis=0)
    index = np.where(list > 0)[0]
    index = intersect(index, update_list)
    if len(index) > 0:
        return np.random.choice(index)
    else:
        return -1

@jit('i8[:](f8[:, :], f8, i8)', nopython=True)
def avoidance_neighbor(matrix, threshold, index):
    list = np.where((matrix[:, index] < threshold) == 1)[0]
    return list[~(list==index)]

@jit('i8[:](f8[:, :], f8, i8)', nopython=True)
def avoidance_neighbor_self_included(matrix, threshold, index):
    return np.where((matrix[:, index] < threshold) == 1)[0]

@jit('UniTuple(f8, 3)(f8, f8, f8)',nopython=True)
def unite_vector(sx, sy, sz):
    norm = np.sqrt(sx*sx+sy*sy+sz*sz)
    return sx/norm, sy/norm, sz/norm

@jit('UniTuple(f8, 3)(f8[:], f8[:], f8[:])',nopython=True)
def center_of_mass(sx, sy, sz):
    return np.mean(sx), np.mean(sy), np.mean(sz)

@jit('f8[:, :](f8[:, :], f8[:, :])', nopython=True)
def distance_matrix(X, D):
    M = X.shape[0]
    N = X.shape[1]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

@jit('UniTuple(f8, 3)(f8, f8, f8, f8, f8, f8)', nopython=True)
def make_new_arg(shita1, phi1, arg1, shita2, phi2, arg2):

    cx1, cy1, cz1 = compute_xyz_single(shita1, phi1)
    cx2, cy2, cz2 = compute_xyz_single(shita2, phi2)

    center_x = (cx1 + cx2)/2
    center_y = (cy1 + cy2)/2
    center_z = (cz1 + cz2)/2
    tmp = np.sqrt(center_x*center_x+center_y*center_y+center_z*center_z)

    center_x = center_x/tmp
    center_y = center_y/tmp
    center_z = center_z/tmp

    arg = np.arccos(center_x * cx1 + center_y * cy1 + center_z * cz1)
    center_phi, center_shita = compute_shita_phi_single(center_x, center_y, center_z)

    if arg1 + arg2 - arg > 0:
        if arg1 > (2*arg + arg2):
            arg_new = arg2
        elif arg2 > (2*arg + arg1):
            arg_new = arg1
        else:
            arg_new = (arg1 + arg2 - arg)/2
    else:
        arg_new = 2 * np.pi


    return center_phi, center_shita, arg_new