# %%
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import time
import random

# 固定随机种子，确保结果可重复
np.random.seed(42)
random.seed(42)

# 导入自定义模块（请确保路径正确）
from zxgj1.llz import clllz1
from zxgj1.validpt import Vpt
from zxgj1.工具 import f变量变形
from zxgj1.下一代储层计算 import c下一代储层计算

vpt = Vpt.vpt3
plt.rcParams.update({'font.size': 6.5})

# ----------------------------- 数据加载 ---------------------------------
xx = np.load('data/llzc50.npz')
ww1_original, yy = xx['arr_0'], xx['arr_1'][::2]
xx1 = yy[1000:15000]
xx2 = yy[15000:20000]
suo = np.ceil(np.max(xx1) + 1)
xh1 = xx1 / suo
xh2 = xx2 / suo

# 全局参数（正则化指数从 -1.0 到 -9.5，步长 -0.5）
alzhishu = np.arange(-1.0, -9.5, -0.5)   # 共17个值
alwhen = 10 ** alzhishu
qstart = 100
l1, l2 = xh1.shape
Sm = int(l2 / 3)   # 50
print(f"振子数: {Sm}")
xh1gai = f变量变形(xh1)

# 存储统计信息（平均训练时间）
stats = {
    'ING-RC': {'train_times': []},
    'SCNG-RC': {'train_times': []},
    'ZCNG-RC': {'train_times': []},
    'JCNG-RC': {'train_times': []},
    'RCNG-RC': {'train_times': []},
    'CNG-RC': {'train_times': []}
}

# =============================== 1. ING-RC ===============================
print("\n===== 1. ING-RC =====")
A_ing = []
yc2 = zeros((1100, 3*Sm))
for ii in range(Sm):
    A_ing.append(c下一代储层计算())
    yc2[:, ii*3:(ii+1)*3] = A_ing[ii].m总程序(xh1gai[ii], dt=1, grla=1e-6)
print("ING-RC initial VPT:", vpt(yc2[100:], xh2, 0.5))

vpt12 = []
yc_ing = zeros((2100, 3*Sm))
train_times_ing = []

for alw in alwhen:
    t_train = time.time()
    wouts = []
    for ii in range(Sm):
        qxx_i = xh1gai[ii][qstart+1:]
        qRR_i = A_ing[ii].RR[qstart:-1]
        wout = A_ing[ii].qiujie(qxx_i, qRR_i, alw)
        wouts.append(wout)
    t_train_end = time.time()
    train_times_ing.append(t_train_end - t_train)
    for ii in range(Sm):
        rrr_i = A_ing[ii].RR[-1].copy()
        yc_ing[:, ii*3:(ii+1)*3] = A_ing[ii].yuce(rrr_i, wouts[ii], 2000)
    af = vpt(yc_ing[100:], xh2, 0.5)
    vpt12.append(af)
    print(alw, ' ', A_ing[0].wu.max(0), af)

stats['ING-RC']['train_times'] = train_times_ing

# =============================== 2. SCNG-RC（原始耦合） ===============================
print("\n===== 2. SCNG-RC (original coupling) =====")
cc_sc = []
for ii in range(Sm):
    zz = []
    for jj in range(Sm):
        if ww1_original[ii, jj] > 0.01:
            zz.append(jj)
    cc_sc.append(zz)

dLt, dt, start, qstart, grla, n = 3, 1, 80, 100, 1e-6, 1000
线性项个数 = dLt * 3
非线性项个数1 = int((线性项个数 + 1) * 线性项个数 / 2)
非线性项个数2_sc = np.zeros(Sm, dtype=int)
线性项个数2_sc = np.zeros(Sm, dtype=int)
for ii in range(Sm):
    线性项个数2_sc[ii] = len(cc_sc[ii]) * 线性项个数
    非线性项个数2_sc[ii] = 线性项个数 * len(cc_sc[ii]) * 线性项个数
总项个数_sc = 线性项个数 + 线性项个数2_sc + 非线性项个数1 + 非线性项个数2_sc + 1

RR_sc = []
for ii in range(Sm):
    RR_sc.append(np.zeros((l1, 总项个数_sc[ii])))

for zz in range(Sm):
    olin = np.c_[xh1gai[zz][start:,:],
                 xh1gai[zz][start-dt:-dt,:],
                 xh1gai[zz][start-2*dt:-2*dt,:]]
    RR_sc[zz][start:,:线性项个数] = olin

for zz in range(Sm):
    xu = 线性项个数
    for ij in cc_sc[zz]:
        RR_sc[zz][:, xu:xu+线性项个数] = RR_sc[ij][:, :线性项个数]
        xu += 线性项个数
    for i in range(线性项个数):
        for j in range(i, 线性项个数):
            RR_sc[zz][start:, xu] = RR_sc[zz][start:,:线性项个数][:,i] * RR_sc[zz][start:,:线性项个数][:,j]
            xu += 1
    for i in range(线性项个数):
        for ij in cc_sc[zz]:
            for j in range(线性项个数):
                RR_sc[zz][:, xu] = RR_sc[zz][:, i] * RR_sc[ij][:, j]
                xu += 1
    RR_sc[zz][:, xu] = 1

qrr_sc = [RR_sc[ii][qstart:-1] for ii in range(Sm)]
qxx_sc = [xh1gai[ii][qstart+1:] for ii in range(Sm)]
rrzhong_init = [RR_sc[ii][-1].copy() for ii in range(Sm)]

vpt13 = []
train_times_sc = []

for alw in alwhen:
    t_train = time.time()
    wouts_sc = []
    for ii in range(Sm):
        wout = A_ing[ii].qiujie(qxx_sc[ii], qrr_sc[ii], alw)
        wouts_sc.append(wout)
    t_train_end = time.time()
    train_times_sc.append(t_train_end - t_train)
    rrzhong = [row.copy() for row in rrzhong_init]
    yc_sc = np.zeros((Sm, 100+n, 3))
    yc_sc[:, :100] = xh1gai[:, -100:]
    for it in range(100, 100+n):
        for zz in range(Sm):
            yc_sc[zz][it] = wouts_sc[zz] @ rrzhong[zz]
        for zz in range(Sm):
            olin = np.r_[yc_sc[zz][it, :], yc_sc[zz][it-dt, :], yc_sc[zz][it-2*dt, :]]
            rrzhong[zz][:线性项个数] = olin
        for zz in range(Sm):
            xu = 线性项个数
            for ij in cc_sc[zz]:
                rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
                xu += 线性项个数
            for i1 in range(线性项个数):
                for i2 in range(i1, 线性项个数):
                    rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                    xu += 1
            for i1 in range(线性项个数):
                for ij in cc_sc[zz]:
                    for jj in range(线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                        xu += 1
    yct = f变量变形(yc_sc)
    af = vpt(yct[100:], xh2, 0.5)
    vpt13.append(af)
    print(alw, ' ', A_ing[0].wu.max(0), af)

stats['SCNG-RC']['train_times'] = train_times_sc

# ========== 辅助函数：随机添加/删除边 ==========
def add_edges(matrix, num_edges=2):
    new_mat = matrix.copy()
    Sm = matrix.shape[0]
    possible = [(i, j) for i in range(Sm) for j in range(Sm) if i != j and new_mat[i, j] == 0]
    chosen = np.random.choice(len(possible), size=num_edges, replace=False)
    for idx in chosen:
        i, j = possible[idx]
        new_mat[i, j] = 1.0
    return new_mat

def remove_edges(matrix, num_edges=2):
    new_mat = matrix.copy()
    Sm = matrix.shape[0]
    existing = [(i, j) for i in range(Sm) for j in range(Sm) if i != j and new_mat[i, j] > 0]
    chosen = np.random.choice(len(existing), size=num_edges, replace=False)
    for idx in chosen:
        i, j = existing[idx]
        new_mat[i, j] = 0.0
    return new_mat

# =============================== 3. ZCNG-RC（添加2条边，跑5次取平均） ===============================
print("\n===== 3. ZCNG-RC (original + 2 random edges, average over 5 runs) =====")
n_runs = 5
vpt15_acc = np.zeros(len(alwhen))
train_times_zc_acc = np.zeros(len(alwhen))
for run in range(n_runs):
    print(f"  Run {run+1}/{n_runs}")
    ww1_zc = add_edges(ww1_original, num_edges=2)
    cc_zc = []
    for ii in range(Sm):
        zz = []
        for jj in range(Sm):
            if ww1_zc[ii, jj] > 0.01:
                zz.append(jj)
        cc_zc.append(zz)
    线性项个数2_zc = np.zeros(Sm, dtype=int)
    非线性项个数2_zc = np.zeros(Sm, dtype=int)
    for ii in range(Sm):
        线性项个数2_zc[ii] = len(cc_zc[ii]) * 线性项个数
        非线性项个数2_zc[ii] = 线性项个数 * len(cc_zc[ii]) * 线性项个数
    总项个数_zc = 线性项个数 + 线性项个数2_zc + 非线性项个数1 + 非线性项个数2_zc + 1
    RR_zc = []
    for ii in range(Sm):
        RR_zc.append(np.zeros((l1, 总项个数_zc[ii])))
    for zz in range(Sm):
        olin = np.c_[xh1gai[zz][start:,:],
                     xh1gai[zz][start-dt:-dt,:],
                     xh1gai[zz][start-2*dt:-2*dt,:]]
        RR_zc[zz][start:,:线性项个数] = olin
    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc_zc[zz]:
            RR_zc[zz][:, xu:xu+线性项个数] = RR_zc[ij][:, :线性项个数]
            xu += 线性项个数
        for i in range(线性项个数):
            for j in range(i, 线性项个数):
                RR_zc[zz][start:, xu] = RR_zc[zz][start:,:线性项个数][:,i] * RR_zc[zz][start:,:线性项个数][:,j]
                xu += 1
        for i in range(线性项个数):
            for ij in cc_zc[zz]:
                for j in range(线性项个数):
                    RR_zc[zz][:, xu] = RR_zc[zz][:, i] * RR_zc[ij][:, j]
                    xu += 1
        RR_zc[zz][:, xu] = 1
    qrr_zc = [RR_zc[ii][qstart:-1] for ii in range(Sm)]
    qxx_zc = [xh1gai[ii][qstart+1:] for ii in range(Sm)]
    vpt_run = []
    train_run = []
    for idx, alw in enumerate(alwhen):
        t_train = time.time()
        wouts_zc = []
        for ii in range(Sm):
            wout = A_ing[ii].qiujie(qxx_zc[ii], qrr_zc[ii], alw)
            wouts_zc.append(wout)
        t_train_end = time.time()
        train_run.append(t_train_end - t_train)
        rrzhong = [RR_zc[ii][-1].copy() for ii in range(Sm)]
        yc = np.zeros((Sm, 100+n, 3))
        yc[:, :100] = xh1gai[:, -100:]
        for it in range(100, 100+n):
            for zz in range(Sm):
                yc[zz][it] = wouts_zc[zz] @ rrzhong[zz]
            for zz in range(Sm):
                olin = np.r_[yc[zz][it, :], yc[zz][it-dt, :], yc[zz][it-2*dt, :]]
                rrzhong[zz][:线性项个数] = olin
            for zz in range(Sm):
                xu = 线性项个数
                for ij in cc_zc[zz]:
                    rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
                    xu += 线性项个数
                for i1 in range(线性项个数):
                    for i2 in range(i1, 线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                        xu += 1
                for i1 in range(线性项个数):
                    for ij in cc_zc[zz]:
                        for jj in range(线性项个数):
                            rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                            xu += 1
        yct = f变量变形(yc)
        af = vpt(yct[100:], xh2, 0.5)
        vpt_run.append(af)
    vpt15_acc += np.array(vpt_run)
    train_times_zc_acc += np.array(train_run)

vpt15 = vpt15_acc / n_runs
train_times_zc = train_times_zc_acc / n_runs
stats['ZCNG-RC']['train_times'] = train_times_zc.tolist()
print("ZCNG-RC average VPS (over 5 runs):", vpt15[:3])

# =============================== 4. JCNG-RC（删除2条边，跑5次取平均） ===============================
print("\n===== 4. JCNG-RC (original - 2 random edges, average over 5 runs) =====")
vpt16_acc = np.zeros(len(alwhen))
train_times_jc_acc = np.zeros(len(alwhen))
for run in range(n_runs):
    print(f"  Run {run+1}/{n_runs}")
    ww1_jc = remove_edges(ww1_original, num_edges=2)
    cc_jc = []
    for ii in range(Sm):
        zz = []
        for jj in range(Sm):
            if ww1_jc[ii, jj] > 0.01:
                zz.append(jj)
        cc_jc.append(zz)
    线性项个数2_jc = np.zeros(Sm, dtype=int)
    非线性项个数2_jc = np.zeros(Sm, dtype=int)
    for ii in range(Sm):
        线性项个数2_jc[ii] = len(cc_jc[ii]) * 线性项个数
        非线性项个数2_jc[ii] = 线性项个数 * len(cc_jc[ii]) * 线性项个数
    总项个数_jc = 线性项个数 + 线性项个数2_jc + 非线性项个数1 + 非线性项个数2_jc + 1
    RR_jc = []
    for ii in range(Sm):
        RR_jc.append(np.zeros((l1, 总项个数_jc[ii])))
    for zz in range(Sm):
        olin = np.c_[xh1gai[zz][start:,:],
                     xh1gai[zz][start-dt:-dt,:],
                     xh1gai[zz][start-2*dt:-2*dt,:]]
        RR_jc[zz][start:,:线性项个数] = olin
    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc_jc[zz]:
            RR_jc[zz][:, xu:xu+线性项个数] = RR_jc[ij][:, :线性项个数]
            xu += 线性项个数
        for i in range(线性项个数):
            for j in range(i, 线性项个数):
                RR_jc[zz][start:, xu] = RR_jc[zz][start:,:线性项个数][:,i] * RR_jc[zz][start:,:线性项个数][:,j]
                xu += 1
        for i in range(线性项个数):
            for ij in cc_jc[zz]:
                for j in range(线性项个数):
                    RR_jc[zz][:, xu] = RR_jc[zz][:, i] * RR_jc[ij][:, j]
                    xu += 1
        RR_jc[zz][:, xu] = 1
    qrr_jc = [RR_jc[ii][qstart:-1] for ii in range(Sm)]
    qxx_jc = [xh1gai[ii][qstart+1:] for ii in range(Sm)]
    vpt_run = []
    train_run = []
    for idx, alw in enumerate(alwhen):
        t_train = time.time()
        wouts_jc = []
        for ii in range(Sm):
            wout = A_ing[ii].qiujie(qxx_jc[ii], qrr_jc[ii], alw)
            wouts_jc.append(wout)
        t_train_end = time.time()
        train_run.append(t_train_end - t_train)
        rrzhong = [RR_jc[ii][-1].copy() for ii in range(Sm)]
        yc = np.zeros((Sm, 100+n, 3))
        yc[:, :100] = xh1gai[:, -100:]
        for it in range(100, 100+n):
            for zz in range(Sm):
                yc[zz][it] = wouts_jc[zz] @ rrzhong[zz]
            for zz in range(Sm):
                olin = np.r_[yc[zz][it, :], yc[zz][it-dt, :], yc[zz][it-2*dt, :]]
                rrzhong[zz][:线性项个数] = olin
            for zz in range(Sm):
                xu = 线性项个数
                for ij in cc_jc[zz]:
                    rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
                    xu += 线性项个数
                for i1 in range(线性项个数):
                    for i2 in range(i1, 线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                        xu += 1
                for i1 in range(线性项个数):
                    for ij in cc_jc[zz]:
                        for jj in range(线性项个数):
                            rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                            xu += 1
        yct = f变量变形(yc)
        af = vpt(yct[100:], xh2, 0.5)
        vpt_run.append(af)
    vpt16_acc += np.array(vpt_run)
    train_times_jc_acc += np.array(train_run)

vpt16 = vpt16_acc / n_runs
train_times_jc = train_times_jc_acc / n_runs
stats['JCNG-RC']['train_times'] = train_times_jc.tolist()
print("JCNG-RC average VPS (over 5 runs):", vpt16[:3])

# =============================== 5. RCNG-RC（随机重连，边数相同，跑5次取平均） ===============================
print("\n===== 5. RCNG-RC (random rewiring, same edges, average over 5 runs) =====")
original_edges = np.sum(ww1_original > 0) - np.trace(ww1_original > 0)
print(f"Original graph has {original_edges} directed edges")
vpt17_acc = np.zeros(len(alwhen))
train_times_rc_acc = np.zeros(len(alwhen))
for run in range(n_runs):
    print(f"  Run {run+1}/{n_runs}")
    ww1_rc = np.zeros((Sm, Sm))
    all_positions = [(i, j) for i in range(Sm) for j in range(Sm) if i != j]
    chosen = np.random.choice(len(all_positions), size=original_edges, replace=False)
    for idx in chosen:
        i, j = all_positions[idx]
        ww1_rc[i, j] = 1.0
    cc_rc = []
    for ii in range(Sm):
        zz = []
        for jj in range(Sm):
            if ww1_rc[ii, jj] > 0.01:
                zz.append(jj)
        cc_rc.append(zz)
    线性项个数2_rc = np.zeros(Sm, dtype=int)
    非线性项个数2_rc = np.zeros(Sm, dtype=int)
    for ii in range(Sm):
        线性项个数2_rc[ii] = len(cc_rc[ii]) * 线性项个数
        非线性项个数2_rc[ii] = 线性项个数 * len(cc_rc[ii]) * 线性项个数
    总项个数_rc = 线性项个数 + 线性项个数2_rc + 非线性项个数1 + 非线性项个数2_rc + 1
    RR_rc = []
    for ii in range(Sm):
        RR_rc.append(np.zeros((l1, 总项个数_rc[ii])))
    for zz in range(Sm):
        olin = np.c_[xh1gai[zz][start:,:],
                     xh1gai[zz][start-dt:-dt,:],
                     xh1gai[zz][start-2*dt:-2*dt,:]]
        RR_rc[zz][start:,:线性项个数] = olin
    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc_rc[zz]:
            RR_rc[zz][:, xu:xu+线性项个数] = RR_rc[ij][:, :线性项个数]
            xu += 线性项个数
        for i in range(线性项个数):
            for j in range(i, 线性项个数):
                RR_rc[zz][start:, xu] = RR_rc[zz][start:,:线性项个数][:,i] * RR_rc[zz][start:,:线性项个数][:,j]
                xu += 1
        for i in range(线性项个数):
            for ij in cc_rc[zz]:
                for j in range(线性项个数):
                    RR_rc[zz][:, xu] = RR_rc[zz][:, i] * RR_rc[ij][:, j]
                    xu += 1
        RR_rc[zz][:, xu] = 1
    qrr_rc = [RR_rc[ii][qstart:-1] for ii in range(Sm)]
    qxx_rc = [xh1gai[ii][qstart+1:] for ii in range(Sm)]
    vpt_run = []
    train_run = []
    for idx, alw in enumerate(alwhen):
        t_train = time.time()
        wouts_rc = []
        for ii in range(Sm):
            wout = A_ing[ii].qiujie(qxx_rc[ii], qrr_rc[ii], alw)
            wouts_rc.append(wout)
        t_train_end = time.time()
        train_run.append(t_train_end - t_train)
        rrzhong = [RR_rc[ii][-1].copy() for ii in range(Sm)]
        yc = np.zeros((Sm, 100+n, 3))
        yc[:, :100] = xh1gai[:, -100:]
        for it in range(100, 100+n):
            for zz in range(Sm):
                yc[zz][it] = wouts_rc[zz] @ rrzhong[zz]
            for zz in range(Sm):
                olin = np.r_[yc[zz][it, :], yc[zz][it-dt, :], yc[zz][it-2*dt, :]]
                rrzhong[zz][:线性项个数] = olin
            for zz in range(Sm):
                xu = 线性项个数
                for ij in cc_rc[zz]:
                    rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
                    xu += 线性项个数
                for i1 in range(线性项个数):
                    for i2 in range(i1, 线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                        xu += 1
                for i1 in range(线性项个数):
                    for ij in cc_rc[zz]:
                        for jj in range(线性项个数):
                            rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                            xu += 1
        yct = f变量变形(yc)
        af = vpt(yct[100:], xh2, 0.5)
        vpt_run.append(af)
    vpt17_acc += np.array(vpt_run)
    train_times_rc_acc += np.array(train_run)

vpt17 = vpt17_acc / n_runs
train_times_rc = train_times_rc_acc / n_runs
stats['RCNG-RC']['train_times'] = train_times_rc.tolist()
print("RCNG-RC average VPS (over 5 runs):", vpt17[:3])

# =============================== 6. CNG-RC ===============================
print("\n===== 6. CNG-RC (coupling with neighbors' linear and quadratic features) =====")
cc_cng = []
for ii in range(Sm):
    zz = []
    for jj in range(Sm):
        if ww1_original[ii, jj] > 0.01:
            zz.append(jj)
    cc_cng.append(zz)

线性项个数_self = dLt * 3
二次项个数_self = int((线性项个数_self + 1) * 线性项个数_self / 2)
邻居线性项个数 = 线性项个数_self
邻居二次项个数 = 二次项个数_self
交叉项个数 = 线性项个数_self * 线性项个数_self

总项个数_cng = []
for ii in range(Sm):
    num_nei = len(cc_cng[ii])
    总项 = (线性项个数_self + 二次项个数_self +
            num_nei * (邻居线性项个数 + 邻居二次项个数 + 交叉项个数) + 1)
    总项个数_cng.append(总项)

RR_cng = []
for ii in range(Sm):
    RR_cng.append(np.zeros((l1, 总项个数_cng[ii])))

for zz in range(Sm):
    olin_self = np.c_[xh1gai[zz][start:,:],
                      xh1gai[zz][start-dt:-dt,:],
                      xh1gai[zz][start-2*dt:-2*dt,:]]
    RR_cng[zz][start:, :线性项个数_self] = olin_self
    xu = 线性项个数_self
    for i in range(线性项个数_self):
        for j in range(i, 线性项个数_self):
            RR_cng[zz][start:, xu] = olin_self[:, i] * olin_self[:, j]
            xu += 1
    for ij in cc_cng[zz]:
        olin_nei = np.c_[xh1gai[ij][start:,:],
                         xh1gai[ij][start-dt:-dt,:],
                         xh1gai[ij][start-2*dt:-2*dt,:]]
        RR_cng[zz][start:, xu:xu+邻居线性项个数] = olin_nei
        xu += 邻居线性项个数
        for i in range(邻居线性项个数):
            for j in range(i, 邻居线性项个数):
                RR_cng[zz][start:, xu] = olin_nei[:, i] * olin_nei[:, j]
                xu += 1
        for i in range(线性项个数_self):
            for j in range(邻居线性项个数):
                RR_cng[zz][start:, xu] = olin_self[:, i] * olin_nei[:, j]
                xu += 1
    RR_cng[zz][start:, xu] = 1

qrr_cng = [RR_cng[ii][qstart:-1] for ii in range(Sm)]
qxx_cng = [xh1gai[ii][qstart+1:] for ii in range(Sm)]
rrzhong_cng_init = [RR_cng[ii][-1].copy() for ii in range(Sm)]

vpt18 = []
train_times_cng = []
for alw in alwhen:
    t_train = time.time()
    wouts_cng = []
    for ii in range(Sm):
        wout = A_ing[ii].qiujie(qxx_cng[ii], qrr_cng[ii], alw)
        wouts_cng.append(wout)
    t_train_end = time.time()
    train_times_cng.append(t_train_end - t_train)
    rrzhong = [row.copy() for row in rrzhong_cng_init]
    yc = np.zeros((Sm, 100+n, 3))
    yc[:, :100] = xh1gai[:, -100:]
    for it in range(100, 100+n):
        for zz in range(Sm):
            yc[zz][it] = wouts_cng[zz] @ rrzhong[zz]
        for zz in range(Sm):
            olin_self = np.r_[yc[zz][it, :], yc[zz][it-dt, :], yc[zz][it-2*dt, :]]
            rrzhong[zz][:线性项个数_self] = olin_self
        for zz in range(Sm):
            xu = 线性项个数_self
            for i1 in range(线性项个数_self):
                for i2 in range(i1, 线性项个数_self):
                    rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                    xu += 1
            for ij in cc_cng[zz]:
                rrzhong[zz][xu:xu+邻居线性项个数] = rrzhong[ij][:邻居线性项个数]
                xu += 邻居线性项个数
                for i1 in range(邻居线性项个数):
                    for i2 in range(i1, 邻居线性项个数):
                        rrzhong[zz][xu] = rrzhong[ij][i1] * rrzhong[ij][i2]
                        xu += 1
                for i1 in range(线性项个数_self):
                    for jj in range(邻居线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                        xu += 1
            rrzhong[zz][xu] = 1
    yct = f变量变形(yc)
    af = vpt(yct[100:], xh2, 0.5)
    vpt18.append(af)
    print(alw, ' ', A_ing[0].wu.max(0), af)

stats['CNG-RC']['train_times'] = train_times_cng

# =============================== 可视化：VPS对比曲线（6条曲线） ===============================
plt.figure(dpi=300, figsize=(17/2.54, 5/2.54))
plt.plot(vpt12, 'b+-', label='ING-RC')
plt.plot(vpt13, 'g*-', label='SCNG-RC', markersize=6)   # 星形标记
plt.plot(vpt15, 'm', label='ZCNG-RC (+2 edges, avg 5 runs)')
plt.plot(vpt16, 'orange', label='JCNG-RC (-2 edges, avg 5 runs)')
plt.plot(vpt17, 'purple', label='RCNG-RC (random same edges, avg 5 runs)')
plt.plot(vpt18, 'brown', label='CNG-RC')
plt.xlabel('$\\lg(\\lambda)$')
plt.ylabel('VPS')
plt.legend(loc='upper left')
plt.xticks(range(len(alzhishu)), alzhishu)
plt.title("Comparison of six methods (50 oscillators) - VPS")
plt.show()

# =============================== 训练时间对比 ===============================
print("\n" + "="*80)
print("六种方法训练时间对比（平均值，单位：毫秒）")
print("="*80)
print(f"{'方法':<15} {'平均训练时间(ms)':<20}")
print("-"*80)
for name, data in stats.items():
    avg_train = np.mean(data['train_times']) * 1000 if data['train_times'] else 0
    print(f"{name:<15} {avg_train:<20.2f}")
print("="*80)

plt.figure(dpi=150, figsize=(12, 5))
methods = list(stats.keys())
train_means = [np.mean(stats[m]['train_times']) * 1000 for m in methods]
bars = plt.bar(methods, train_means, color='skyblue')
plt.ylabel('Average Training Time (ms)', fontsize=13)
plt.title('Training Time Comparison of Six Methods (50 oscillators)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(train_means),
             f'{height:.1f}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

# 保存 VPS 结果
ts = time.strftime('%Y%m%d%H%M%S', time.localtime())
np.savez(f'shuju/vpt_6methods_50osc_{ts}.npz',
         alzhishu=alzhishu,
         vpt12=vpt12, vpt13=vpt13, vpt15=vpt15, vpt16=vpt16, vpt17=vpt17, vpt18=vpt18)
print(f"\nVPS results saved to shuju/vpt_6methods_50osc_{ts}.npz")