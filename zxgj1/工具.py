# -*- coding: utf-8 -*-
import numpy as np
from numpy import zeros,ones_like
# from numpy import tanh,eye,sin
from math import log, exp, sqrt
from collections import defaultdict
def f变量变形(xx1 : np.ndarray,qt=3):
    if xx1.ndim==2:
        l1,l2=xx1.shape
        Sm=int(l2/qt)#振子个数
        xx1gai=zeros((Sm,l1,qt))
        for ii in range(Sm):
            xx1gai[ii]=xx1[:,ii*qt:(ii+1)*qt]
    elif xx1.ndim==3:
        l1,l2,l3=xx1.shape
        xx1gai=zeros((l2,l1*l3))
        for ii in range(l1):
            xx1gai[:,ii*l3:(ii+1)*l3]=xx1[ii]
    return xx1gai

def f变量变形1(xx1 : np.ndarray,qt=3):
    if xx1.ndim==2:
        l1,l2=xx1.shape
        Sm=int(l2/qt)#振子个数
        xx1gai1=xx1.reshape(l1,Sm,3)
        xx1gai=np.transpose(xx1gai1,(1,0,2))
    elif xx1.ndim==3:
        l1,l2,l3=xx1.shape
        xx1gai1=np.transpose(xx1,(1,0,2))
        xx1gai=xx1gai1.reshape(l2,l1*l3)
    return xx1gai

def f求解序列(ww1):
    Sm=ww1.shape[0]
    cc=[]
    for ii in range(Sm):
        zz=[]
        for jj in range(Sm):
            if ww1[ii,jj]>0.01:
                zz.append(jj)
        cc.append(zz)
    return cc
def fC(n, m, method='dc'):
    """
    从n个不同元素中选取m个元素的组合数
    C(n, m) = n! / (m! * (n - m)!)
    :param n: int,总元素个数
    :param m: int,需要选择的元素个数
    :param method: str,下述列表字符其中之一:['dc', 'dp', 'log', 'prime']
                   - dc:暴力相除 - dp:动态规划 - log:log求解法，非精确解 - prime:质因数分解法
    :return: res 组合数
    """
    if method == 'dc':
        # 暴力相除，有溢出风险，python最大float约为1.79e+308
        # 时间复杂度 O(n), 空间复杂度O(1)
        nums1, nums2 = 1, 1
        m = min(m, n - m)
        for x in range(1, m + 1):
            nums1 *= x
            nums2 *= x + n - m
        return nums2 // nums1
    elif method == 'dp':
        # DP，较慢
        # 时间复杂度 O(nm) 空间复杂度 O(mn) 可以优化到O(m)
        # 从n个元素中取m个元素可以划分为两个子问题：
        # 对于元素i,选它等于从n-1个元素中选取m-1个元素
        # 对于元素i，不选它等于从n-1个元素中选取m个元素
        # dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
        # 初始化：从i个元素中选取0个元素的组合数为1，从0个元素中选取j(>0)个元素的组合数为0
        # 遍历顺序：从上往下从左往右遍历即可
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(m + 1, i + 1)):
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
        return dp[-1][-1]
    elif method == 'log':
        # log估算法,不够精确
        # 时间复杂度 O(n) 空间复杂度 O(1)
        log_value = 0
        m = min(m, n - m)
        for i in range(1, m + 1):
            log_value = log_value + log(n - m + i) - log(i)
        return int(round(exp(log_value), 0))
    elif method == 'prime':
        # 质数分解法
        # 时间复杂度 O(n ** 2) 空间复杂度 O(n)  看上去很高，但实际运行中质数越往后越稀疏，比DP快很多
        # C的值必为非负整数，分母可以分解为若干质数，分子也可以分解为若干质数，将等量的质数相除，得到纯分母
        # 这里的相除不需要真实相除，只需要计算每个质数的个数然后将个数相减即可
        primes = set()
        for i in range(2, n + 1):
            is_prime = True
            for j in range(2, int(sqrt(i)) + 1):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.add(i)

        n_factors = defaultdict(int)
        m_factors = defaultdict(int)
        m = min(m, n - m)

        for i in range(1, m + 1):
            k = i
            while k not in primes and k > 1:
                for p in primes:
                    if k % p == 0:
                        k = k // p
                        m_factors[p] += 1
                        break
            if k > 1:
                m_factors[k] += 1

            l = n - m + i
            while l not in primes and l > 1:
                for p in primes:
                    if l % p == 0:
                        l = l // p
                        n_factors[p] += 1
                        break
            if l > 1:
                n_factors[l] += 1

        res = 1
        for k, v in n_factors.items():
            res *= k ** (v - m_factors[k])
        return int(res)


if __name__=='__main__':
    """
%timeit xx1gai=f变量变形(xx1)
%timeit xx1gai1=f变量变形1(xx1)
599 µs ± 8.97 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
1.44 µs ± 4.88 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

%timeit xx1fa=f变量变形(xx1gai)
%timeit xx1fa1=f变量变形1(xx1gai)
653 µs ± 7.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
555 µs ± 2.32 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

xx1gai=f变量变形(xx1)
xx1gai1=f变量变形1(xx1)
xx1gai[0,:10,:]=0
xx1不会改变
xx1gai1[0,:10,:]=0
xx1会改变
f变量变形()不会改变原数组;. f变量变形1()会改变原数组

xx1fa=f变量变形(xx1gai)
xx1fa1=f变量变形1(xx1gai)
xx1fa1[:100]=0
xx1fa[:100]=0
xx1gai不会改变
f变量变形()f变量变形1()好像都不会改变原数组

import math
# 返回排列数
def permutation(n, k):
    return math.perm(n, k)
 
# 返回组合数
def combination(n, k):
    return math.comb(n, k)

标准库itertools:
itertools.permutations(iterable,r = None) ：从iterable（可迭代对象，常见的比如，列表，元组，字符串……）中选取r个元素进行排列。

itertools.combinations(iterable,r = None)：同理， 从iterable（可迭代对象，常见的比如，列表，元组，字符串……）中选取r个元素进行组合。






    """