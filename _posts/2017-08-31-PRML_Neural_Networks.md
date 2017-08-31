---
layout: post
title: PRML Chapter5 Neural Network
category: PRML
tags: machine_learning
---

* content
{:toc}






### Neural Network的定义

主要是在我们已经知道的Generalized Linear Model 的基础上在加入一个新的 nonlinear function：


$$
y_k(x,w) = f(\sum\limits_{j=0}^M w_{kj}^{(2)}h(\sum\limits_{i=0}^Dw_{ji}^{(1)}x_i))
$$


在这里$y_k(x,w)$是神经网络的第k个输出。M为隐藏层的unit的个数，D 为输入数据点 x 的维度； h 为 nonlinear
function， $f$ 为 linear 或 nonlinear function。

通常 h 取 logistic 函数，那么可见神经网络其实是 logistic regression 的推广而已。但这一推广意义重大，理论上它可以以任意精度一致逼近任意连续函数；而且，这个三层的神经网络能力与更多层的神经网络相同。



### Neural Network的优化目标函数

1. 对于回归的问题，一般采用sum-of-squares error function：
   $$
   E(w) = \frac{1}{2} \sum\limits_{n=1}^N \parallel y(x_n,w)-t_n \parallel^2
   $$
   在这里$y(x_n,w)$是一个K维的输出向量；$x_n$是一个D维的输入向量；$t_n$是目标的输出向量。

2. 对于分类的问题可以采用 negative logarithm of likelihood function
   $$
   E(w) = -\sum\limits_{n=1}^N \sum\limits_{k=1}^K t_{nk} ln[y_k(x_n,w)]
   $$
   在这里输入的数据$x_n$的目标输出值$t_n$采用1-of-K的coding scheme（如果一个 input 属于第 k类，则除 $t_{nk} =1$外其他的分量都是 0）。



### 目标函数的优化方法

也就是给定 observation data，求出使得 $E(w)$最小的参数 w。采用的方法是 Gradient decent，具体分为两种：

1. Off-line gradient 或 batch gradient。这个类型的算法仍旧有许多，如 steepest decent，conjugate gradients 和 quasi-Newton methods。其主要的特点是一次性考虑全部的 observation data；
2. On-line gradient decent 或 sequential gradient decent 或 stochastic gradient decent。这种方法的特点是observation data 可以是逐个考虑，也可以随机选取。



### BP计算过程

1. 在当前的w值下，计算每个节点的输入和输出值。这其实就是神经网络的 evaluation （ forward propagation）；
2. 根据$E(w)$，计算输出节点（output layer）的$\delta_k$；
3. 进行BP中的反向传播，计算每一个隐藏层的$\delta_j$；
4. 最后是得出结果，也就是 $E(w)$的梯度。



### 前向传播

Evaluation 指的是： 给定已知参数 w 的神经网络和一个 D 维 input vector $x$，计算相应的output vector。

这个比较容易，方法称为 forward propagation，具体而言，就是用上面曾用到的公式：


$$
a_j=\sum\limits_{i} w_{ji}z_i,z_j=h(a_j)
$$


之后可以进行逐层的计算，时间复杂度为$O(W)$。



### 神经网络正则化

对于 Frequentist method 来说，有两种方法：

1. regularizer：跟 regression 中一样， 往目标优化函数中加入一个 w 的 penalty 函数。通常是 quadratic。 但这种 regularizer 不满足神经网络的 linear transformation invariance 要求；
2. early stopping：即用一个 validation set，来确定恰当的迭代停止时间。



