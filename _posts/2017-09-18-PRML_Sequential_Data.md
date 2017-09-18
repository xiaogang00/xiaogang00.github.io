---
layout: post
title: PRML Chapter13 Sequential Data
category: PRML
tags: machine_learning
---

* content
{:toc}






### Hidden Markov Model (HMM)

HMM 的定义：对于每个 observation $x_n$，都有一个相应的 discrete latant variable $z_n$，二者的维度尽可不同；而 latent variable 形成一个 Markov chain。因此 HMM 可以表示成一个 Bayesian Network。

考虑 latent Markov chain 是 homogeneous，也就是说， transition probability $p(z_n \mid z_{n-1})$对任意的n是一样的。由于$z_n$是离散的，假设其可以取K个不同的值(K个状态)，那么可以用1-of-K的编码方式来表示$z_n$。并且 transition probability $p(z_n \mid z_{n-1})$可以表示成一个 K-by-K的矩阵A。（ A 的每个 entry 是一个概率值，并且每一行都是归一的）。

现在给出HMM的描述：


$$
p(z_n\mid z_{n-1},A) = \prod\limits_{k=1}^K  \prod\limits_{j=1}^K  A_{jk}^{z_{n-1,j} z_{nk}}
$$


$p(z_1) = \prod\limits_{k=1}^K \pi_k^{z_{1k}}$，其中$\sum\limits_k \pi_k = 1, 0 \leq \pi_k \leq 1$

$p(x_n\mid z_n ,\phi) = \prod\limits_{k=1}^K p(x_n \mid \phi_k) ^{z_{nk}}$



**HMM 的 likelihood function**
与之前很多模型不同， HMM 中获得的 observation $\{ x_n:n=1,\dots,N \}$ 并不是独立同分布的。因此 likelihood function 不能够写成 the product over all data points of the probability evaluated at each data point。 HMM 的 likelihood 可按照 Bayesian Network 的分解写出：


$$
p(X,Z\mid \theta) = p(z_1\mid \pi) [ \prod\limits_{n=2}^N p(z_n\mid z_{n-1}, A)][\prod\limits_{m=1}^N p(x_m\mid z_m,\phi)]
$$


其中，$X = \{ x_n: n =1,\dots,N \}，Z=\{ z_n:n=1,\dots,N \}, \theta = \{\pi, A,\phi\}$，可见HMM的全部参数是$\theta = \{ \pi,A,\phi \}$。

**HMM 的 learning： Maximum Likelihood EM**

所谓 learning，就是 根据 observation $X = \{ x_n:n=1,\dots,N \}$ 来学习 HMM 的参数$\theta = \{\pi, A,\phi \}$。采用 MLE 进行参数估计。由于含有 latent variable，所以只能求得 observation 的 likelihood，即$p(X\mid \theta) = \sum\limits_Z p(X,Z\mid \theta)$。用 EM 算法求解使该 likelihood 最大化的参数$\theta$。

在 EM 算法迭代中，将会涉及到一个 HMM 的 inference 问题，就是计算 local posterior marginals for latent variables。 包括两个：


$$
\begin{align} \gamma(z_n) &= p(z_n\mid X,\theta^{old}) \\\xi (z_{n-1}, z_n) &= p(z_{n-1},z_n \mid X,\theta^{old})  \end{align}
$$


Forward-backward 算法可计算这两个 marginal。由于 HMM 的 factor graph 是一颗树，所以也可以用 PGM 通用的 sum-product 算法。

**HMM 的 inference： Viterbi 算法**
所谓 inference，就是计算 marginal 或者 conditional (表现为后验概率)。 HMM 中最重要的inference 就是计算：$Z^{\ast} = \arg \max\limits_{Z} p(Z\mid X,\theta)$。其含义是，对于给定的 HMM（也就是给定$\theta$），在观察到 $X = \{ x_n:n=1,\dots,N \}$的情况下，最可能的 latent states（ sequence of states）是什么。 求解的方法是在 PGM 通用的 max-sum 算法，在 HMM 中则对应为 Viterbi 算法。



### Linear Dynamical System (LDS)

LDS 具有和 HMM 完全一个的 PGM，其所不同的是： LDS 的 latent variables 是连续的；并且其 transition probabilisty 与 emission probabilisty 都是 linear-Gaussion。所以 LDS 的描述是：


$$
\begin{align} p(z_n\mid z_{n-1}) &= N(z_n\mid Az_{n-1}, \Gamma) \\ p(x_n\mid z_n) &= N(x_n \mid Cz_n, \Sigma) \\p(z_1) &= N(z_1\mid \mu_0, V_0) \end{align}
$$


