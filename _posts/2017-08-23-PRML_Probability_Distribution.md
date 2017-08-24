---
layout: post
title: PRML Chapter2 Probability Distribution
category: PRML
tags: machine_learning
---

* content
{:toc}
### 1. 共轭先验

现在假设我们有这样几类概率：$P(\theta)$（先验分布）， $P(\theta\mid X)$（后验分布），$p(X), p(X \mid \theta)$ （似然函数）它们之间的关系可以通过贝叶斯公式进行连接： 




$$
后验分布 = 似然函数\times 先验分布/ P(X)
$$




之所以采用共轭先验的原因是可以使得先验分布和后验分布的形式相同，这样一方面合符人的直观（它们应该是相同形式的）另外一方面是可以形成一个先验链，即现在的后验分布可以作为下一次计算的先验分布，如果形式相同，就可以形成一个链条。为了使得先验分布和后验分布的形式相同，我们定义：如果先验分布和似然函数可以使得先验分布和后验分布有相同的形式，那么就称先验分布与似然函数**是共轭的**

一些常见的共轭先验：

| Distribution                             | Conjugate Prior             |
| ---------------------------------------- | --------------------------- |
| Bernoulli                                | Beta distribution           |
| Multinomial                              | Dirichlet distribution      |
| Gaussian , Given variance, mean unknown  | Gaussian distribution       |
| Gaussian, Given mean, variance unknown   | Gamma distribution          |
| Gaussian, both mean and variance unknown | Gaussian-Gamma distribution |

Conjugate prior的意义：方便进行Bayesian inference，甚至是sequential Bayesian inference. 何何为sequential Bayesian inference？：

得到一个 observation 后，可以算出 posterior；由于选取的是共轭先验，所以 posterior 和原来的 prior 形式一样，可以把该 posterior 当作新的 prior，用于下一个 observation，如此迭代下去。对于 stream of data 的情况，这种方式可以实现 real-time learning。

分布的共轭先验可推导的基本思路为：先确定参数在 prior 中具有的 functional form，然后归一化求系数。

很多的分布具有十分有趣的性质，比如有些先验分布可以证明，随着数据的增加方差越来越小，分布越来越陡，最后坍缩成狄拉克函数，这时贝叶斯方法和频率派方法是等价的。举个第三章的贝叶斯线性回归的例子，对于中间参数W的高斯先验分布，随着数据不断增加，参数后验分布的不确定性逐渐减少，朝一个点坍缩。



### 2. 多元高斯分布

一般来说一元高斯分布函数可以在R上的积分计算，而且任意 random vector 的协方差矩阵是非负定的。在这里需要证明和解决的是证明多元高斯的任意 random vector 的协方差矩阵是非负定的。

多元高斯分布的形式为：


$$
N(x|\mu , \Sigma) = \frac{1}{(2\pi )^{D/2}} \frac{1}{|\Sigma|^{1/2}} exp \bigl\lbrace -\frac{1}{2} (\textbf{x}-\mathbf{\mu})^T \Sigma^{-1} (\textbf{x}- \mu)\bigr\rbrace
$$


多元高斯的特点：

1. 参数太多，计算复杂（协方差矩阵是维度的平方级的参数个数）

2. 其函数为单锋函数，建模能力有限，所以可以进行拓展，支持multimodal function。 introducing discrete latent variables：例如 Gaussian Mixtures model。

   ​

### 3. 指数族分布

指数族的分布形式为：


$$
p(x\mid \eta) = h(x)\cdot g(\eta) \cdot exp\{ \eta^T \cdot \mu(x) \}
$$


其中$\eta$是natural parameter，它跟一个分布通常说的参数可能不同，而是由通常的参数经过变换而来（以符合指数族分别的形式）。

假设用 MLE 方法进行参数估计， 我们首先对下式关于$\eta$求导：


$$
g(\eta) \int h(x) \cdot exp\{  \eta^T \cdot \mu(x)\} dx = 1
$$


并令导数为0后， 得到：


$$
-\triangledown ln(g(\eta)) = E[\mu(x) ] = \frac{1}{N} \sum\limits_{n=1}^N \mu(x_n)
$$


最右端得到的 vector 的各个分量都是指数族分别的充分统计量。



### 4. 无信息先验

这句话很值得思考：The prior intended to have as litter influence on the posterior distribution as possible.

一般来说，当参数的取值有界时，均匀分布是无信息先验；当参数的取值无界时，均匀分布不能 normalize，是 improper 的。Translation invariant 和 Scale invariant 的两类分布的无信息先验。



### 5. 无参数方法(Nonparametric methods)

一般来说我们需要在这里解决的问题是：给定 D 维空间中观察到的 N 个数据样本， 估计密度函数 p(x) 

方法：

1. 在足够小的区域 R 中考虑问题。任取一个点 x， 设落入 R 的概率是 P。设观察到N 个样本，则 R 中落入 K 个点的概率是分别 $Bin(K\mid N,P)$， Bin为二项分布；
2. 由于 R 足够小，所以 p(x)在 R 中近似常数，所以：$P = p(x) \times V$， V 是 R 的测度；
3. 由于 N 足够大，二项分别 $Bin(K\mid N,P)$的取值集中在均值 $N\times P$ 附近，即：$K = N \times P$。
4. 以上两式联立，可以得到区域 R 上的密度函数近似值：$ p(x) = \frac{K} {(N\times V)}$。

**Kernel density estimator**

固定 V（一个超立方体），在数据集上计算 V 范围内的 K。 常采用高斯函数做 smoothing kernel function。

为了计算某个区域内的样本的点数，我们可以定义一个kernel function：


$$
k(\mu) = \begin{cases} &1, &|\mu_i| \leq 1/2, i=1,\dots , D  \\ &0 , &otherwise  \end{cases}
$$


所以如果$k(\frac{x-x_n}{h}) = 1$，那么表示的就是数据点$x_n$落在以x为中心，边长为 h 的 hypercube 中；否则
不在该 hypercube 中。那么在 N 个点中，落入该 hypercube 中的点数是：


$$
K = \sum\limits_{n=1}^N = k(\frac{x-x_n}{h})
$$


由此我们当然就可以求出 hypercube 中的概率密度。这种 kernel function 将导致 artificial discontinuity，因此需要平滑的函数，通常是 Gaussian。

但是在这里h的大小比较难以确定，是一个缺点。



