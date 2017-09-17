---
layout: post
title: PRML Chapter12 Continuous Latent Variables
category: PRML
tags: machine_learning
---

* content
{:toc}






### Principal Component Analysis (PCA)

一种 unsupervised learning， 应用于 dimentionality reduction， feature extraction， data visualization 以及 lossy data compression 等。

* 问题一：Maximum Variance Subspace

  假设有 N 个 D 维的 observation data $\{ x_n : n = 1,\dots,N \}$，现在要找到一个 M<D 维的空间，使得 N 个 data 往该空间投影后， variance of the projected data is maximize。

  在M>1维的空间中，variance 无定义。所以可以逐个维度考虑：先找到一个维度$u_1$，使得在维度$u_1$上的 projected data 具有最大的方差；第二次再找一个与$u_1$ 正交的维度$u_2$， 使variance of the projected data 最大。这个过程不断进行，直到找到 M 个正交维度，从而构成一个 M 维空间。

  **Formulation**：首先考虑寻找$u_1$， N 个 observation data 在该向量方向上的 variance of projected data 是：$\frac{1}{N} \sum\limits_{n=1}^N \{ u_1^Tx_n - u_1^T \overline{x} \}^2 = u_1^T S u_1$，其中$\overline{x} = \frac{1}{N}\sum\limits_n x_n$，S是 sample covariance matrix：$S = \frac{1}{N} \sum\limits_n (x_n - \overline{x}) (x_n - \overline{x})^T $。现在问题就是：最大化$u_1^T S u_1$ with respect to $u_1$。为了避免$\parallel u_1 \parallel \rightarrow \infty$，需要对其做限定，合理的限定是让其为单位向量，即$u_1^T u_1 = 1$。

  用 Lagrange multiplier 计算，可以得到：$S u_1 = \lambda_1 u_1$，可见$\lambda_1$是S的特征向量。进一步可以得出$u_1^T S u_1 = \lambda_1$。由于$u_1^T S u_1$是目标函数，因此$\lambda_1$取S的最大特征值可使目标最大化。此时的$u_1$ 是对应最大特征值$\lambda_1$的单位特征向量。

  用于的推导可发现，$u_2$取为 S 的第二大特征值对应的单位特征向量。从而得到了 M 最大特征值对应的特征向量。



* 问题二：Minimum Projection Error

  与上个问题一样，不过此处最小化的是$x_n$与其 projection $\overline{x}_n$之间的差异。

  **Formulation**： 跟 Maximum Variance Subspace 一样，逐个考虑。比如$\frac{1}{N} \sum\limits_n \parallel x_n - (u_1^T x_n) u_1 \parallel^2$，满足约束$u_1^T u_1 = 1$。推导以后发现结果和 Maximum Variance Subspace 是相同的。



### Probabilistic PCA (PPCA)

用 latent variable z 对应 principle-component subspace； observed data x 对应于原来的 data point。并且假设：$p(z) = N(z\mid 0, I)$， $p(x\mid z) = N(x\mid Wz + \mu , \sigma^2 I)$.

W是一个从 principle-component subspace 到原空间的线性变换。这是一个 linear-Gaussian 模型，所以可以相应得到：$p(x) = N(x\mid \mu, C)$，$p(z\mid x) = N(z\mid M^{-1} W^T (x-\mu), \sigma^2 M)$，其中$C = WW^T + \sigma^2 I_D, M = W^T W + \sigma^2 I_M$。

关于 p( ) x ，有一个特点值得注意：任意 latent space 中的 z，对其进行 rotation 后，得到$Rz$， 其中 R 是一个 orthogonal matrix；那么得到的$p(x)$是不变的。因为，设$\tilde{W} = WR$，那么利用R的正交性，$\tilde{W}\tilde{W}^T = WRR^T W^T = WW^T$。

在得到一个 observed data set 后，可以对 PPCA 中的参数$W, \mu,\sigma^2$进行估计。估计的方法可以是直接 MLE （有 closed-form solution），但为了避免矩阵运算，提高计算效率也可以 EM。



### Factor analysis

与PPCA相似：


$$
\begin{align} p(z) &= N(z\mid 0, I) \\ p(x\mid z) &= N(x\mid Wz + \mu, \Psi)  \end{align}
$$


其中$\Psi$是diagonal matrix。这是 Factor analysis 与 PPCA 的唯一不同之处（ PPCA 是一个 isotropic matrix）。



### Kernel PCA

在 PCA 中，从原空间到 principle-component subspace 的 projection 是线性的。而 Kernel PCA
要做的是一个非线性的 projection。

考虑一个非线性的 feature map $\phi(x)$，它将原空间的每一个 data point 映射到高维（设为 M维）feature space中。然后在 feature space进行 PCA，这对应于原空间的非线性 projection。假设 feature space 中的 data point 的均值为 0，即$\sum\limits_n \phi(x_n) = 0$。那么在这个 M 维空间中的 sample covariance 是：$C = \frac{1}{N} \sum\limits_n \phi(x_n) \phi(x_n)^T$。按照 PCA 的套路，应该求得 C 的特征值及特征向量$C \nu_i = \lambda_i \nu_i, i = 1,\dots, M$。跟其他 kernel 方法一样，我们不能直接在这个 M 维空间中解此 eigenvector equation，而应该寻找 kernel trick。

经过一些替换，可以得到$K a_i = \lambda_i N a_i$。其中，K由$K(x_n, x_m) = \phi(x_n)^T \phi(x_m)$构成的$N \times N$ 矩阵。 这样我们就可以通过求解一个 N 为方阵的 eigenvector equation，而避免了直接在 feature space 上求解。用以上方法得到 K 的特征向量后， 就可以在 feature space 中对任意 data point x 进行projection；其向第 i 个特征向量$\nu_i$的投影是：


$$
y_i(x) = \phi(x)^T \nu_i = \sum\limits_{n=1}^N a_{in} k(x,x_n)
$$


上面的过程假设$\sum\limits_n \phi(x_n) =0$。对于一般的情况， 可以对 data 进行 centralize，即$\tilde{\phi}(x_n) = \phi(x_n) - \frac{1}{N} \sum\limits_n \phi(x_n)$。计算出


$$
\tilde{K}_{nm} = \tilde{\phi}(x_n)^T \tilde{\phi}(x_m) \quad and \quad K_{nm} = \phi(x_n)^T \phi(x_m)
$$


之间的关系。用后者把前者表示计算出来，剩下的就是针对$\tilde{K}_{nm}$解 eigenvector equation。



### Nonlinear latent variable models

PPCA 的 latent variable 尽管是 continuous （相较 GMM 和 HMM 的 latent variable 是 discrete）的，但 observed variable 对其的依赖仍旧是线性的，即 linear-Gaussian；而且 latent variable本身的分布是 Gaussian。

**Independent component analysis**

假设： Observed variable 对 latent variable 是线性依赖关系（线性组合），但是 latent variable不再是 Gaussian，而是满足性质$p(z) = \prod\limits_{j=1}^M p(z_j)$的任意分布。

**Autoassociative neural network**
也叫 auto-encoder，是 neural network 用于 unsupervised learning 的一种方法。 这种 NN 的网络结构是： D 个 inputs， D 个 outputs， M (< D)个 hidden units；构成 3 层网络。 优化的目标函数是：$E(w) = \frac{1}{2} \sum\limits_n \parallel y(x_n,w) - x_n \parallel^2$。也就是使得输入和输出尽可能的接近。
如果 hidden unit 采用 linear activation function，那么$E(w)$有全局唯一的最优解。这个解相应于将这 N 个 D 维的数据投影到一个 M 维的子空间。在 neural network 上，每个 hidden unit的 D 条入边的权重构成一个 D 维向量，为 principle component。 不过这些向量不一定是正交和归一化的。

如果 hidden unit 采用 nonlinear activation function，最小化目标函数的解仍旧是对应于将数据映射到 M 维 principle-component space。要进行 nonlinear principle component analysis， 只能增加 hidden layer。