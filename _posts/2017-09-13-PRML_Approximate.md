---
layout: post
title: PRML Chapter10 Approximate Inference
category: PRML
tags: machine_learning
---

* content
{:toc}






### Approximation

Probabilistic model 中的一个 central task： 给定一组 observation$X$，计算 latent variables$Z$的后验概率$P(Z\mid X)$，以及一些 expectation with respect to $P(Z\mid X)$。在很多情况下$P(Z\mid X)$是analytically intractable 的。 这就需要有 approximation 方法。

Latent variable: 我们只要没有观察到的都可以归为Latent variable。例如在Bayesian 中的 parameter
（它们是 random variable）。 在 Probablistic Graphical Model 的观点看， parameter 和狭义的latent variable 的不同就是， parameter 的个数和观察到的数据的个数无关，但是狭义的 latent variable 则与其相关。Approximation 方法： 分为 确定性（deterministic ）方法和 随机性（stochastic） 方法。前者包括 Laplace approximation， variational inference 等；后者包括 MCMC sampling 等。



### Variational inference

这里的问题是：一个probablistic model $P(X,Z)$，含有observed variables$X = \{ x_1,\dots,x_N \}$，和latent variable $Z = \{ z_1, \dots, z_N \}$

我们的目的是为后验概率$P(Z\mid X)$和model evidence $P(X)$找到 approximation。

思路：

我们可以引入一个分布$q(Z)$，从而把$P(X)$分解开来：$ln[p(X)] = L(q) + KL(q\parallel p)$，其中$L(q) = \int q(Z) ln\{ \frac{p(X,Z)}{q(Z)}dZ \}$，$KL(q\parallel p) = -\int q(Z) ln\{ \frac{p(Z\mid X)}{q(Z)} \}dZ$

我们现在要用$q(Z)$来近似$P(Z \mid X)$，主要是找到一个$q(Z)$，使得$KL(q \parallel p)$最小化。但是$P(X\mid Z)$本身就是 intractable 的，所以直接难以找到使$KL(q \parallel p)$最小化的$q(Z)$，但是我们可以用更加容易处理的$P(X,Z)$，由于$ln[p(X)]$的值与$q(Z)$的选取无关，所以最小化$KL(q \parallel p)$，等价于最大化$L(q)$。

我们假设$q(Z) = \prod \limits_{i=1}^M q_i(Z_i)$，也就是假设满足分解性质，$q(Z)$被称为 variational distribution。

**推导**

我们把$q(Z)$的分解性质代入：


$$
\begin{align} L(q) &= \int \prod\limits_i q_i \{  ln[p(X,Z)]-\sum\limits_i ln [q_i] \} dZ \\ &= \int q_j  ln[\tilde{p}(X, Z_j) ]dZ_j - \int q_j ln [q_j] dZ_j + const \\ &= -KL(q_j \parallel ln \tilde{p} (X,Z_j)) + const\end{align}
$$


其中$ln[\tilde{p}(X,Z_j)] = E_{i \neq j} [ln\{p(X,Z)\}]+const$，而$E_{i\neq j}$表示的是在分布$\prod \limits_{i\neq j} q_i (Z_i)$下求期望。根据KL的性质，我们有$ln[q^{\ast}(Z_j)] = E_{i\neq j} [p(X,Z)] + const$。

所以 variational inference 求解使 $L(q)$最大化的 $q(Z)$的方法是:

1. 初始化每个$q_i(Z_i)$；
2. 然后，逐个考虑$q_i(Z_i)$，在其他不变的条件下，求出$q_i^{\ast}(Z_i)$；
3. 进行这个过程直到收敛。



### Bayesian Gaussian Mixture Model

这种方法也被称为是Bayesian GMM:

首先GMM是$p(X \mid Z, \mu, \Lambda) =\prod\limits_{n=1}^K \prod\limits_{k=1}^K N(x_n \mid \mu_k, \Lambda_k ^{-1}) ^{z_{nk}}$

在贝叶斯中，每个参数都是 random variable，需要为它们引入 prior（都是 conjugate prior）：


$$
\begin{align} p(Z\mid \pi) &= \prod\limits_{n=1}^N \prod\limits_{k=1}^K \pi_k^{z_{nk}}  \\ p(pi) &= Dir(\pi \mid \alpha_0) \\ p(\mu,\Lambda) &= p(\mu\mid \Lambda) p(\Lambda) = \prod_k N(\mu_k \mid m_0, (\beta_0 \Lambda_k)^{-1} )W(\Lambda_k \mid W_0, \nu_0)\end{align}
$$


每个 component的参数的 conjugate prior 都是 Gaussian-Wishart 分布。最后，根据这些 random variable 之间的关系，可以得到 PGM 和相应的 joint distribution 分解： $p(X,Z,\pi, \mu,\Lambda) = p(X\mid Z, \mu, \Lambda) p(Z\mid \pi)p(\pi) p(\mu \mid \Lambda) p(\Lambda)$

**近似后验分布$p(Z,\pi, \mu,\Lambda \mid X)$**

考虑 variational distribution $q(Z,\pi,\mu,\Lambda)$，并且假设其分解性质:$q(Z, \pi, \mu,\ Lambda) = q(Z) q(\pi, \mu,\Lambda)$，也就是在 parameter random variable 和狭义 latent variable 之间的分解性。接下来的就是标准的 variational inference 过程了。首先，针对 $q(Z)$优化 $L(q)$，得到：$ln[q^{\ast}(Z)] = E_{\pi}[p(Z\mid \pi)] + E_{\mu, \Lambda}[p(X\mid Z, \mu, \Lambda)] + const$

进一步解出来就是：



$$
\begin{align} ln[q^{\ast}(Z)] &= \sum\limits_{n=1}^N \sum\limits_{k=1}^K z_{nk} ln\rho_{nk} + const \\ ln \rho_{n} &= E_k[ln\pi_k] + \frac{1}{2} E[ln\mid \Lambda_k \mid]  - \frac{D}{2} ln(2\pi) - \frac{1}{2} E_{\mu_k, \Lambda_k}[(x_n - \mu_k)^T \Lambda_k (x_n - \mu_k)]\end{align}
$$


然后我们针对$q(\pi, \mu,\Lambda)$来优化$L(q)$，得到：



$$
ln[q^{\ast} (\pi,\mu,\Lambda)] = ln[p(\pi)] + \sum\limits_{k=1}^K lnp(\mu_k, \Lambda_k) + E_Z[p(Z\mid \pi)] + \sum_n \sum_k E[z_n] N_k( x_n \mid \mu_k , \Lambda_k ^{-1}) + const
$$

由此可见，计算$q^{\ast}(\pi , \mu, \Lambda)$依赖于Z的期望，另外我们可以看到$ln[q^{\ast}(\pi,\mu,\Lambda)]$是可以分解成关于$\pi$的一部分以及关于$\mu,\Lambda$的一部分的。所 以variational inference 的针对 $q(\pi, \mu,\Lambda)$的最优化$L(q)$ 可以细分为针对$q(\pi)$和K个$q(\mu_k ,\Lambda_k)$。以上两个过程迭代就可以得到最优化的解$q(Z),q(\pi,\mu,\Lambda)$。



**Bayesian GMM 自动选择最优的分支数 K：**
在上面的求解过程中，值得关注各个分支的混合因子 mixing coefficient $\pi$在后验分布下的期望值。 近似的后验分布 $q^{\ast}( \pi)$是一个 Dirchelet 分布。经计算可以发现，$\pi$的 K 个分量中，最后会有部分的期望 $E[\pi_k ]$ 趋于0，从而在 GMM 中不起作用。这样的话，我们可以实现自动的训练出最优的分支数：设置一个较大的 K，然后用上述 variational inference 得到近似后验分布，再把其中$E[\pi_k ]$接近 0 的分支去掉，剩下的即最终的 $K^*$值。

**Predictive distribution:**
对 Bayesian 来说，得到后验概率（的近似）不是最终目的。剩下还要计算对给定一个新的x 的 predictive distribution。

$p(x \mid X) = \sum\limits_z \iiint p(x\mid z,\mu,\Lambda) p(z\mid \pi) p(\pi, \mu, \Lambda \mid X)d\pi d\mu d\Lambda$

把前面得到的近似后验分布 $q^{\ast}(\pi, \mu,\Lambda)$代替上面的$p(\pi, \mu, \Lambda\mid X)$，并且把其他已知的代入，最后得到的$p(x\mid X)$就是 K 个 student t 分布的线性组合。



### Expectation Propagation

在 variational inference 中，近似后验概率的方法是最小化$KL(q(Z) \parallel p(Z\mid X))$ with respect to $q(Z)$，其中$q(Z\mid X)$作为近似的目标是固定的，而第一个参数是可以变化的。在 expectation propagation 中，考虑的是 reversed form 的 KL，即：最小化$KL(p\parallel q)$ with respect to q，而p固定。也就是关于 KL 的第二个参数进行优化。如果限定$q(z)$在 exponential family($q(z) = h(z) g(\eta) exp\{ \eta^T [u(z)] \}$)的范围内变化，那么我们就有$KL(p\parallel q) = -ln[g(\eta)] - \eta^T E_{p(z)}[u(z)] + const$，const 是 与 natuaral parameter $\eta$无关的量 。 为最小化KL，我们对$\eta$求导，并且设置为0，我们可以得到$-\triangledown ln[g(\eta)] = E_{p(z)}[u(z)]$。然而根据 $q(z)$是 exponential family 的分布，所以其本身满足性质：$-\triangledown ln[g(\eta)] = E_{q(z)}[u(z)]$。因此得到$E_{p(z)}[u(z)] = E_{q(z)}[u(z)]$。最终确定$q(z)$的方法就是让$q(z)$的各个成分统计量等于$p(z)$的相应统计量。此过程即 moment matching。



**问题：**我们已知某个 probabilistic model 中，joint distribution 可以这样分解：$p(D,\theta) = \prod\limits_i f_i(\theta)$。其中D 是 observed data， $\theta$是 latent variables（包括参数）。 现在要计算后验分布$p(\theta \mid D)$和model evidence $p(D)$，我们可以看到$p(\theta \mid D) = \frac{1}{p(D)} \prod_i f_i(\theta), p(D) = \int \prod_i f_i(\theta) d\theta$

这个 model 的一个实例是独立同分布的情况。这时候$f_i (\theta) = p(x_i \mid \theta)$。



**基本思路**：用分布$q(\theta) = \frac{1}{Z} \prod_i \tilde{f}_i(\theta)$来近似$p(\theta \mid D)$。每个factor$\tilde{f}_i(\theta)$相对应于model中的$f_i(\theta)$，如果我们假定$\tilde{f}_i(\theta)$是exponential family 的分布。 这样的话，显然 $q(\theta)$也是 exponential family 的分布。为此实现此近似，方法是最小化$KL(p(\theta \mid D) \parallel q(\theta))$ with respect $q(\theta)$。



**详细思路**

对于$q(\theta)$，我们逐个考虑因子$\tilde{f}_j(\theta)$，我们用$f_j(\theta)$替换$\tilde{f}_j(\theta)$，从而得到另一个分布：$\frac{1}{Z_j} f_j(\theta) q^{\setminus j}(\theta)$，其中我们有$q^{\setminus j}(\theta) = \frac{q(\theta)}{\tilde{f}_j (\theta)}$，$Z_j$是归一化因子。我们可以最小化$KL(\frac{1}{Z_j} f_j(\theta) q^{\setminus j}(\theta) \parallel q^{new} (\theta))$ with respect to $q^{new} (\theta)$，这可以通过 moment matching 来得到。得到$q^{new}(\theta)$之后，我们就可以得到$\tilde{f}_j^{new} (\theta) = K \frac{q^{new} (\theta)} {q^{\setminus j}(\theta)}$，$K = Z_j$。以上过程进行多轮，每一轮对每个$f_j(\theta)$ 逐个考虑， 从而不断迭代直至收敛，就得到了最后的$q(\theta)$。

