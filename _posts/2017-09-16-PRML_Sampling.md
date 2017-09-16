---
layout: post
title: PRML Chapter11 Sampling Method
category: PRML
tags: machine_learning
---

* content
{:toc}






### 基于proposal distribution的sampling

我们在这里假设从分布$p(z)$采样是困难的，但是可以找到一个相对容易采用的分布$q(z)$，也就是proposal distribution。

#### Rejection sampling

我们以sing-variable的情况进行考虑，假设分布$p(z) = \frac{1}{Z_p} \tilde{p}(z)$，其中$Z_p$是$p(z)$中与z无关的一个因子。之所以分开来写是因为有的时候$p(z)$中与z无关的部分困难比较复杂，难以计算。例如有$Gam(z\mid a,b) = \frac{b^a z^{a-1} exp(-bz)}{\Gamma (a)}$，其中我们可以看到$\frac{b^a}{\Gamma(a)}$与z无关，可以作为$Z_p$提出来。更加重要的是，有时候我们对于分布$p(z)$的认识集中在$p(z)$的与z相关的部分。其 normalization的$Z_p$还处于一个难以计算的未知状态。 因此，在 sampling 中，如果可以发明只利用与 z 有关部分即可进行采样的算法，就避开了对复杂的$Z_p$的计算。

Rejection sampling 要为 $p(z)$找一个 proposal distribution $q(z)$，而且 $q(z)$必须是很容易进行sampling 的。然后找到一个尽可能小的常数k，使得$kq(z) \geq \tilde{p}(z)$。

Rejection sampling 的过程是： 首先从 $q(z)$中 sample 出一个数$z_0$，然后从均匀分布$[0, kq(z_0)]$中sample出另外一个数$u_0$，这个时候平面上的点$(z_0, u_0)$是$kq(z)$下方的区域中的均匀分布。如果$u_0 \geq \tilde{p}(z_0)$，那么拒绝点$z_0$的采样，并且重复前面的步骤，否则接受$z_0$为符合分布$p(z)$的点。上述过程看出了 k 为什么要尽可能小。 K 越小，才能使$z_0$被拒绝的概率尽可能小，从而提高 rejection sampling 的效率。

Rejection sampling 的缺点是：维数越高，拒绝率越高，采样效率越低。例如高维的球，可计算其测度主要集中在球的表面；而 rejection sampling 中，$u_0 \geq \tilde{p}(z_0)$ 的部分正是高维几何体的表层。这就是导致很高的拒绝率。

#### Importance sampling

我们在这里假设对于$p(z)$采样是困难的，不过对于一个给定的z，却可容易地计算其概率值$p(z)$。

假设现在不是要对分布 $p(z)$采样，而是要计算一个函数 $f(z)$在该分布下的期望。采样现在是作为近似计算该期望的方法，比如有 L个 $p(z)$的采样，那么$E[f] = \frac{1}{L} \sum\limits_{l=1}^L f(z^{(l)})$。但是p(z)本身采样困难，所以我们还是得像 Rejection sampling 那样，找到一个更容易采样的分布$q(z)$， 并且假设从$q(z)$采样了L个样本，那么：


$$
E[f] = \int p(z)f(z)dz  = \int \frac{p(z) f(z)}{q(z)} q(z) dz  \simeq \frac{1}{L} \sum\limits_{l=1}^L \frac{p(z^{(l)})}{q(z^{(l)})} f(z^{(l)})
$$


其中$r_l = \frac{p(z^{(l)})}{q(z^{(l)})}$被称为是importance weights。

再进一步假设： 对分布 p(z)的认识集中在 p(z)的与 z 相关的部分$\tilde{p}(z)$，其 normalization constants $Z_p$是未知的，那么同时也从$q(z)$中分离出一个常数$Z_p$，那么


$$
E[f] = \int p(z) f(z) dz  = \frac{Z_q}{Z_p} \int \frac{\tilde{p}(z) f(z)} {\tilde{q}(z)} q(z) dz \simeq \frac{Z_q}{Z_p} \frac{1}{L} \sum\limits_{l=1}^L \tilde{r}_l f(z^{(l)})
$$


其中$\tilde{r}_l = \frac{\tilde{p}(z^{(l)})}{\tilde{q}(z^{(l)})}​$。同样地，可以计算：$\frac{Z_q}{Z_p} = \frac{1}{Z_p}\int \tilde{p}(x)dx = \int \frac{q(x)}{\tilde{q}(x)} \tilde{p}(x)dx = \frac{1}{L} \sum\limits_{l=1}^L \tilde{r}_l​$

最终我们可以得到：$E(f) = \frac{1}{L} \sum\limits_{l=1}^{L} w_l f(z^{(l)})$。

其中， $z^{(l)}$是分布从 q(z)采样的 L 个样本，而$w_l = \frac{\tilde{r}_l}{\sum\limits_{m=1}^L \tilde{r}_m}$

注意，在$w_l$的计算中，已经只需要分布 $p(z)$的与 z 有关部分$\tilde{p}_z$ 。 从而达到了目的。

#### Sampling-importance-resampling (SIR)

Rejection sampling 要找一个常数 k，Importance sampling 不能得到 p(z)的样本而只有期望，Sampling-importance-resampling 克服了这两个问题。 SIR 同样基于一个易于采用的 proposal distribution q(z)，其做法分两步：

1. 从q(z)中抽样L个样本$z^{(l)}$，并且计算 Importance sampling 中所定义的 $w_l$；
2. 从集合$\{ z^{(l)} : l = 1,\dots,L \}$中有放回地采样L个样本，其中$z^{(l)}$被抽中的概率是$w_l$。可以证明， 这样得出来的 L 个样本近似目标分布 p(z)；当 L 趋于无穷时，将完全服从目标分布。



### Monte Carlo EM algorithm

在 EM 算法中，E 步是计算 complete-log likelihood 在 latent variables 的后验概率下的期望，也就是：$Q(\theta, \theta^{old}) = \sum\limits_Z p(Z\mid X,\theta^{old}) ln[p(X,Z\mid \theta)]$。现在用sample的方法来近似这个求和。从当前对 latent variables 的后验概率的估计分布$p(Z\mid X, \theta^{old})$中采样出L个样本，于是得到$Q(\theta, \theta^{old}) \simeq \frac{1}{L} \sum\limits_l ln[p(Z^{(l)},X\mid \theta)]$。

在 M 步中，还是跟 EM 算法一样，最大化$Q(\theta, \theta^{old})$with respect to $\theta$。



### Metropolis-Hastings算法

与 Rejection sampling 和 Importance sampling 一样，需要为目标分布 p(z)找一个易于采用的proposal sampling q(z)。 该算法的采样过程是：假设当前已经 sampling 出的样本是$z^{(t)}$，那么下一个样本从分布$q(z\mid z^{(t)})$中获得，记新采样出的样本是$z^{\ast}$，我们以概率$A(z^{\ast}, z^{(t)})$接受该样本。也就是说：
$$
z^{(t+1)} =\begin{cases} z^{\ast} & if \quad accept \\z^{(t)} &if \quad reject \end{cases}
$$
其中$A(z^{\ast}, z^{(t)}) = \min \{ 1, \frac{\tilde{p}(z^{\ast})q(z^{(t)}\mid z^{\ast})}{\tilde{p}(z^{(t)})q(z^{\ast} \mid z^{(t)})} \}$，而$\tilde{p}(z)$是$p(z)$只与z相关的部分。

以上算法其实定义了一个一阶 Markov chain$\{ z^{(t)}: t = 1, \dots \}$。需要证明的是，当$t \rightarrow \infty$时，z 的分布趋于目标分布 $p(z)$。可以验证 $p(z)$满足 detailed balance 条件，因此是该 Markov chain 的 invariant distribution。

对于 continuous state space，一般用 Gaussian centred on the current state 作为 proposal distribution。而该分布的 variance 就是一个选择的难点了： variance 太小，则遍历 state space 慢，效率低； variance 太大，则导致拒绝率高，同样也效率低。



### Gibbs sampling

Metropolis-Hastings 算法的一个特例。假设要从分布$p(z) = p(z_1,\dots,z_M)$中采样。 Gibbs sampling 的做法是：

1. 给每个$\{ z_i:i = 1,\dots,M \}$一个初始值，作为第$t=1$次sample的结果；
2. 考虑从第$t$到第$t+1$次的sample：
   * sample $z_1^{t+1} \sim p(z_1\mid z_2^{t}, z_3^{t}, \dots,z_M^{t})$
   * sample $z_2^{t+1} \sim p(z_2\mid z_1^{t+1}, z_3^{t}, \dots,z_M^{t})$
   * $\dots$
   * sample $z_j^{t+1} \sim p(z_j\mid z_1^{t+1},\dots, z_{j-1}^{t+1},z_{j+1}^{t},\dots,z_M^{t})$
   * sample $z_M^{t+1} \sim p(z_M\mid z_1^{t+1}, z_2^{t+1}, \dots,z_{M-1}^{t+1})$



现在从 Metropolis-Hastings 算法的角度认识 Gibbs sampling。 考虑对$z_k$的更新采样，在得到$z_k^{t+1}$后的z记为$z^{\ast}$，那么就有$q(z^{\ast} \mid z) = p(z_k\mid z_{\setminus k})$，其中$z_{\setminus k}$表示集合$\{ z_i:i = 1,\dots,M \}$去掉$z_k$。在z和$z^{\ast}$之间，只有$z_k$是不同的，因此$z_{\setminus k} = z_{\setminus k}^{\ast}$。于是在 Metropolis-Hastings 算法中的接受概率为：$A(z^{\ast}, z^{(t)}) = 1$。也就是说，所有 Metropolis-Hastings 的状态都会接受。这正是 Gibbs sampling 所呈现的。



