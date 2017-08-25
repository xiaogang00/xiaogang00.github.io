---
layout: post
title: PRML Chapter3 Linear Models for Regression
category: PRML
tags: machine_learning
---

* content
{:toc}






### Curve fitting(曲线拟合)

一般在这里的问题就是在给定N个样本数据点，以及其对应的函数值的情况下，通过回归的方法求出该函数。而在整个过程中需要遵循平方误差最小的原则(minimize the sum of squares of error)：

假设函数的形式是 $y(x,w)$，其中 w 是该函数的待估计参数， x 则是 input variable；该方法认为，该函数应该使得平方误差之和最小化：


$$
E(w) = \frac{1}{2} \sum\limits_{n=1}^N \{ y(x_n, w)-t_n \}^2
$$


实际的估计方法一般有两种，最大似然估计和最大后验估计：

1. MLE(最大似然估计)：假设每个观察到的样本数据点的函数值的 t 是一个以真实的函数 y(x, w)均值的高斯
   分布产生的。然后求使似然最大的参数值 w，即完成参数估计。
2. MAP(最大后验估计)：与 MLE 类似，不过是求解使后验概率最大的参数值 w。

在以上问题中：MLE 等价于 minimize the sum of squares of error；MAP 等价于 minimize the sum of squares of error with regularizer (MAP with Gaussian prior)



### 线性模型(linear model)

线性模型指的是：函数关于参数，而不是 input variable 是线性，则是线性模型。例如下面的就是linear basis function model：


$$
y(x,w) = w_0 + \sum\limits_{j=1}^{M-1} w_j \phi_j(x) = \mathbf{w}^T \mathbf{\phi}(x)
$$


这里面，basis function 可以任意的选择，而函数 y(x, w)关于 w 始终是线性的。Basis function的选择有：Polynomial, Gaussian, Logistic sigmoid function, Fourier basis, wavelets等。



### 损失函数(loss function)

假设对于输入数据x，估计函数值是 y(x)， 而实际的函数值是 t，这种可能的不准确导致的 loss function 是 $L(t, y(x))$。 那么 Expected loss 是：


$$
E(L) = \iint L(t,y(x)) p(x,t) dxdt
$$


对于回归问题，通常采用的是 squared loss：$L(t,y(x)) = [y(x) -t]^2$

把这个L带入E[L]中后，为了最小化 expected loss E[L]，对函数 y(x)求变分，可以得到的使E[L]最小化的 y(x)是：

$y(x) = E[t\mid x]$

对 squared loss 的 L 函数变形：


$$
E[L] = \int\{ y(x)-E[t\mid x] \}^2 p(x)dx + \iint\{E[t \mid x]-t \}^2 p(x,t)dxdt
$$


如果令$h(x) = E[t\mid x]$，那么我们可以看到E[L]中的第二项$\iint \{ h(x)-t \}^2p(x,t)dxdt$是一个与我们自己的预测y(x)无关的量。我们的任务是能够找到y(x)，使得其中的第一项尽可能小。



### Bias-Variance trade-off

我们基于给定的 data set D 对参数 w 进行 point estimate，但是不同的 data set D，估计出来的参数 w 可能不同，因而导致 y(x)不同，故可以把 D 对y(x)的这种影响记作 $y(x; D)$。换句话说，每个 D 对应了一个其所训练出来的模型 $y(x; D)$.

我们假设有很多不同的 data sets，每个都是从 p(t, x)中采用出来，并且每个 data set 含 N 个样本。如果考虑某个data set D所训练的模型$y(x;D)$，那么：


$$
\{ y(x;D-h(x)\}^2 =  \{ y(x;D)-E_D[y(x;D)]\}^2 + \{ E_D[y(x;D)] -h(x)\}^2 + Loss\\
$$


在这里的Loss项是$Loss = 2\{ y(x;D)-E_D[y(x;D)] \} \{ E_D[y(x;D)] -h(x) \}$；并且我们可以得到：


$$
E_D[\{ y(x;D)-h(x) \}^2] = \{ E_D[y(x;D)]-h(x) \}^2 + E_D[\{ y(x;D)-E_D[y(x;D)] \}^2]
$$


上面： 第一项为 $bias^2$，表示的是 average prediction over all data sets 与 desired regression function（也就是最佳预测 h(x)）之间的差距， 也就是说 average model 与 best model 的接近程度；第二项为 variance，表示的是在单个 data set 上的预测模型 $y(x; D)$在整体平均值 $E[y(x;D)]$附近波动的程度（这本身其实就是一个方差），也就是说模型 $y(x; D)$对数据集 D 的选择的敏感度。



对于回归问题，目标是找到函数 y(x)，使得期望损失 E[L]最小化；而 E[L]本身可以分解开来，由 bias, variance 和 noise 构成。 噪音noise 是一个和 y(x)无关的量，也就是说与回归做的好不好无关，是一个不可避免的量。而 bias 与 variance 则是一对矛盾量：flexible model 可以具有 low bias（ average model 能够很接近 best model）， high variance（但是 single model 对 data set D 的选取很敏感）， rigid model 则有 high bias， low variance。 好的模型应该在 bias 与 variance 之间达到一个折衷。



### Bayesian的model complexity理论

假设我们有多个模型，他们分别是$\{M_i:i=1,\dots,L \}$，观察到的 data set 是D。 Bayesian 的模型比较方法是，比较各个模型的后验概率：$P(M_i\mid D) \propto p(M_i)p(D\mid M_i)$，而在这里我们一般假设模型的先验概率都是一样的，那么比较的关键就是$p(D \mid M_i)$，被称为model evidence 或 marginal likelihood。

一个替代的方法是不进行模型选择，而是通过将多个模型进行平均。以上分析可以看出，各个 model 的后验概率是关键，而计算后验概率的关键又是 model evidence。

在这里$M_i​$相当于超参数，而w是模型参数，模型之间的不同是由超参数引起的。