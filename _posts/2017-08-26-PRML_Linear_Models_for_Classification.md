---
layout: post
title: PRML Chapter4 Linear Models for Classification
category: PRML
tags: machine_learning
---

* content
{:toc}








### 超平面

超平面(hyperplane)：在一个D维Euclidean space中的（超）平面是一它的一个 D-1维流形，而且该空间是一个线性空间。如果分布于 D 维空间中的全部数据点可以用超平面无错地分隔成多个类，那么这种情况被称为是线性可分的。

Coding scheme： 一般我们指的1-of-K binary coding scheme，即如果有 K 个类，某数据点属于第 i 个类，则表示为一个 K 维向量，该向量除了第 i 个分量是 1，其余都是 0。



### 分类方法

在这之前还需要先说明下Generalized Linear Model的概念。一般来说，这种模型可以是一个激活函数作用于一个线性模型的特征变量上：


$$
y(x) = f(w^T \phi(x) + w_0)
$$


这也就被称为是Generalized Linear Model (GLM)。一般来说：

1. 当 f 是 nonlinear function 的时候， GLM 是一个 classification model；
2. 当 f 是 identity function 的时候， GLM 是一个 regression model。

我们需要注意的是用于分类的GLM和线性回归模型是不同的，在GLM里面，由于激活函数不是线性的，所以y也不再是输入参数w的线性函数。

在这里介绍了三种分类的方法：

1. 生成模型(Generative model)：在这里我们对类条件概率密度$p(x\mid C_k)$或者$p(x,C_k)$和类先验概率分布$p(C_k)$建模，然后使⽤这两个概率密度通过贝叶斯定理计算后验概率密度$p(C_k \mid x)$；
2. 判别函数(Discriminant function)：通过判别函数的，将输入的x在判别规则下，判别为某个类；
3. 判别模型(Discriminant model)：一般直接计算$p(C_k \mid x)$，对于GLM而言，后验概率$p(C_K \mid x)$ 将表达为$p(C_k \mid x) = f(w^T \phi(x) + w_0)$，我们可以利用训练数据来直接估计(infe)出GLM里面的参数w，根据这个infer阶段不同阶段采用的不同方法，可以有类似Logistic Regression 和 Probit regression 的分类器。



### 线性判别函数

线性判别是用超平面作为决策面对输入数据点 x 判定其类别。 另外根据这种分类的思想，我们可以从二分类扩展到多分类。一般来说，有两种方法来实现多分类的要求：

1. 一对多(One-versus-the-rest)：用 K 个 two-class discriminant 来完成 K 类的分类，每个 discriminant
   把一个类的数据与非该类的数据分开；
2. 一对一(One-versus-one)：用 $\frac{K(K-1)}{2}$ 个 binary discriminant 来完成 K 类的分类。

但是两种方法都存在不足，有可能造成一个region里面的被分配到多个类中。这种问题被称为是ambiguous region 问题。

特别地，我们可以用K个线性函数来组成K类的判别器，每个线性函数的形式可以写成：$y_k(x) = w_k^T x + w_{k0}$，类别m和n之间的决策超平面为：


$$
(w_m - w_n)^T x + (w_{m0} - w_{n0}) = 0
$$


可以证明这种 K-class discriminant 形成的决策空间(decision region) 是连通凸空间，所以不存在 ambiguous region 问题。如何来获取以上判别器中的参数，可以使用和回归方法中类似的方法，比如通过最小平方误差的方法，典型的包括我们熟悉的感知器算法。下面还有一个比较经典和重要的判别方法：Fisher’s linear discriminant

#### Fisher’s linear discriminant

注意到输入数据点 x 是一个D维向量，但是 $y=w^Tx$ 确是一个只有一维的 scalar，这个过程可以看作是D维的一个向量投影到1维空间上。 Fisher 判别的思想是，把分类看作选择一个1维空间，并把原 D 维数据投影到该空间的过程；选择1维空间的准则是 Fisher criterion，包含两方面的要求：一方面要求投影到1维空间后，不同类的数据是分开的；而另一方面要求同一类的数据能够尽可能聚集在一块。Fisher criterion 是这两方面要求的量化。求解Fisher criterion 最大化后，得到参数 w，即确定了决策超平面的（法）方向；剩下只需要再在1维上确定一个阈值 y0，表明该超平面的位置即可。



### 生成模型(Generative model)

如果我们需要求解二分类问题，也就是求解类别的后验概率$p(C_k \mid x)$：

1. 对后验概率进行变形：


$$
p(C_1 \mid x) = \frac{1}{1+exp(-a)} , a = ln(\frac{p(x\mid C_1)p(C_1)}{p(x\mid C_2)p(C_2)})
$$


2. 需要假设类条件分布的形式，即$p(x\mid C_k) = N(\mu_k, \Sigma)$
3. 在获得类条件分布和先验概率的情况下，可以求出后验概率:


$$
\begin{align} p(C_1\mid x) &= \sigma(w^T x + w_0) \\w &=\Sigma^{-1}(\mu_1 - \mu_2) \\ w_0 &= -\frac{1}{2}\mu_1^T \Sigma^{-1}\mu_1 + \frac{1}{2}\mu_2^T \Sigma^{-1}\mu_2 + ln\frac{p(C_1)}{p(C_2)}   \end{align}
$$

4. MLE参数估计

   我们假设$p(C_1) = \pi, p(C_2) =1- \pi$，训练集为$\{x_n,t_n \}$，如果$x_n \in C_1$，则$t_n=1$，否则$t_n=0$。

   我们可以得到似然函数，并且通过对各个参数求导就得出了 MLE 的参数估计值。这些值带入后验概率，就完成了求解。：

$$
p(t \mid \pi , \mu_1,\mu_2, \Sigma) = \prod \limits_{n=1}^N [\pi N(x_n \mid \mu_1,\Sigma)]^{t_n} [(1-\pi) N(x_n \mid \mu_2,\Sigma)]^{1-t_n}
$$

在这里总结下如何进行估计的方法：

1. MLE 估计该分布中的参数（从而得到了 class-conditional distribution）；
2. 计算每个类别的后验概率。



### 判别模型(Discriminant model)

判别模型具有优点：参数相比较而言少，是维数 D 的线性函数；相比之下 生成模型的参数是 D 的平方级。

#### 逻辑斯蒂回归(Logistic regression)

其数学的表达式是：


$$
p(C_k \mid \phi) = \frac{exp(a_k)} {\sum\limits_j exp(a_j)}，a_k = w_k^T \phi
$$


当只有 2 个类时，上面的公式退化成 logistic function。以 2 个类的情况为例，现在的目标就变成为，给定训练数据$\{ \phi_n,t_n \}$，其中$\phi_n$是一个 D 维的 feature vector，而$t_n \in \{0,1\}$是类属标记。 此时的似然函数是：


$$
p(t\mid w) = \prod \limits_{n=1}^N y_n ^{t_n}(1-y_n) ^{1-t_n}
$$


求出最小化 negative logarithm of the likelihood（也就是交叉熵损失函数 cross-entropy error function）的参数w，从而完成对后验概率$p(C_k\mid \phi)$的inference：


$$
E(w) = -\sum\limits_{n=1}^N [t_N ln(y_n) + (1-t_n) ln(1-y_n)]
$$


具体如何最小化这个损失函数，可以使用类似牛顿迭代的方法，其迭代公式为$w^{new} = w^{old} -H^{-1} \triangledown E(w)$，其中H为Hessian矩阵。



### 拉普拉斯近似

在这里就是寻找一个高斯分布$q(z)$，来近似一个复杂的概率分布$p(z) = \frac{1}{Z} f(z)$，其中Z是归一化因子。

一般首先寻找$f(z)$的一个驻点$z_0$，然后在这里将$ln[f(z)]$进行泰勒展开：



$$
ln[f(z)] \simeq ln[f(z_0)] -\frac{1}{2} A(z-z_0)^2,A = -\frac{d^2}{dz^2}ln[f(z)]\mid_{ z=z_0}
$$


由此我们可以得到高斯分布的近似:



$$
q(z) \simeq (\frac{A}{2\pi})^{1/2} exp\{ -\frac{A}{2}(z-z_0)^2 \}
$$



这就是用来近似原复杂分布$p(z)$的高斯分布。对于高维情况，用相应的高维泰勒展开和高维高斯分布即可。





