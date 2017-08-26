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

但是两种方法都存在不足，有可能造成一个region里面的被分配到多个类中。

