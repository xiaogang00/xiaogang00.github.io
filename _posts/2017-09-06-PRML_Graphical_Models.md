---
layout: post
title: PRML Chapter8 Graphical Models
category: PRML
tags: machine_learning
---

* content
{:toc}




### Probabilistic Graphical Model(PGM)的有关概念

一个 joint distribution 的 PGM 是指一个图，图中的每个顶点对应为该分布中的一个随机变量，而图中的边的含义是：

1. 如果是有向图（ Bayesian network），则对任意一个点 $x_i$，设其边的直接前驱节点的集合是$pa_i$，那么对于该 joint distribution， 当给定$pa_i$后，$x_i$与任意非其 descendant 的点条件独立。
2. 如果是无向图（ Markov network），则对任意一条边 $(x_i,x_j)$，那么对于该joint distribution， 当给定除这两个点之外的节点$(x \backslash \{x_i, x_j\})$后，$x_i$和$x_j$不条件独立。

以上定义中， 对一个 graph，用 joint distribution 的性质对其赋予了 semantics。 这使得该graph中的图论性质可以对应解释为 joint distribution中随机变量之间的性质（条件独立性）。因此，可把 graph 理解为对 joint distribution 中独立性的 modeling 或 encoding。



### joint distribution中的独立性与graph中的独立性

上述对 PGM 的定义，要依赖于一个 joint distribution P；graph G中每一条边能不能存在都要看该 joint distribution 是否满足相应的条件独立性质。现在问题是：

Bayesian network 在定义中体现了给定后与其非 descendant 节点的独立性，记全部这些独立性为集合$I_l(G)$；而 Markov network 在定义中体现了对于不存在边直连的点$x_i,x_j$，当除此之外其他点都给定后，二者的条件独立性，记全部这些独立性为集合$I_p(G)$。

假设 $I(G)$表示图 G（有向或无向） 所蕴含的全部条件独立性， $I(P)$表示 joint distribution P所蕴含的全部条件独立性。

我们可以证明有如下的概念：

1. I-map：如果一个 graph 满足$I(G) \subseteq I(P)$，则成这个 graph 是这个 distribution 的 I-map；
2. D-map: 如果一个graph满足$I(G) \supseteq I(P)$，则成这个 graph 是这个 distribution 的 D-map；
3. Perfect map: 如果一个 graph 满足$I(G) = I(P)$，则成这个 graph 是这个 distribution 的Perfect map。并不是所有的分布都有 perfect map。



### PGM的基本意义

G 是 P 的一个 I-map。也就是说，这么定义出来的一个图 G，它所蕴含的全部条件独立性 I(G)是 P 所蕴含全部条件独立性 I(P)的一个子集。

对于 Bayesian network：分布 P 可以 factorizes over G，当且仅当 G 是 P 的 I-map；

对于 Markov network：如果 P 是一个 Gibbs distribution factorizes over G，那么 G 是 P 的 I-map；反之，我们假设 P 是一个 positive distribution，如果 G 是 P 的一个 I-map，那么 P 是一个 Gibbs distribution factorizes over G。

**P factorizes over G的意义**

1. 对于 Bayesian network，是指：给定图 G，分布 P 可以按照 G 进行这样的分解$p(x) = \prod\limits_{k=1}^K p(x_k \mid pa_k)$，其中$pa_k$表示的是节点$x_k$的前驱节点集合，K是全部的random variables的数目；
2. 对于 Markov network，是指：给定图 G，分布 P 可以按照 G 进行这样的分解$p(x) = \frac{1}{Z} \prod\limits_C \varPsi_C(X_C)$，其中$X_C$是图中的一个极大团，$\varPsi_C(X_C) \geq 0$定义在该极大团上面的函数（ potential function），$p(x)$把全部的极大团的 potential function 相乘，然后用 Z 来归一化。



因此，对于图的理解是： Conditional Independence 的角度， 以图表达条件独立性；Factorization 的角度， 以图表达分解性。



### 从joint distribution到graph

如何寻找分布 P 一个 I-map G， 使 I(G)最大。 或者说，怎么构造出这样的一个 I-map 图。 事实上，这样的 I-map 叫做 minimal I-map。

构造算法是：

1. 首先，每个 random variable 表达为图中的一个顶点。
2. 然后，在节点之间加入边。这时候有两套方法，即 Bayesian network 和 Markov network。
   * 对于无向图 Markov network，边是这样建立：任意两个节点$x_i,x_j$ ，如果它们关于$x \backslash \{x_i, x_j\}$条件独立，那么二者之间无边连接；否则连接一条无向边。对于给定的 joint distribution 来说，我们可以反复的询问它上述问题$C_K^2$次（ K 是 random variable 个数），从而构建起该分布的无向图 I-map。
   * 对于有向图的 Bayesian network，边则这样建立：任意节点$x_i$，考虑条件分布 $p(x_i \mid x_1,\dots, x_{i-1})$。假如存在变量$x_j \in \{ x_1,\dots,x_{i-1} \}$，在此 joint distribution 中满足条件独立$p(x_i \mid x_1,\dots,x_{i-1}) = p(x_i \mid \{ x_1,\dots,x_{i-1} \} \backslash x_j )$，那么把$x_j$从$ \{ x_1,\dots,x_{i-1} \}$中删除。此过程持续，直到不存在这样的变量，得到一个子集$pa_i \subset\{ x_1,\dots,x_{i-1} \}$，那么就以每个$pa_i$中的顶点为起点，以$x_i$ 为终点连有向边。对于给定的 joint distribution 来说，我们可以逐个节点的考虑，从而构建起该分布的有向图 I-map。（注意：考虑节点$x_i$时，我们给予它的 condition 是$ \{ x_1,\dots,x_{i-1} \}$，这样做确保了最后的 graph 一定是一个 DAG 图，从而符合 Bayesian network的定义。）



由于 graph 可以看作是一个 conditional independences 的集合，那么，如果两个 graph 其所蕴含的全部 conditional independences 一样，那么就认为这两个图是等价的，称为I-equivalent。 这是一个等价关系，全部 graph 可在此关系下划分成等价类。 两个结构很不一样的 graph，在 I-equivalent 关系下都可能是等价的。

在有向图DAG中有如下的定义：

* Skeleton：一个有向图的 skeleton 是指把图中的有向箭头去掉后剩下的无向图基图。
* Immorality：有向图中的一个 head-to-head 结构（即 XZY 结构）是一个 immorality，如果 X 和 Y 之间没有直接相连的边。

**定理**：假如 G1 和 G2 是同一组随机变量上的 graph，二者 I-equivalent 当且仅当 G1 与 G2的 skeleton 相同，而且具有相同的 immorality。



### 独立性总结

1. 对于一个 Bayesian network G：Local independence 指的是：任意一个$x_k$，当给定$pa_k$后， $x_k$与除它的 descendants 之外的任何节点独立。记全部 Local independence 为 $I_l(G)$；

   Global independence 指的是：根据 D-separation 导出的条件独立的集合。记全部 Global independence 为$I(G)$；

   三者满足的关系是$I_l(G) \subseteq I(G) \subseteq I(P)$，P 是 joint distribution。另外，$I(G)$是完备的，也就是说，任意 G 可反映的 conditional independence，都一定已经包含在$I(G)$中。

2. 对于一个 Markov network G：Local independence有两种。第一种是 pairwise independence，即对任意两个节点$x_i,x_j$ ，当给定除它们外的其他点后，二者独立；这种独立性记为$I_p(G)$。第二种是 Markov blanket independence，即对任意一个节点$x_k$，当给定它的 Markov blanket后，该节点与其他非 Markov blanket 上的节点独立；这种独立性记为$I(G)$。 

   Global independence 指的是：对于两个 G 中节点的子集 X 和 Y，如果对于给定的另一个节点子集 Z，使得 X 和 Y 中任意两个节点之间的任意路径都必经过 Z， 那么 X 和 Y 关于 Z 条件独立。记这种独立性为I(G)。三者的关系是$I_p(G) \subseteq I_l(G) \subseteq I(G)$。

   ​

   ​

### PGM的思想过程与实际过程

PGM 思想过程：Joint distribution 是困难的（ random variable 太多，之间的关系太复杂，导致分布的维度太高，难以获得 direct representation）。但是 random variables 之间存在的独立性为简化 joint distribution 提供了可能。把 joint distribution 所蕴含的独立性找出来，表达到一个 graph 中。然后 work on the graph，基于 graph 研究 inference 的算法（计算 marginal 或 conditional）。

