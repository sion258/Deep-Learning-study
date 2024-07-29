---
date:2024/07/28
tag:deep learning
aurthor:sion
---

# 支持向量回归(SVR)

## 支持向量回归模型的解

支持向量不仅能用作分类问题，在回归问题上也能有不错的表现。

类似线性回归模型，在线性回归模型中我们计算样本到线性超平面的均差来衡量损失从而得到最好的线性超平面。在支持向量回归中，我们使用间隔的定义来计算样本到超平面的“距离”。

![image-20240729130049456](支持向量回归(SVR).assets/image-20240729130049456.png)

设$\epsilon$是样本能容忍的$f(x)$与$y$之间的偏差，仅当样本点在$f(x)+\epsilon$和$f(x)-\epsilon$之间再计算误差。

于是SVR问题可以抽象为：
$$
\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^m\ell_i(f(x_i)-y_i)\\
\text{其中C为正则化常数}，\ell_{\epsilon}是\epsilon 的不敏感损失函数\\
\ell_{\epsilon}(z)=\left\{
\begin{matrix}
0,\quad if|z|<=\epsilon;\\
|z|-\epsilon,\quad otherwise.\\
\end{matrix}
\right.
$$
引入松弛变量$\xi_i$和$\hat\xi_i$(两侧松弛情况完全可能不同)，得到一般式：
$$
\min_{w,b,\xi_i,\hat\xi_i}\frac{1}{2}||w||^2+C\sum_{i=1}^m(\xi_i+\hat\xi_i)\\
s.t. f(x_i)-y_i<=\epsilon+\xi_i,\\
y_i-f(x_i)<=\epsilon+\hat\xi_i,\\
\xi_i>=0,\hat\xi_i>=0,\quad i=1,2,...,m.
$$
再通过拉格朗日乘子法转化为对偶问题:
$$
L(w,b,\alpha,\hat\alpha,\xi,\hat\xi,\mu,\hat\mu)=
\\ \frac{1}{2}||w||^2+C\sum_{i=1}^m(\xi_i+\hat\xi_i) + \sum_{i=1}^m \mu_i\xi_i+\sum_{i=1}^m\hat\mu_i\hat\xi_i\\
+\sum_{i=1}^m\alpha_i(\epsilon+\xi_i-f(x_i)+y_i)+\sum_{i=1}^m\hat\alpha_i(y_i-f(x_i-\epsilon-\hat\xi_i)\ .
$$
求偏导得：
$$
w=\sum_{i=1}^m(\hat\alpha_i-\alpha_i)x_i\ ,
\\0 = \sum_{i=1}^m(\hat\alpha_i-\alpha_i)\ ,
\\C=\alpha_i+\mu_i\ ,
\\C=\hat\alpha_i+\hat\mu_i\ .
$$


代入原拉格朗日乘子式，对于支持向量回归（SVR），其对偶问题为：

$$
\max_{\alpha, \hat{\alpha}} \sum_{i=1}^m y_i (\alpha_i - \hat{\alpha_i}) - \epsilon (\alpha_i + \hat{\alpha_i}) - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m (\alpha_i - \hat{\alpha_i})(\alpha_j - \hat{\alpha_j}) x_i^T x_j
$$

$$
s.t.\quad \sum_{i=1}^m (\alpha_i - \hat{\alpha_i}) = 0
$$

$$
0 \leq \alpha_i \leq C
$$

$$
0 \leq \hat{\alpha_i} \leq C
$$

根据 KKT 条件，上述问题需要满足以下约束：

$$
\begin{cases}
\alpha_i (f(x_i) - y_i - \epsilon - \xi_i) = 0, \\
\hat{\alpha_i} (y_i - f(x_i) - \epsilon - \hat{\xi_i}) = 0, \\
\alpha_i \hat{\alpha_i} = 0, \\
\xi_i \hat{\xi_i} = 0, \\
(C - \alpha_i) \xi_i = 0, \\
(C - \hat{\alpha_i}) \hat{\xi_i} = 0
\end{cases}
$$

- $$\alpha_i (f(x_i) - y_i - \epsilon - \xi_i) = 0$$: 这个条件表示当 $$\alpha_i$$ 非零时，约束 $$f(x_i) - y_i - \epsilon - \xi_i = 0$$ 必须严格成立。
- $$\hat{\alpha_i} (y_i - f(x_i) - \epsilon - \hat{\xi_i}) = 0$$: 这个条件表示当 $$\hat{\alpha_i}$$ 非零时，约束 $$y_i - f(x_i) - \epsilon - \hat{\xi_i} = 0$$ 必须严格成立。
- $$\alpha_i \hat{\alpha_i} = 0$$: 这个条件表示 $$\alpha_i$$ 和 $$\hat{\alpha_i}$$ 不能同时为非零，即它们互斥。
- $$\xi_i \hat{\xi_i} = 0$$: 这个条件表示 $$\xi_i$$ 和 $$\hat{\xi_i}$$ 不能同时为非零，即它们互斥。
- $$(C - \alpha_i) \xi_i = 0$$: 这个条件表示当 $$\xi_i$$ 非零时，$$\alpha_i$$ 必须为 $$C$$。
- $$(C - \hat{\alpha_i}) \hat{\xi_i} = 0$$: 这个条件表示当 $$\hat{\xi_i}$$ 非零时，$$\hat{\alpha_i}$$ 必须为 $$C$$。

最后得到$\text{SVR}$解：
$$
f(x)=\sum_{i=1}^m(\hat\alpha_i-\alpha_i)x_i^Tx+b
$$
若考虑非线性样本空间分步，则可以有映射$\phi(x_i)=w^Tx_i+b$，$L(·)$对$w$求偏导为
$$
w=\sum_{i=1}^m(\hat\alpha_i-\alpha_i)\phi(x_i)\ .
$$
最终解为：
$$
f(x)=\sum_{i=1}^m(\hat\alpha_i-\alpha_i)\kappa(x,x_i)+b
$$
其中$\kappa(x_i,x_j)=\phi(x_i)^T \phi(x_j)$

## 核方法与核线性扩展

令 $\mathcal{H}$ 为核函数 $k$ 对应的再生核希尔伯特空间 (RKHS)， $\|h\|_{\mathcal{H}}$ 表示 $\mathcal{H}$ 空间中函数 $h$ 的范数。对于任意单调递增函数 $\phi: [0, \infty) \rightarrow [0, \infty)$ 和任意非负损失函数 $L: \mathbb{R}^m \rightarrow [0, \infty)$，优化问题

$$ \min_{h \in \mathcal{H}} \phi(\|h\|_{\mathcal{H}}) + L(h(x_1), h(x_2), \ldots, h(x_m)) \tag{6.57} $$

的解总可写为

$$ h^*(x) = \sum_{i=1}^{m} \alpha_i k(x, x_i). \tag{6.58} $$

表示定理对损失函数没有限制，对正则化项仅要求单调递增，甚至不要求是凸函数。这意味着对于一般的损失函数和正则化项，优化问题的最优解 $h^*(x)$ 都可表示为核函数 $k(x, x_i)$ 的线性组合。这显示出核函数的巨大威力。

人们发展出一系列基于核函数的学习方法，统称为“核方法”（kernel methods）。最常见的，是通过“核化”（即引入核函数）来将线性学习器拓展为非线性学习器。下面我们以线性判别分析为例，来演示如何通过核化对其进行非线性拓展，从而得出“**核线性判别分析”（Kernelized Linear Discriminant Analysis，简称 KLDA）**。

我们先假设可通过某种映射函数 $\phi: X \rightarrow \mathcal{F}$ 将样本映射到一个特征空间 $\mathcal{F}$，然后在 $\mathcal{F}$ 中执行线性判别分析，以求得

$$ h(x) = \omega^T \phi(x). \tag{6.59} $$

类似于式LDA，KLDA 的学习目标是

$$ \max_{\omega} J(\omega) = \frac{\omega^T S_b \omega}{\omega^T S_w \omega}, \tag{6.60} $$

其中 $S_b$ 和 $S_w$ 分别为训练样本在特征空间 $\mathcal{F}$ 中的类间散度矩阵和类内散度矩阵。令 $X_i$ 表示第 $i$ 类样本的集合，其样本数为 $m_i$；总样本数为 $m = m_0 + m_1$，类样本在特征空间 $\mathcal{F}$ 中的均值为

$$ \mu_i = \frac{1}{m_i} \sum_{x \in X_i} \phi(x), $$

两个散度矩阵分别为

$$ S_b = \sum_{i=0}^{1} m_i (\mu_i - \mu)(\mu_i - \mu)^T, $$

$$ S_w = \sum_{i=0}^{1} \sum_{x \in X_i} (\phi(x) - \mu_i)(\phi(x) - \mu_i)^T. $$

**通常我们难以知道映射 $\phi$ 的具体形式，因此使用核函数 $k(x, y) = \langle \phi(x), \phi(y) \rangle$ 来隐式地表达这个映射和特征空间** $\mathcal{F}$。把 $J(\omega)$ 作为优化问题 的损失函数，再令 $h(x) = \sum_{i=1}^{m} \alpha_i k(x, x_i)$，由表示定理，函数 $h(x)$ 可写为

$$ h(x) = \sum_{i=1}^{m} \alpha_i k(x, x_i), \tag{6.64} $$

于是由式$h(x) = \omega^T \phi(x).$可得

$$ \omega = \sum_{i=1}^{m} \alpha_i \phi(x_i). \tag{6.65} $$

令 $K \in \mathbb{R}^{m \times m}$ 为核函数 $k$ 所对应的核矩阵，其中 $K_{ij} = k(x_i, x_j)$。$1_i \in \{0, 1\}^{m \times 1}$ 为第 $i$ 类样本的指示向量，即 $1_i$ 的第 $j$ 个分量为 1 当且仅当第 $j$ 个样本属于第 $i$ 类，否则为 0。再令

$$ \hat\mu_0=\frac{1}{m_0}K1_{0},$$

$$ \hat\mu_1=\frac{1}{m_1}K1_{0},$$

$$ M = (\mu_0 - \mu_1)(\mu_0 - \mu_1)^T, \tag{6.67} $$

$$ N = K^T K - \sum_{i=0}^{1} \mu_i \mu_i^T. \tag{6.68} $$

于是，式 $ \max_{\omega} J(\omega) = \frac{\omega^T S_b \omega}{\omega^T S_w \omega}$等价为(推导参考阅读资料)

$$ J(\alpha) = \frac{\alpha^T M \alpha}{\alpha^T N \alpha}. \tag{6.69} $$

显然，使用线性判别分析求解方法即可得到 $\alpha^*$(**参考阅读资料的核对数几率回归**)

进而可由式  $h(x) = \sum_{i=1}^{m} \alpha_i k(x, x_i), \tag{6.64}$ 得到投影函数 $h(x)$。

## 阅读材料(核方法和KLDA)

![image-20240729141857977](支持向量回归(SVR).assets/image-20240729141857977.png)

![image-20240729141907301](支持向量回归(SVR).assets/image-20240729141907301.png)

![image-20240729141917060](支持向量回归(SVR).assets/image-20240729141917060.png)

![image-20240729141927545](支持向量回归(SVR).assets/image-20240729141927545.png)

![image-20240729141935362](支持向量回归(SVR).assets/image-20240729141935362.png)
