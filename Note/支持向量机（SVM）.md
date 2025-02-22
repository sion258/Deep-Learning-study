---
date:2024/07/27
tag:deep learn
aurthor:sion
---

# 支持向量机（SVM）

[TOC]

在感知机模型中，我们通过找到一个超平面$w^Tx - \theta = 0$来将样本空间划分来完成分类任务，在感知机模型中，主要通过设置$w^T和\theta$的更新公式完成超平面的确定。现在我们考虑最合适的超平面的确认方式--**超平面离样本尽可能远离并且正好处于“正中间”**。这就是SVM想要求解的情形。

## 间隔

在样本空间中，定义任意点$x$到超平面$(w,b)$的距离可写为：
$$
r=\frac{|w^Tx + b|}{||w||}
$$
将训练样本分类好的超平面应当满足：

（1）离样本足够远

（2）正好处于最中间

根据超平面性质，y为1被划分到正空间，y为-1被划分到负空间：
$$
\left\{
\begin{matrix}
w^Tx+b>=0,y=+1\\
w^Tx+b<0,y=-1\\
\end{matrix}
\right.
$$
接下来满足（2）：
$$
\left\{
\begin{matrix}\
\frac{|w^Tx_i+b|}{||w||}>\frac{|w^Tx_*^{+}+b|}{||w||}\\
\frac{|w^Tx_i+b|}{||w||}>\frac{|w^Tx_*^{-}+b|}{||w||}\\
\end{matrix}
\right.
$$
由两式得
$$
\left\{
\begin{matrix}
w^Tx+b>=1,y=+1\\
w^Tx+b<=-1,y=-1\\
\end{matrix}
\right.
$$
![image-20240727160223370](C:\Users\唐浩钏\AppData\Roaming\Typora\typora-user-images\image-20240727160223370.png)

**当且仅当$x_i$为离超平面最近的点时。等号成立，这些样本点都是一个特征向量，这样的向量被称为“支持向量”**。两个不同类的支持向量到超平面的距离之和为：
$$
\gamma = \frac{2}{||w||}
$$
被称为**间隔（margin）**，**满足（1）意味着间隔最大**

到这里我们明确了，想要找到满足条件的超平面，要找到满足$\left\{
\begin{matrix}
w^Tx+b>=1,y=+1\\
w^Tx+b<=-1,y=-1\\
\end{matrix}
\right.$的超平面的参数$w$和$b$使得$\gamma$最大：
$$
max_{w,b}\frac{2}{||w||}
\\ \qquad s.t.\quad y_i(w^Tx_i+b)>=1,\quad i=1,2...,m.
$$
要求解$\frac{2}{||w||}$最大的情形，等价于求$||w||^2$最小的情形，因此上述问题等价于式（2）：
$$
min_{w,b}\frac{||w||^2}{2}
\\ \qquad s.t.\quad y_i(w^Tx_i+b)>=1,\quad i=1,2...,m.
$$

> **ps:这里转化为$||w||^2$而不是$||w||$的原因在于，求$||w||^2$是一个凸优化问题，更容易使用优化算法求解。**
>
> $s.t.\quad y_i(w^Tx_i+b)>=1,\quad i=1,2...,m.$是由$\left\{
> \begin{matrix}
> w^Tx+b>=1,y=+1\\
> w^Tx+b<=-1,y=-1\\
> \end{matrix}
> \right.$得到的

## 对偶问题

对于式（2），我们这里**采用拉格朗日乘子法求得“对偶问题”**，对于约束添加拉格朗日乘子$\alpha_i>=0$,有式（3）：
$$
L(w,b,\alpha) = \frac{1}{2}||w||^2+\sum_{i=1}^m \alpha_i(1-y_i(w^Tx_i+b))
$$
分别对$w$,$b$求偏导：
$$
w=\sum_{i=1}^m\alpha_iy_ix_i\\
0=\sum_{i=1}^m\alpha_iy_i\ .
$$
带入原式得式（2）的对偶问题式（4）：
$$
max_\alpha\sum_{i=1}^m-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
\\ \qquad s.t.\quad \sum_{i=1}^m\alpha_iy_i=0,\\
\alpha_i>=0,\quad i=1,2...,m.
$$

> 为什么求其对偶问题：
>
> （1） 式 （2） 中的未知数是$ w $和$ b$，式 (4) 中的未知数是 $α$，$w$ 的维度 $d$ 对应样本特征个数，$α$ 的 维度 $m$ 对应训练样本个数，通常 $m ≪ d$，所以求解式(4) 更高效，反之求解式 (2) 更高效
>
> （2）式(4)中有样本内积$x_i^Tx_j$这一项，后续可以很自然地引入**核函数**，进而使得支持向量机也能对在原始特征空间线性不可分的数据进行分类(pumpkin book)）	

解出$\alpha$后，得到$w$与$b$，于是可以得到模型(*)：
$$
f(x) = w^Tx+b\\
=\sum_{i=1}^m\alpha_iy_ix_i^Tx+b
$$

## 核函数

与单层感知层模型一样，上面的讨论我们假设了样本空间是线性可分的，因此可以由一个线性超平面来划分样本空间：
$$
f(x) =w^Tx+b
$$
所以当面对“异或”这样的非线性可分任务时就无法运作，一个好的办法是**将原本样本空间的特征映射到一个更高维的空间**里去：

![image-20240727213848901](支持向量机（SVM）.assets/image-20240727213848901.png)

**根据投影定理，任何有限维空间都可以嵌入到更高维的空间中**。这个更高维的空间包含了原始空间的所有特征，同时还可以容纳更多的特征，从而使得数据分析和处理变得更加灵活和准确。

令$\phi(x)$表示将$x$映射后的特征向量，那么划分样本空间的超平面表示为：
$$
f(x)=w^T\phi(x)+b
$$
类似的，求解$w^T$满足凸优化问题：
$$
min_{w,b}\frac{1}{2}||w||^2\\
s.t.\quad y_i(w^T\phi(x_i)+b)>=1,\quad i=1,2,3,...,m.
$$
类似地，也有对偶问题(*):
$$
max_\alpha\sum_{i=1^m}\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_i)\\
s.t. \quad \sum_{i=1}^m\alpha_iy_i = 0,\quad \alpha_i>=0,\quad i= 1,2,..,m.
$$
在(*)中，显式地处理$\phi(x_i)^T\phi(x_i)$显得十分麻烦，因为升维之后的空间可能维度很高，甚至无限维(维数灾难)，所以我们考虑有函数$k(x_i,x_j)$，在升维之后在样本空间的内积能改写为在原始样本空间的$k(x_i,x_j)$：
$$
max_\alpha\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_jk(x_i,x_i)\\
s.t. \quad \sum_{i=1}^m\alpha_iy_i = 0,\quad \alpha_i>=0,\quad i= 1,2,..,m.
$$
求解后可以得到：
$$
f(x) = w^T\phi(x)+b \\= \sum_{i=1}^m \alpha_iy_i\phi(x_i)^T\phi(x)+b 
\\ =\sum_{i=1}^m\alpha_i y_i k(x,x_i)+b\ .
$$
这里的映射$k(·,·)$就是**核函数（kwenel function）**，模型的最优解可以通过训练样本的核函数展开，这一展开式也叫做**"支持向量展式"（support vector expansion）**

核函数$k$隐式地包含在空间里，我们无法直接知道合适的核函数的存在性以及具体形式。满足以下性质的函数被称为核函数。

### 核函数的定义

> 核函数 \($\kappa(x, y)$\) 是一个定义在输入空间 \($\mathcal{X}$\) 上的函数，它满足以下条件：
> 1. **对称性：\($\kappa(x, y)$ = $\kappa(y, x)$\)**
> 2. **对于任意数据集 \($D = \{x_1, x_2, \ldots, x_m\\$)，对应的核矩阵 \( $K$ \) 是半正定的。**
>

$$
K = \begin{bmatrix}
\kappa(x_1, x_1) & \kappa(x_1, x_2) & \cdots & \kappa(x_1, x_m) \\
\kappa(x_2, x_1) & \kappa(x_2, x_2) & \cdots & \kappa(x_2, x_m) \\
\vdots & \vdots & \ddots & \vdots \\
\kappa(x_m, x_1) & \kappa(x_m, x_2) & \cdots & \kappa(x_m, x_m) \\
\end{bmatrix}
$$

核函数隐含地定义了一种将输入数据从原始空间映射到一个高维特征空间的方法。在这个高维特征空间中，原本在低维空间中非线性可分的数据可能变得线性可分。这种特征空间的映射是通过核技巧实现的，而不需要显式地计算映射后的特征。

> 只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用.事实上，对于一个半正定核矩阵，总能找到一个与之对应的映射$\phi$. 换言之，任何一个核函数都隐式地定义了一个称为"再生核希尔伯特空间" (Reproducing Kernel Hilbert Space ，简称 RKHS) 的特征空间.

### 常见核函数

以下是一些常见的核函数及其参数的表格：

| 名称       | 表达式                                                       | 参数                                          |
| ---------- | ------------------------------------------------------------ | --------------------------------------------- |
| 线性核     | $\kappa(x_i, x_j) = x_i^T x_j$\)                             | 无                                            |
| 多项式核   | $\kappa(x_i, x_j) = (x_i^T x_j + c)^d$\)                     | \($d \geq 1$\) 为多项式的次数，\($c$\) 为常数 |
| 高斯核     | $\kappa(x_i, x_j) = \exp \left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right)$\) | \($\sigma > 0$\) 为高斯核的带宽（$width$）    |
| 拉普拉斯核 | $\kappa(x_i, x_j) = \exp \left( -\frac{\|x_i - x_j\|}{\sigma} \right)$\) | \($\sigma > 0$\)                              |
| Sigmoid 核 | $\kappa(x_i, x_j) = \tanh (\beta x_i^T x_j + \theta)$\)      | \($\beta > 0$, $\theta < 0$\)                 |

以下是一些常见核函数组合方式及其公式：

$$
\text{若 } \kappa_1 \text{ 和 } \kappa_2 \text{ 为核函数，则对于任意正数 } \gamma_1, \gamma_2, \text{ 其线性组合}
$$
$$
\gamma_1 \kappa_1 + \gamma_2 \kappa_2 \tag{6.25}
$$
也是核函数；

$$
\text{若 } \kappa_1 \text{ 和 } \kappa_2 \text{ 为核函数，则核函数的直积}
$$
$$
(\kappa_1 \otimes \kappa_2)(x, z) = \kappa_1(x, z) \kappa_2(x, z) \tag{6.26}
$$
也是核函数；

$$
\text{若 } \kappa_1 \text{ 为核函数，则对于任意函数 } g(x),
$$
$$
\kappa(x, z) = g(x) \kappa_1(x, z) g(z) \tag{6.27}
$$
也是核函数。

这些性质表明，通过不同的组合方式，可以构造新的核函数以满足特定应用需求。常见的组合方法包括核函数的加.

## 软间隔与正则化

在实际问题中，尽管样本空间可能满足线性可分，然而总有一些样本可能出现不满足约束的分布：
				![image-20240728110133746](支持向量机（SVM）.assets/image-20240728110133746.png)

所以我们的模型应该使得SVM允许一些样本犯错，这样的约束条件称为软间隔，使得一些样本可以不满足$y_i(w^Tx_i+b)>=1$.

### 正则化

为了描述好这个问题，我们可以将问题$min_{w,b}\frac{1}{2}||w||^2$进行正则化表示：
$$
\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^ml_{0/1} (y_i(w^Tx_i+b)-1)
$$
其中$l_{0/1}$称为“0/1损失函数”：
$$
l_{0/1}(z)=
\left\{
\begin{matrix}
1,\quad z<0.\\
0,\quad otherwise.\\
\end{matrix}
\right .
$$
可以看到当$C$接近$+\infin$，$y_i(w^Tx_i+b)-1$必须严格满足间隔条件。

由于"0/1损失函数"数学性质不好，所以有一些替代算是函数：

以下是列出三种损失函数的表格，包含公式：

| 损失函数   | 公式                                          |
| ---------- | --------------------------------------------- |
| Hinge 损失 | $$\ell_{\text{hinge}}(z) = \max(0, 1 - z)$$   |
| 指数损失   | $$\ell_{\text{exp}}(z) = \exp(-z)$$           |
| 对率损失   | $$\ell_{\text{log}}(z) = \log(1 + \exp(-z))$$ |

为了方便研究，引入**松弛变量(slack variables)**:
$$
\epsilon_i = \ell_{hinge}(y_i(w^Tx_i+b))
$$
此时原优化问题的约束条件发生变化,以$Hinge$损失为例：
$$
\max(0,1-y_i(w^Tx_i+b)) = \epsilon_i\\
\\
\epsilon=
\left\{
\begin{matrix}
0,\quad 1-y_i(w^Tx_i+b)<=0\\
1-y_i(w^Tx_i+b),\quad 1-y_i(w^Tx_i+b)>0
\end{matrix}
\right .\\
\\
y_i(w^Tx_i+b)>=1-\xi_i
$$
我们就得到了**软间隔支持向量机**：
$$
\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^m\xi_i\\
s.t.\quad y_i(w^Tx_i+b)>=1-\xi_i,\\
\epsilon_i>=0\quad i=0,1,2,3,...,m.
$$
对于这个问题仍可以使用拉格朗日乘子法得出对偶问题：

### 软间隔SVM求解

设出目标函数，目标函数涉及一个正则项和松弛变量的惩罚项：
$$
L(w, b, \alpha, \xi, \mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i + \sum_{i=1}^m \alpha_i (1 - \xi_i - y_i (w^T x_i + b)) - \sum_{i=1}^m \mu_i \xi_i
$$
其中，$$\alpha_i > 0$$，$$\mu_i \geq 0$$ 是拉格朗日乘子。

通过对 $$w$$, $$b$$, $$\xi$$ 取偏导并令其等于零，可得：
$$
w = \sum_{i=1}^m \alpha_i y_i x_i
$$
$$
0 = \sum_{i=1}^m \alpha_i y_i
$$
$$
C = \alpha_i + \mu_i
$$

将上述结果代入原始目标函数中，可以得到对偶问题：
$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j
$$
$$
\text{s.t.} \sum_{i=1}^m \alpha_i y_i = 0,
$$
$$
0 < \alpha_i < C, \quad i = 1, 2, \ldots, m.
$$

通过求解上述对偶问题，可以得到支持向量机的最优解。

好的，下面是详细内容，其中所有的数学符号都用 $$ 包围。

### 软间隔支持向量机的KKT条件

对于软间隔支持向量机，KKT条件如下：

$$
\begin{cases}
\alpha_i \geq 0, \\
\mu_i \geq 0, \\
y_i (f(x_i)) - 1 + \xi_i \geq 0, \\
\alpha_i (y_i (f(x_i)) - 1 + \xi_i) = 0, \\
\xi_i \geq 0, \\
\mu_i \xi_i = 0.
\end{cases}
$$

- $$\alpha_i \geq 0$$：拉格朗日乘子必须是非负的。
- $$\mu_i \geq 0$$：用于松弛变量的拉格朗日乘子必须是非负的。
- $$y_i (f(x_i)) - 1 + \xi_i \geq 0$$：这是约束条件，确保样本 $$i$$ 的预测结果满足软间隔约束。
- $$\alpha_i (y_i (f(x_i)) - 1 + \xi_i) = 0$$：这是互补松弛条件，意味着如果 $$\alpha_i$$ 非零，那么约束 $$y_i (f(x_i)) - 1 + \xi_i = 0$$ 必须严格成立。
- $$\xi_i \geq 0$$：松弛变量必须是非负的。
- $$\mu_i \xi_i = 0$$：这是互补松弛条件，意味着如果 $$\xi_i$$ 非零，那么 $$\mu_i$$ 必须是零，反之亦然。

对于任何训练样本 $$(x_i, y_i)$$，总有 $$\alpha_i = 0$$ 或 $$y_i (f(x_i)) - 1 + \xi_i = 0$$。若 $$y_i (f(x_i)) - 1 + \xi_i > 0$$，则 $$\alpha_i = 0$$，表明该样本对优化目标无影响；若 $$\alpha_i > 0$$，则表明 $$y_i (f(x_i)) - 1 + \xi_i = 0$$，即该样本是支持向量。

这些KKT条件描述了软间隔支持向量机的优化问题的解的性质，确保模型在满足软间隔约束的同时，最大化分类间隔。

最终通过求解$\alpha_i$可以得到$w^T$和$b$的最佳值。

## 阅读材料

### 拉格朗日对偶问题

KKT条件：结合原始约束条件，拉格朗日式的约束（求导），拉格朗日乘子的非负性以及互补松弛性

![image-20240727163326561](C:\Users\唐浩钏\AppData\Roaming\Typora\typora-user-images\image-20240727163326561.png)

![image-20240727163346279](C:\Users\唐浩钏\AppData\Roaming\Typora\typora-user-images\image-20240727163346279.png)

![image-20240727163400552](C:\Users\唐浩钏\AppData\Roaming\Typora\typora-user-images\image-20240727163400552.png)

### 投影定理

在有限维的原始空间中，许多数据分布是非线性的，无法通过简单的线性分类器进行有效分类。通过将数据映射到更高维的特征空间，可以利用更高维空间的特性来解决这些问题。以下是一些关于为什么在有限维空间中有一个更高维的特征空间的原因：

#### Hilbert空间

在许多情况下，原始有限维空间的数据可以被映射到一个无限维的Hilbert空间。在这个空间中，许多复杂的非线性关系可以通过简单的线性关系来描述。这是函数空间和再生核Hilbert空间（RKHS）概念的基础。

根据投影定理，任何有限维空间都可以嵌入到更高维的空间中。这个更高维的空间包含了原始空间的所有特征，同时还可以容纳更多的特征，从而使得数据分析和处理变得更加灵活和准确。

#### 数学证明

考虑一个非线性映射函数 \($\phi$\)，它将原始空间 \($\mathcal{X}$\) 中的点映射到高维特征空间 \($\mathcal{H}$\)，即：

$$
\phi: \mathcal{X} \to \mathcal{H}
$$

如果 $\mathcal{H}$ 是一个Hilbert空间，则存在一个核函数 \($k$\)，使得：

$$
k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}
$$

这个核函数 \($k$\) 定义了一个更高维的特征空间，而这个特征空间可能是无限维的。

总之，在有限维空间中，通过合适的映射函数或核函数，可以构造一个更高维的特征空间，以解决原始空间中的非线性问题。这种技术在机器学习中非常重要，特别是在支持向量机和其他核方法中。

### 关于核函数的性质

#### 半正定矩阵

一个矩阵 \( $K$\) 被称为半正定（positive semi-definite），如果对于任意非零向量 \( $\mathbf{z}$ \)，都有：

$$
\mathbf{z}^T K \mathbf{z} \geq 0
$$

也就是说，对于任何向量 \( $\mathbf{z} $\)，当它与矩阵 \( $K$ \) 进行二次型运算时，结果总是非负的。半正定矩阵的一个重要性质是它的特征值非负。

#### 再生核希尔伯特空间（RKHS）

每一个核函数都隐含地定义了一个再生核希尔伯特空间（$RKHS$）。在这个空间中，核函数具有“再生性”，即对于任意函数 \($f$\) 属于这个空间，都有：

$$
f(x) = \langle f, \kappa(x, \cdot) \rangle_{\mathcal{H}}
$$

这意味着在 RKHS 中，通过核函数可以很方便地计算内积，从而进行各种线性操作。

这段话的意思是，如果我们有一个半正定的核矩阵 \(K\)，那么我们总能找到一个特征映射函数 \(\phi\)，将原始数据点映射到一个高维的特征空间，并且在这个特征空间中，核函数的计算就等同于特征映射后的内积。换句话说，任何一个符合条件的核函数都隐含地定义了一个再生核希尔伯特空间（RKHS）。

#### 核函数与 RKHS 之间的关系

根据 Mercer 定理，任何一个满足条件的半正定核函数都可以视作某个高维空间（可能是无限维空间）中的内积，即存在一个特征映射函数 \($\phi$\)，使得：

$$
\kappa(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}
$$

(与RKHS对比)这意味着，我们可以通过核函数 \($\kappa$\) 定义一个特征映射 \($\phi$\)，将输入数据从原始空间 \($\mathcal{X}$\) 映射到一个高维的特征空间 \($\mathcal{H}$\)。在这个高维空间中，核函数 \($\kappa(x, y)$\) 就是特征向量 \($\phi(x)$\) 和 \($\phi(y)$\) 之间的内积。

#### 从半正定矩阵到特征映射

给定一个半正定的核矩阵 \($K$\)，根据特征值分解（Eigenvalue Decomposition），我们可以将 \($K$\) 分解为：

$$
K = Q \Lambda Q^T
$$

其中，\(Q\) 是特征向量矩阵，\($\Lambda$) 是特征值对角矩阵。我们可以通过特征向量和特征值构造特征映射 ($\phi$)，使得 \($\phi(x_i)$\) 在高维空间中的表示满足核矩阵的半正定性要求。

### 正则化

好的，以下是含有替代损失函数的支持向量机（SVM）目标函数及其解释，其中所有的数学符号都用 $$ 包围。

#### 含有替代损失函数的SVM目标函数

我们可以将 0/1 损失函数替换成别的替代损失函数，从而得到不同的学习模型。这些模型的性质与所用的替代函数直接相关，目标函数可以写为：

$$
\min_f \Omega(f) + C \sum_{i=1}^m \ell(f(x_i), y_i)
$$

其中：

- $$\Omega(f)$$ 称为“结构风险”（structural risk），用于描述模型 $$f$$ 的某些性质。
- $$\sum_{i=1}^m \ell(f(x_i), y_i)$$ 称为“经验风险”（empirical risk），用于描述模型 $$f$$ 在训练数据上的误差。
- $$C$$ 是一个正的常数，用于平衡结构风险和经验风险的权重。

#### 结构风险和经验风险

- $$\Omega(f)$$ 用于描述模型的复杂度或平滑度。常见的 $$\Omega(f)$$ 形式包括 $$L_p$$ 范数（例如，$$L_2$$ 范数 $$\|w\|^2$$）等。
- $$\ell(f(x_i), y_i)$$ 是损失函数，用于衡量模型在样本 $$x_i$$ 上的预测值 $$f(x_i)$$ 与真实标签 $$y_i$$ 之间的差异。

#### 正则化

在机器学习中，正则化用于防止过拟合，确保模型在训练数据上的良好表现能够推广到未见过的数据上。目标函数中的正则化项 $$\Omega(f)$$ 可以帮助控制模型的复杂度。

综上所述，含有替代损失函数的SVM目标函数通过平衡结构风险和经验风险，优化模型的性能，确保在训练数据和新数据上都有良好的表现。
