# 神经网络（neural networks）

## 神经元

## 感知机模型（Perceptron）

### 感知机模型

感知机是由神经元复合的**线性分类器**。
$$
y =f(\sum_{i=1}^n w_ix_i -\theta) = f(w^Tx-\theta)
$$
其中：

- x为样本的特征向量。
- w为权重向量。
- θ称为偏置（阈值）

如何这里的$f$为阶跃函数$\epsilon(·)$，则感知机模型可以表示为激活函数：
$$
(1)y =\epsilon(w^Tx-\theta) =\left\{
\begin{matrix}
\ 1,w^Tx>=\theta\\
\ 0,w^Tx<\theta 
\end{matrix}
\right.
$$
由$n$维空间中的超平面方程：
$$
w_1x_1+w_2x_2+w_3x_3+...+w_nx_n+b=w^Tx+b=0
$$
可知$(1)$式将n维空间划分，可以看作一个超平面，将n维空间划分为$w^Tx>=\theta$和$w^Tx<\theta$两个空间，落在前者的输出值为1，后者的输出值为0，于是实现了分类功能。

### 感知机的学习策略以及参数的调整规则

所以**线性模型**中的思考，我们需要对权重向量$w^T$和偏置$\theta$进行选择，这就是**感知机的学习**

不妨考虑一种简单的情况，现在有特征矩阵$x_1$=[2, 3]，$x_2$=[1,5]，包含标签的向量$y=\{1,0\}$

- 初始化权重向量$w$：假设$w=[1,1]$，偏置$\theta=2$；

- 计算输出：

  - 根据公式$(1)$, 有$w^Tx_1-\theta=1·2+1·3-2=3>0$, 因此被判定为1，符合真实输出；
  - 对于$x_2$，计算得到$w^Tx_2-\theta = 6-2=4>0$，不符合输出，因此需要调整权重向量$w$和偏置$\theta$

- 提出以下优化规则：
  $$
  w' = w+η(y-y_{pred})x\\
  \theta'=\theta-η(y-y_{pred})
  $$
  ​	假设模型的学习率$η$为1，于是$w' = w + η(0-1)·[1,5] = [0,-4]$,$\theta' = 2 + 1 =3$

  - 对于$x_1$，再次计算$w^Tx_1-\theta= -15 < 0$，与真实输出不符；
  - 对于$x_2$，易得$w^Tx_1-\theta = -23 < 0$，输出为0符合标签

  由于输出仍不满足，因此继续根据规则调整权重和偏置验证计算。

  

  ## 多层前馈神经网络

  在前面我们提到，感知机是一种线性分类器，利用感知机模型构成的单层神经元只有输出层神经原进行激活函数处理，只能解决**线性可分的问题**。
  
  ![image-20240726164829724](神经网络（neural networks）.assets/image-20240726164829724.png)
  
  其中“与”“或”“非”都是线性可分的，即存在一个线性超平面能将他们分开，在感知机的学习过程中调整权重矩阵和偏置最终使得学习过程收敛（converge），而对于非线性可分的问题，例如图上的"异或"问题，可以看到不存在有且仅有一个线性超平面将他们分开，这使得学习过程发生振荡（fluctuation），无法确定解。
  
  ### 含隐藏层的多层功能神经元
  
  为了解决非线性可分问题，我们引入多层功能神经元的概念，这里两层感知机模型就能解决上“异或”问题.
  
  ![image-20240726165420935](神经网络（neural networks）.assets/image-20240726165420935.png)
  
   更一般的多层功能神经元构成的神经网络如图所示：
  
  ![image-20240726165606198](神经网络（neural networks）.assets/image-20240726165606198.png)

**每层神经元与下一层神经元全互连，同层神经元不进行连接，也不跨层连接，上一层神经元的输出作为下一层神经元的输入**，我们定义这样的神经网络为"**多层前馈神经网络**"（multi-layer feedback neural networks）.神经元之间用**“连接权”和阈值**进行连接。

单隐层前馈网络是指只有一个隐藏层，双隐层前馈网络有两层隐藏层，需要注意的是，输入层神经元没有激活函数进行处理，仅仅作为输入。

## 误差逆传播算法(BP神经网络)

由于构造了复杂的学习模型多层前馈神经网络，我们的单层感知机模型的学习方法$\Delta w = η(y-y_{pred})x$

$\Delta\theta = -η(y-y_{pred})$已经无法正常运行。因此我们需要更适用的算法。

**误差逆传播算法(error BackPropagation，BP)**是一种适用于多层前馈神经网络的学习算法（BP算法不止适用于多层前馈神经网络）。

![image-20240726171227862](神经网络（neural networks）.assets/image-20240726171227862.png)

我们来解释一下BP算法的具体过程。

1. **训练集定义：**
   - 设定训练集 $(D = \{(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)\})$，其中$ (x_i \in \mathbb{R}^d)，(y_i \in \mathbb{R}^l)$，即每个示例由 $d$个属性输入，输出 $l$ 维实值向量。

2. **BP网络结构：**
   - 图 5.7 展示了一个具有输入层、隐藏层和输出层的三层前馈神经网络：
     - **输入层：** 包含$ (d)$ 个输入神经元。
     - **隐藏层：** 包含 $(q) $个神经元。
     - **输出层：** 包含 $(l)$ 个神经元。

3. **数学符号和公式：**
   
   - **隐层神经元的输入：**
     - 第 $h$ 个隐层神经元接收到的输入为：$(a_h = \sum_{i=1}^{d} v_{ih} x_i)$，其中 $w_{ih}$ 表示从输入层第 $i$ 个神经元到隐层第 $h$ 个神经元的连接权值。
   - **输出层神经元的输入：**
     - 第 $j$ 个输出层神经元接收到的输入为：$(β_j = \sum_{h=1}^{q} w_{hj} b_h)$，其中 \($b_h$\) 是隐层第 \($h$\) 个神经元的输出，\($w_{hj}$\) 是从隐层第 \($h$\) 个神经元到输出层第 \($j$\) 个神经元的连接权值。
   
4. **激活函数：**
   
   - 假设隐层和输出层神经元都使用 $Sigmoid $函数作为激活函数。
   
5. **输出层的计算：**
   
   - 对于训练例 $((x_k, y_k))$，假设神经网络的输出为 $$(y_{k1}^*, y_{k2}^*, ..., y_{kl}^*)$$(表示第k个输入的第j个神经元输出)，即 $$(y_{kj}^* = f(β_j - θ_j))$$，其中 \($f$\) 为激活函数，\($θ_j$\) 为第 \($j$\) 个输出神经元的阈值。
   
   网络中有 **$(d+l+1)q+l$**个参数需确定**:输入层到隐层的$d\times q$个权值，隐层到输出层的$q\times l$个权 层神 的权值、 $q$ 个隐层神经元的偏置，$l$个输出层神经元的阈值。**
   
   直到上面为止还是能沿用单层感知机的理解，接下来我们也同单层模型一样，考虑学习方法以确定权重矩阵和偏置。
   
   ### 均方误差
   
   为了找到一组好的参数值$(w,\theta)$，我们需要一个评估指标描述什么是"好的"。此时我们想起线性回归模型的处理方法，这里我们适用均方误差来进行。
   
   对于训练例 $(x_k, y_k)$，定义网络在 $(x_k, y_k)$ 上的均方误差为：
   $$
   E_k = \frac{1}{2} \sum_{j=1}^{l} (y_{kj}^* - y_{kj})^2
   $$
   
   
   ### 更新过程与梯度下降
   
   $v$为任意参数，$v$的更新过程表示为更新估计式：
   $$
   v \leftarrow v+\Delta v \ .
   $$
   以隐层到输出层的连接权$w_{hj}$为例，使用**梯度下降**(gradient descent)策略，梯度下降在线性回归模型中也关键方法。给定学习率$η$，有：
   $$
   \Delta w_{hj} = -η\frac{∂E_k}{∂w_{hj}}
   $$
   
   ### 链式法则
   
   我们可以注意到$E_k$是$w_{hj}$的复合函数，这里使用复合函数求导的链式法则，有：
   $$
   (1)\frac{∂E_k}{∂w_{hj}}=\frac{∂E_k}{∂y^*_{kj}}·\frac{∂y^*_{kj}}{∂\beta_j}·\frac{∂\beta_j}{∂w_{hj}}\ .
   $$
   由于$(β_j = \sum_{h=1}^{q} w_{hj} b_h)$，所以：
   $$
   (2)\frac{∂\beta_j}{∂w_{hj}} = b_h\ .
   $$
   定义$g_j$:
   $$
   (3)g_j =-\frac{∂E_k}{∂y^*_{kj}}·\frac{∂y^*_{kj}}{∂\beta_j}\\
   	=-(y^*_{kj}-y_{kj})f'(\beta_j-\theta_j)\\
   	=y^*_{kj}(1-y^*_{kj})(y_{kj}-y^*_{kj})
   $$
   最终可以得到
   $$
   \Delta w_{hj} =ηg_jb_h\ .
   $$
   类似可得其他参数：
   $$
   \Delta \theta_j = - \eta g_j，
   $$
   $$
   \Delta w_{ih} = \eta e_h x_i， 
   $$
   $$
   \Delta \gamma_h = - \eta e_h， \quad
   $$
   
   $$
   e_h = - \frac{\partial E_k}{\partial b_h} \cdot \frac{\partial b_h}{\partial \alpha_h}
   $$
   $$
   = - \sum_{j=1}^{l} \frac{\partial E_k}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial b_h} f'(\alpha_h - \gamma_h)
   $$
   $$
   = \sum_{j=1}^{l} w_{hj} g_j f'(\alpha_h - \gamma_h)
   $$
   $$
   = b_h (1 - b_h) \sum_{j=1}^{l} w_{hj} g_j。 \quad (5.15)
   $$
   
   

### BP算法手动实现

对每个训练样例，BP算法执行以下操作：

**（1）先将输入示例提供给输入层神经元，然后逐层将信号前传，直到产生输出层的结果；**

**（2）然后计算输出层的误差（第4-5行）**

**（3）再将误差逆向传播至隐层神经元（第6行）**

**（4）最后根据各层神经元的误差来对连接权和阈值进行调整（第7行）。**

该迭代过程循环进行，直到达到某些停止条件为止，例如训练误差达到一个很小的值。图给出了在2个属性、5个样本的西瓜数据上，随着训练样例的增加，网络参数和分类边界的变化情况。

![image-20240726180539860](神经网络（neural networks）.assets/image-20240726180539860.png)

```
输入：训练集 $D = \{(x_k, y_k)\}_{k=1}^m$
      学习率 $\eta$

过程：

1. 在 $[-1, 0, 1]$ 范围内随机初始化网络中所有连接权和阈值
2. repeat
3. for all $(x_k, y_k) \in D$ do
4. 根据式(5.3)计算当前样本的输出 $y_{kj}^*$
5. 根据式(5.10)计算输出层神经元的梯度 $g_j$
6. 根据式(5.11)计算隐层神经元的梯度 $e_h$
7. 根据式(5.12)-(5.14)更新连接权 $w_{hj}, v_{ih}$ 和阈值 $\theta_j, \gamma_h$
8. until 达到停止条件

输出：连接权与阈值确定的多层前馈神经网络
```

附上手写的源码：

```python
import numpy as np

class nn:
    def __init__(self, input_size, hidden_size, output_size, eta):
        self.weight_1 = np.random.randn(hidden_size, input_size)
        self.weight_2 = np.random.randn(output_size, hidden_size)
        self.thre_1 = np.random.randn(hidden_size)
        self.thre_2 = np.random.randn(output_size)
        self.eta = eta

    def _get_input(self, weight, X):
        return np.dot(X, weight.T)
    
    def _sigmoid(self, input, thre):
        input = np.array(input) - thre
        return 1 / (1 + np.exp(-input))
    
    def _get_output(self, input, thre):
        return self._sigmoid(input, thre)
    
    def _get_MSE(self, output, pre_output):
        return np.mean((pre_output - output) ** 2) / 2
    
    def _get_new_weight(self, grad, X):
        delta_weight = self.eta * np.outer(grad, X)
        return delta_weight
    
    def _get_new_thre(self, grad):
        return -self.eta * np.sum(grad, axis=0)
    
    def _get_output_grad(self, output, y):
        error_output = output - y
        grad_output = error_output * output * (1 - output)
        return grad_output
    
    def _get_hide_grad(self, weight, output, grad_output):
        error_hidden = np.dot(weight.T, grad_output)
        grad_hidden = error_hidden * output * (1 - output)
        return grad_hidden

    def fit(self, X, y, num_epochs=1000):
        for epoch in range(num_epochs):
            # 前向传播
            input_1 = self._get_input(self.weight_1, X)
            output_1 = self._get_output(input_1, self.thre_1)
            
            input_2 = self._get_input(self.weight_2, output_1)
            output_2 = self._get_output(input_2, self.thre_2)
            
            # 计算损失
            loss = self._get_MSE(y, output_2)
            
            # 反向传播
            grad_output = self._get_output_grad(output_2, y)
            grad_hidden = self._get_hide_grad(self.weight_2, output_1, grad_output)
            
            # 更新权重和偏置
            delta_weight_2 = self._get_new_weight(grad_output, output_1)
            self.weight_2 -= delta_weight_2
            self.thre_2 -= self._get_new_thre(grad_output)
            
            delta_weight_1 = self._get_new_weight(grad_hidden, X)
            self.weight_1 -= delta_weight_1
            self.thre_1 -= self._get_new_thre(grad_hidden)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        input_1 = self._get_input(self.weight_1, X)
        output_1 = self._get_output(input_1, self.thre_1)
        
        input_2 = self._get_input(self.weight_2, output_1)
        output_2 = self._get_output(input_2, self.thre_2)
        
        return output_2
        
input_size = 4
hidden_size = 5
output_size = 3
eta = 0.7

model = nn(input_size, hidden_size, output_size, eta)

X = np.array([0.5, 1.0, 1.5, 2.0])
y = np.array([0.1, 0.2, 0.3])

model.fit(X, y)

prediction = model.predict(X)
print(f"Prediction: {prediction}")

'''
Epoch 0, Loss: 0.08452484439144649
Epoch 100, Loss: 0.00043097425030784627
Epoch 200, Loss: 6.173784504106113e-05
Epoch 300, Loss: 6.111382932194645e-06
Epoch 400, Loss: 5.274844786786648e-07
Epoch 500, Loss: 4.445371701622411e-08
Epoch 600, Loss: 3.7266722662723944e-09
Epoch 700, Loss: 3.1195917455011153e-10
Epoch 800, Loss: 2.6102862257159366e-11
Epoch 900, Loss: 2.1838569148871845e-12
Prediction: [0.1000008  0.19999945 0.3000004 ]
'''

```



### 累积误差逆传播算法（累积BP）

前面介绍的BP算法针对单个均方误差$E_k$，这意味着每次参数更新都只针对一个样例数据，导致更新频繁以及可能出现的不同样例导致的参数调整抵消等问题。所以为了使数据集整体达到同样的误差极小点，累积BP直接针对累积误差最小化:$E = \frac{1}{m} \sum_{k=1}^{m}E_k$，读取数据集一遍之后再进行更新，然而累积BP在累积误差下降到一定程度时，可能出现下降缓慢的情况，这时标准BP会得到更好的解。

**类似随机梯度下降和标准梯度下降的区别。**

### 过拟合

- 早停：将数据集分成训练集和验证集，训练集用来计算梯度，更新连接权和偏置，验证集用来估计误差，如果训练集误差降低但是验证机误差升高，则停止训练（类似决策树的后剪枝）
- 正则化：在误差目标函数增加一个描述网络复杂度的部分，例如**连接权和偏置的平方和**：

$$
E=\lambda \frac{1}{m}\sum_{k=1}^m E_k + (1-\lambda)\sum_i w_i^*\ ,
$$

其中$\lambda \in (0,1)$用于经验误差与网络复杂度这两项进行折中。常使用**交叉验证法**来估计。