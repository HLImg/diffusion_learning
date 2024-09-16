# Score-Matching with Langevin Dynamics

## Introduction

现有的生成模型主要分为两类：

- **基于似然的模型**（显式建模）：通过（近似）最大似然直接学习分布的**概率密度函数**。典型的模型包括：自回归模型、归一化流模型、基于能量的模型和变分自编码器。局限性：**严格限制模型的架构以确保对似然的计算具有易于处理的归一化常数**，并且**必须依赖代理目标来近似最大似然训练**。
- **隐式生成模型**（隐式建模）：概率分布由其采样过程的模型隐式表示。典型的模型就是对抗生成网络（GAN）。局限性：**以GAN为例，其模型训练不稳定，可能导致坍塌**.

一种新的方法可以在一定程度上规避上述生成模型的限制，其核心思想是**对对数概率密度函数的梯度进行建模**，这被称为**Score function**。这种基于分数的模型**不需要有易于处理的归一化常数**（对于基于似然的生成模型），并且**可以直接通过score-matching来学习**。基于分数的模型与归一化流有联系，因此允许精确的似然计算和表示学习。score-based 生成模型具有如下的优点：

1. 可以得到GAN级别的采样效果，而不需要对抗学习
2. 灵活的模型结构：需要要考虑归一化项，即可以选择非归一化概率模型
3. 精确的对数似然估计计算
4. 流程可逆

## Score-based Model

给定数据集 $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_N\} $，其中每个点都是从一个潜在数据分布 $p(\mathbf x) $中独立抽取的。为了构建生成模型，首先需要一种表示概率分布的方法。基于似然的模型是直接对PDF或PMF建模。让 $f_\theta(\mathbf x)\in \mathbb R $表示为一个**由可学习的参数 $\theta $参数化的实值函数**，可以通过以下方式定义PDF：

$$
p_\theta(\mathbf x) = \frac{e^{-f_\theta(\mathbf x)}}{Z_\theta}
$$

其中， $Z_\theta > 0 $是一个取决于 $\theta $的**归一化常数**， $\int p_\theta(\mathbf x)d\mathbf x=1 $。这里的 $f_\theta(\mathbf x) $通常称为**非归一化概率模型**或**基于能量的模型**。

此时，可以通过**最大化**数据的对数似然来训练 $p_\theta(\mathbf x) $：

$$
\max_\theta \sum_{i=1}^N \log p_\theta(\mathbf x_i)
$$

然而上述公式需要 $p_\theta(\mathbf x) $为**归一化的概率密度函数**，这通常是不可能的。因为，为了计算 $p_\theta(\mathbf x) $必须要计算 $Z_\theta $，而 $f_\theta(\mathbf x) $是一个难以处理的量。因此，为了使最大化似然训练可行，基于似然的模型必须**限制其模型架构**使得 $Z_\theta $易于处理（例如自回归模型中的因果卷积，归一化流模型中的可逆网络），或者**近似归一化常数**（例如，变分推理，MCMC采样），这在计算上可能是昂贵的。

通过对**得分函数**而**不是密度函数**进行**建模**，可以**避开难以处理的归一化常数**，分布的得分函数 $p(\mathbf x) $定义为：

$$
\nabla_{\mathbf x}\log p(\mathbf x) = \frac{\partial \log p(\mathbf x)}{\partial \mathbf x}
$$

得分函数的模型称为**基于score的模型**，将其表示为 $s_\theta(\mathbf x) $。基于分数的模型是这样学习的： $s_\theta(\mathbf x)\approx \nabla_{\mathbf x}\log p(\mathbf x) $，并且可以**参数化**而无需担心**归一化常数**。这样可以通过一个基于能量的模型来参数化基于分数的模型：

$$
\begin{aligned}
s_\theta(\mathbf x) &= \nabla_{\mathbf x}\log p_\theta(\mathbf x)\\
&=\frac{\partial \log (\frac{e^{-f_\theta(\mathbf x)}}{Z_\theta})}{\partial \mathbf x}\\
&= -\nabla_{\mathbf x}f_\theta(\mathbf x) - \nabla_{\mathbf x}\log Z_\theta\\
&=-\nabla_{\mathbf x}f_\theta(\mathbf x)
\end{aligned}
$$

上述过程展示了基于分布的模型可以无需考虑过归一化常数 $Z_\theta $。

> Fisher散度通常位于两个分布 $p $和 $q $之间：

$$
\mathbb E_{p(\mathbf x)}[\|\nabla_{\mathbf x}\log p(\mathbf x) - \nabla_{\mathbf x}\log q(\mathbf x)\|_2^2]
$$

与基于似然的模型类似，可以通过**最小化Fisher散度**来**训练基于分数的模型和数据分布之间的关系**:

$$
\mathbb E_{p(\mathbf x)}[\|\nabla_{\mathbf x}\log p(\mathbf x) - s_\theta(\mathbf x)\|^2_2]
$$

直观来说，Fisher散度度量了**真值数据分数**与**基于得分的模型**之间的**平方 $\mathcal l_2 $距离**。然而，直接计算是不可行的，因为它需要**未知的数据分布** $\nabla_{\mathbf x}\log p(\mathbf x) $。存在一系列称为**score-matching**的方法（**去噪分数匹配**和**切片分数匹配**）在**不知道真实数据score的情况下最小化Fisher散度**。分数匹配目标可以**直接在数据集上**估计，并通过随机梯度下降进行优化，类似于训练基于似然的模型的**对数似然目标**（具有已知的归一化常数）。并且可以通过**最小化分数匹配目标**来训练基于分数的模型，**而不需要对抗性优化**。

此外，使用**分数匹配目标**提供了相当大的建模灵活性，Fisher散度本身不需要 $s_\theta(\mathbf x) $是任何归一化分布的实际得分函数（它只是比较二者之间的距离，不需要知道 $s_\theta(\mathbf x) $的形式）。事实上，基于score模型的唯一要求就是它应该是具有**相同输入和输出维度**的向量值函数。

*As a brief summary, we can represent a distribution by modeling its score function, which can be estimated by training a score-based model of free-form architectures with score matching.*

## Langevin dynamics

一旦训练了基于分数的模型 $s_\theta(\mathbf x)\approx \nabla_{\mathbf x}\log p(\mathbf x) $，可以使用称之为**Langevin dynamics**的**迭代过程**从中抽取样本。

*Langevin Dynamics*提供了一个从MCMC程序使用它的分数函数 $\nabla_{\mathbf x}\log p(\mathbf x) $从分布 $p(\mathbf x) $中采样。具体来说，它从**任意先验分布** $\mathbf x_0\sim \pi(\mathbf x) $中**初始化链**，然后迭代以下过程：

$$
\mathbf x_{i+1}\leftarrow \mathbf x_i + \epsilon \nabla_{\mathbf x}\log p(\mathbf x) + \sqrt{2\epsilon}\mathbf z_i, i=0, 1, 2,\cdots, K,
$$

其中， $\mathbf z_i \sim \mathcal N(0, I) $。当 $\epsilon\rightarrow 0, K\rightarrow \infty $时，在某些正则条件下，从上述迭代程序获得的收敛到 $p(\mathbf x) $的一个样本。在实践中，当 $\epsilon $足够小和 $K $足够大的时候，误差是可以被忽略的。

**Langevin dynamics**仅通过 $\nabla_{\mathbf x}\log p(\mathbf x) $来处理 $p(\mathbf x) $。当 $s_\theta(\mathbf x)\approx \nabla_{\mathbf x}\log p(\mathbf x) $时，可以从基于分数的模型中生成样本 $s_\theta(x) $并将其带入上述的迭代过程。

#### Naive score-based generative modeling and its pitfalls

原生的通过score-matching训练score-based模型，然后通过Langevin dynamics生成样本在**实践中**取得的成功有限。

![](http://qiniu.lianghao.work/smld.jpg)

关键的挑战是**在低密度区域估计的得分函数是不准确的**，因为该区域可以用于计算得分匹配目标的数据点很少。这是符合预期的，因为得分匹配最小化了Fisher散度：

$$
\mathbb E_{p(\mathbf x)}[\|\nabla_{\mathbf x}\log p(\mathbf x) - s_\theta(\mathbf x)\|_2^2]\\
= \int p(\mathbf x)\|\nabla_{\mathbf x}\log p(\mathbf x) - s_\theta(\mathbf x)\|^2_2d\mathbf x
$$

因为分数匹配的加权项为 $p(\mathbf x) $，因此当 $p(\mathbf x) $很小时，它们在低密度区域是被忽略的。这导致了生成的样本低于标准。

![](http://qiniu.lianghao.work/pitfalls.jpg)

当数据驻留在高维空间时，而初始样本在低密度区域，使用Langevin dynamics会阻止生成高质量的样本。

#### Score-based generative modeling with multiple noise perturbations

为了避免低密度区域不能准确估计分数的问题，**使用噪声扰乱数据点**，并在数据点上训练基于分布的模型。当**噪声幅度**足够大时，它可以**填充低数据密度区域**来**提高估计分数的准确性**。

![](http://qiniu.lianghao.work/single_noise.jpg)

此时关键的问题是：**如何为这个扰动过程选择合适的噪声尺度**？较大的噪声虽然可以覆盖更多的低密度区域，以获得更好的分数估计，但它会**过度破坏数据并显著改变原始分布**。另一方面，较小的噪声虽然对原始数据分布的破坏较少，但不能覆盖更多的低密度区域。

Song等同时使用**多个尺度的噪声扰动**：假设总是使用**各向同性高斯噪声**扰动数据，并且总共有 $L $个逐渐增加的标准差
 $\sigma_1<\sigma_2<\cdots <\sigma_L $。首先使用每个高斯噪声 $\mathcal N(0, \sigma_i^2\mathbf I),i=1, 2, \cdots, L $来扰动数据分布 $p(\mathbf x) $：

$$
p_{\sigma_i}(\mathbf x) = \int p(\mathbf y)\mathcal N(\mathbf x;\mathbf y,\sigma_i^2\mathbf I) d\mathbf y
$$

通过采样 $\mathbf x\sim p(\mathbf x) $和计算 $\mathbf x + \sigma_i\mathbf z $可以轻松地从 $p_{\sigma_i}(\mathbf x) $中采样。

接下来，估计每个被噪声扰动的分布的得分函数 $\nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x) $，通过训练**噪声条件得分网络** $s_\theta(\mathbf x, i) $和分数匹配，这样对所有的 $i=1,2,\cdots, L $有 $s_\theta(\mathbf x, i)\approx \nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x) $。

![](http://qiniu.lianghao.work/multi_scale.jpg)

**训练的目标 $s_\theta(\mathbf x, i) $是所有噪声尺度的Fisher散度的加权和**：

$$
\sum_{i=1}^L \lambda(i)\mathbb E_{p_{\sigma_i(\mathbf x)}}\left[\|\nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x) - s_{\theta}(\mathbf x, i)\|_2^2\right]
$$

其中， $\lambda(i)\in \mathbb R_{>0} $是一个正权重函数，通常选择为 $\lambda(i)=\sigma_i^2 $。训练目标可以通过**分数匹配**进行优化。

训练基于噪声条件的分数模型 $s_\theta(\mathbf x, i) $后，可以通过**Langevin Dynamics**按照顺序 $i=L, L -1, \cdots, 1 $来生成样本。这种方法称之为**退化Langevin Dynamics**（因为噪声的标准差 $\sigma_i $随着时间的推移逐渐减少）。

**调整具有多个噪声尺度的分数生成模型的实用技巧**

1. 选择 $\sigma_1<\sigma_2<\cdots < \sigma_L $作为几何级数， $\sigma_1 $足够小并且 $\sigma_L $相当于所有训练数据点之间的**最大成对距离**。 $L $通常是数百或数千的数量级。
2. **使用带有跳跃连接的U-Net来参数化基于分数的生成模型 $s_\theta(\mathbf x, i) $**
3. 在测试生成样本时，对分数模型的权重应用EMA

## SDE

### Perturbing data with SDE

当噪声尺度的数量接近无穷大时，本质上随着噪声水平的不断增加而扰动数据分布。在这种情况下，噪声扰动数据是一个**连续时间随机过程**。**许多随机过程（特别是扩散过程）都是随机微分过程（SDE）的解**。一般来说，SDE具有以下形式：

$$
d\mathbf x = f(\mathbf x, t) dt + g(t)d\mathbf w
$$

其中， $f(\cdot, t):\mathbb R^d\rightarrow \mathbb R^d $是一个向量值函数，称为**飘移系数**， $g(t)\in \mathbb R $是一个实值函数，**称为扩散系数，描述了不确定的变化过程**。 $\mathbf w $表示**标准布朗运动**， $d\mathbf w $可以看作**无穷小的白噪声**。SDE的解是随机变量的连续集合 $\{\mathbf x_t\}_{t\in [0, T]} $。让 $p_t(\mathbf x) $表示（边缘）概率密度函数 $\mathbf x(t) $，其中 $ t\in [0, T] $类似于 $i=1,2,\cdots,L $。但使用有限数量的噪声尺度时， $p_t(\mathbf x) $类似于 $p_{\sigma_i}(\mathbf x) $。随机过程持续足够长的时间对数据 $p(\mathbf x) $进行扰动，最终将其近似为**易于处理的噪声分布** $\pi(\mathbf x) $，**称之为先验分布**。

> 在DDPM中，前向扩散过程中第 $t $步样服从 $\mathbf x_t \sim \mathcal N(\sqrt{\bar\alpha_t}\mathbf x_0, (1-\bar\alpha_t)I) $。当固定采样的随机性是， $\mathbf x_t $可以被表示为变量 $t $的一个函数。因此，整个 $\mathbf x_t $是一个随机过程。对于一组确定的 $\{\mathbf x_t\}_{t=1}^T $，称为随机过程的一个实现，或是一条轨迹/轨道，而随机过程可以使用SDE表示。**如果将DDPM的迭代式从离散扩展到连续区间，即 $x_t\rightarrow x_t+\Delta t(\Delta t \rightarrow 0) $**，就可以得到SDE形式的扩散过程。

SDE是手工设计的，类似于手工设计的噪声尺度 $\sigma_1< \sigma_2 < \cdots < \sigma_L $。**添加噪声扰动的方法有多种，并且SDE的选择并不是唯一的**。因此，SDE被视为模型的一部分，就像 $\{\sigma_1,\sigma_2,\cdots, \sigma_L\} $。

### Reversing the SDE for sample generation

在**有限数量的噪声尺度**下，可以通过**反转**退火Langevin Dynamics扰动过程来生成样本。对于**无限噪声尺度**。可以通过反向SDE类似地生成样本。**任何SDE都有相应的反向SDE**，其闭式解：

$$
d\mathbf x = [f(\mathbf x,t)-g^2(t)\nabla_{\mathbf x}\log p_t(\mathbf x)dt + g(t)d\mathbf w]
$$

其中， $dt $表示**负无穷小时间步长**，因为SDE需要在时间（从 $t=T $到 $t=0 $）后向解决。为了计算反向SDE，需要估计 $\nabla_{\mathbf x}\log p_t(\mathbf x) $，这就是得分函数 $p_t(\mathbf x) $。

![](http://qiniu.lianghao.work/sde_schematic.jpg)

### Estimating the reverse SDE with score-based models and score matching

求解逆SDE需要知道**最终的数据分布 $p_t(\mathbf x) $**，以及**得分函数** $\nabla_{\mathbf x}\log p_t(\mathbf x) $。为了估计得分函数，训练一个基于时间的得分模型 $s_\theta(\mathbf x, t) $，使得 $s_\theta(\mathbf x, t) \approx \nabla_{\mathbf x} \log p_t(\mathbf x) $。这类似于基于噪声条件得分模型 $s_\theta(\mathbf x, i) $使用有限的噪声尺度，经过训练使得 $s_\theta(\mathbf x, i)\approx \nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x) $。
训练目标 $s_\theta(\mathbf x, t) $是**Fisher**散度的连续加权组合：

$$
\mathbb E_{t\in \mathcal U(0, T)}\mathbb E_{p_t(\mathbf x)}\left[\lambda(t)\|\nabla_{\mathbf x}\log p_t(\mathbf x) - s_\theta(\mathbf x, t)\|_2^2\right]
$$

其中， $\mathcal U(0, T) $表示时间间隔 $[0, T] $上的均匀分布， $\lambda $是正权重函数，通常使用

$$
\lambda(t)\propto \frac{1}{\mathbb E\left[\|\nabla_{\mathbf x(t)}\log p(\mathbf x(t)|\mathbf x(0))\|^2_2\right]}
$$

来**平衡不同时间段内不同分数匹配损失的大小**。与之前相同，Fisher散度的加权组合可以通过得分匹配方法进行有效优化和切片分数匹配。**一旦分数模型 $s_\theta(\mathbf x, t) $训练到最优**，可以**将其插入**到**反向SDE**的表达式中以获得估计的反向SDE。

$$
d\mathbf x = [f(\mathbf x, t) - g^2(t)s_\theta(\mathbf x, t)]dt + g(t)d\mathbf w
$$

测试采样过程可以先进行采样 $x(\mathbf T)\sim \pi $，然后利用上述反向SDE，得到样本 $\mathbf x(0) $。当分数模型 $s_\theta(\mathbf x, t) $训练收敛后，则有 $p_\theta\approx p_0 $。在这种情况下 $\mathbf x(0) $是数据分布的近似样本 $p_0 $。

**当 $\lambda(t) = g^2(t) $**时，在某些正则条件下，可以得到Fisher散度的加权组合与 $KL(p_0\| p_\theta) $之间的联系：

$$
KL(p_0(\mathbf x)\| p_\theta(\mathbf x)) \le\\ \frac{T}{2}\mathbb E_{t\in \mathcal U(0, T)}\mathbb E_{p_t(\mathbf x)}\left[\lambda(t)\|\nabla_{\mathbf x}\log p_t(\mathbf x) - s_\theta(\mathbf x, t)\|_2^2\right] + KL(p_T\|\pi)
$$

由于与KL散度的联系以及训练中**最小化KL散度**和**最大化似然**之间**等价性**，称 $\lambda(t)=g(t)^2 $为似然加权函数。使用这种似然加权函数，可以训练基于分数的生成模型来实现非常高的似然。

### How to solve the reverse SDE

通过使用数值SDE求解器求解估计的逆SDE，对于样本生成，可以模拟逆随机过程。当应用**Euler-Maruyama**方法与反向SDE时，它使用**有限时间步长**和**小高斯噪声**来**离散SDE**。具体来说，它选择一个小的负时间步长 $\Delta\approx 0 $，初始化 $t\leftarrow T $，并迭代以下过程直到 $t\approx 0 $：

$$
\begin{aligned}
\Delta \mathbf x &\leftarrow [f(\mathbf x, t) - g^2(t)s_\theta(\mathbf x, t)]\Delta t + g(t)\sqrt{|\Delta t|}\mathbf z_t\\
\mathbf x &\leftarrow \mathbf x + \Delta\mathbf x\\
t &\leftarrow t+\Delta t
\end{aligned}
$$

其中， $\mathbf z_t \sim \mathcal N(0, I) $。Euler-Maruyama方法在性质上与Langevin Dynamics相似（二者都通过**被高斯噪声扰动的得分函数**更新 $\mathbf x $）。除了该方法，其他数值SDE求解器也可以直接用于逆SDE以生成样本。此时，逆SDE有两个特殊属性，可以实现更灵活的采样方法：

- 凭借着时间依赖的分数模型 $s_\theta(\mathbf x, t) $有 $\nabla_{\mathbf x}\log p_t(\mathbf x) $的估计
- 只关心从每个边际分布中采样 $p_t(\mathbf x) $。在任意时间步长获得的样本可以具有任意相关性，并且不必形成从逆向SDE采样的特定轨迹。

由于这个属性，可以用MCMC方法来微调从数值SDE求解器获得的轨迹。

## Probability flow ODE

尽管能够生成高质量样本，但基于Langebin MCMC和SDE求解器的采样器**无法提供分数生成模型计算精确对数似然的方法**。基于**常微分**的采样器可以精确地计算似然。  Song等证明 $t $可以将任何SDE转换为ODE，而不改变其边缘分布 $\{p_t(\mathbf x)\}_{t\in [0, T]} $。因此，通过求解ODE，可以从与逆SDE相同的分布中进行采样。SDE对应的ODE称为**概率流ODE**：

$$
d\mathbf x = \left[f(\mathbf x, t) - \frac{1}{2}g^2(t)\nabla_{\mathbf x}\log p_t(\mathbf x)\right]dt
$$

![](http://qiniu.lianghao.work/teaser.jpg)

当 $\nabla_{\mathbf x}\log p_t(\mathbf x) $被 $s_\theta(\mathbf x, t) $近似时，概率流ODE称为神经ODE的特例。

## Controllable generation for inverse problem solving

基于分数的生成模型特别适合inverse问题。从本质上讲，逆问题与贝叶斯推理相同。 $\mathbf {x,y} $上两个随机变量，假定已知从 $\mathbf x $生成 $\mathbf y $的**前向过程**，由转移概率分布 $p(\mathbf y|\mathbf x) $表示。**逆问题则是 $p(\mathbf x|\mathbf y) $**。根据贝叶斯法则，我们有

$$
p(\mathbf x|\mathbf y)=\frac{p(\mathbf x)p(\mathbf y|\mathbf x)}{\int p(\mathbf{y|x})p(\mathbf x)d\mathbf x}
$$

通过**取梯度**可以大大简化这个表达式，对于得分函数遵循该贝叶斯规则：

$$
\nabla_{\mathbf x}\log p(\mathbf{x|y}) = \nabla_{\mathbf x}\log p(\mathbf x) + \nabla_{\mathbf x}\log p(\mathbf y|\mathbf x)
$$

通过**分数匹配**，可以训练一个模型来估计**无条件数据分布**的分布函数，即 $s_\theta(\mathbf x)\approx \nabla_{\mathbf x}\log p(\mathbf x) $。通过上述公式，使得我们能够从已知的前向过程 $p(\mathbf{y|x}) $中轻松地计算**后验分布函数** $\nabla_{\mathbf x}\log p(\mathbf{x|y}) $，并且使用Langevin类采样方法从中进行采样。



**基于分数的生成模型**和**扩散概率模型**都可以被视为**由分数函数确定的随机微分方程的离散化**。扩散概率模型的采样方法可以与基于分数模型的退化Langevin Dynamics相结合·，以创建统一更强大的采样器（预测器、校正器和采样器）

总的来说，这些最新进展似乎表明，具有多重噪声扰动的基于分数的生成模型和扩散概率模型都是同一模型族的不同视角。分数匹配和基于分数的模型的视角允许精确计算对数似然，自然地解决逆问题，并且与基于能量的模型、薛定谔桥和最优传输直接相关。扩散模型的观点与 VAE、有损压缩自然相关，并且可以直接与变分概率推理相结合。

## Reference

[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)

[Sliced Score Matching: A Scalable Approach to Density and Score Estimations

[https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://yang-song.net/blog/2019/ssm/)
