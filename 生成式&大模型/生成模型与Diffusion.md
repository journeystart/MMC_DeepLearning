# 生成模型与Diffusion

[toc]

## 1. 什么是生成模型？

​	*机器学习上，有监督的模型可以分成生成式和判别式；

​	*一般理解：能产生数据集以外的的数据（造假）

## 2. 生成与采样

​	*我们将需要生成的数据建模成一个分布**p(x)** ;

​	*生成过程<==>从**p(x)**采样的过程。

### 3. 计算机如何采样

### (1) 均匀分布：

> ### 一、真随机数生成器（TRNG）
>
> 利用物理世界的随机现象生成，具有不可预测性：
>
> 1. **物理熵源**：
>    - 电子元件的热噪声
>    - 量子效应（如光子行为）
>    - 大气噪声
>    - 用户输入时间间隔
> 2. **硬件设备**：
>    - 英特尔芯片的 `RDRAND` 指令（基于电路热噪声）
>    - 专用硬件随机数生成器（如量子随机数生成器）
> 3. **特点**：
>    - 随机性来自物理过程，理论上不可预测
>    - 生成速度较慢，适合对安全性要求高的场景（如加密密钥生成）
>
> ------
>
> ### 二、伪随机数生成器（PRNG）
>
> 通过确定性算法生成**看似随机**的数列，是最常用的方法：
>
> 1. **核心原理**：
>
>    - 初始值（种子） + 确定性算法 → 生成数列
>    - 相同种子必然产生相同序列
>
> 2. **常见算法**：
>
>    - **线性同余生成器（LCG）**：简单高效，但随机性有限
>
>      ```
>      // 示例公式：X_{n+1} = (a * X_n + c) mod m
>      ```
>
>    - **梅森旋转算法**（如 `MT19937`）：周期长，均匀性好，用于Python、C++等
>
>    - **密码学安全PRNG**：如AES-CTR、ChaCha20（用于加密场景）

### (2) 正态分布

> ##### Box-Muller变换
>
> Box-Muller变换是通过服从均匀分布的随机变量，来构建服从高斯分布的随机变量的一种方法。具体的描述为：选取两个服从[0, 1]上均匀分布的随机变$U_1$、$U_2$, $X$、$Y$​满足
> $$
> X = cos(2πU_1 )−2lnU_2
> $$
>
> $$
> Y = sin(2 π U1)−2lnU_2
> $$
>
> 则$X$、$Y$服从均值为0，方差为1的高斯分布。

### (3) 其他分布

我们获取正态分布可以不用专门构造生成算法，可以利用已有的简单分布（均匀分布）映射到正态分布。这也是现有生成模型一般结构：
$$
简单分布(隐空间采样)\rightarrow数据分布(数据生成)
$$

## 4. 常见的几大生成模型

· [VAE](##5.-VAE)

· GAN

· AR

· Flow

· Diffusion

博客：[5种生成模型（VAE、GAN、AR、Flow 和 Diffusion）的对比梳理 + 易懂讲解 + 代码实现_vae、回归、gan、diffusion-CSDN博客](https://blog.csdn.net/weixin_57972634/article/details/146497448)

![img](https://i-blog.csdnimg.cn/direct/0a18e6653e7045109d47c92d5a01e01c.png)

![image-20251211161940494](.\assets\image-20251211161940494.png)

## 5. VAE

### (1) AE 到 VAE

![image-20251211152122578](.\assets\image-20251211152122578.png)

如上图所示，假设有两张训练图片，一张是全月图，一张是半月图，经过训练我们的自编码器模型已经能无损地还原这两张图片。接下来，我们在code空间上，两张图片的编码点中间处取一点，然后将这一点交给解码器，我们希望新的生成图片是一张清晰的图片（类似3/4全月的样子）。但是，实际的结果是，生成图片是模糊且无法辨认的乱码图。一个比较合理的解释是，因为编码和解码的过程使用了深度神经网络，这是一个非线性的变换过程，所以在code空间上点与点之间的迁移是非常没有规律的。

​    如何解决这个问题呢？我们可以引入噪声，使得图片的编码区域得到扩大，从而掩盖掉失真的空白编码点。

![image-20251211152144426](.\assets\image-20251211152144426.png)

 如上图所示，现在在给两张图片编码的时候加上一点噪音，使得每张图片的编码点出现在绿色箭头所示范围内，于是在训练模型的时候，绿色箭头范围内的点都有可能被采样到，这样解码器在训练时会把绿色范围内的点都尽可能还原成和原图相似的图片。然后我们可以关注之前那个失真点，现在它处于全月图和半月图编码的交界上，于是解码器希望它既要尽量相似于全月图，又要尽量相似于半月图，于是它的还原结果就是两种图的折中（3/4全月图）。

​    由此我们发现，给编码器增添一些噪音，可以有效覆盖失真区域。不过这还并不充分，因为在上图的距离训练区域很远的黄色点处，它依然不会被覆盖到，仍是个失真点。为了解决这个问题，我们可以试图把噪音无限拉长，使得对于每一个样本，它的编码会覆盖整个编码空间，不过我们得保证，在原编码附近编码的概率最高，离原编码点越远，编码概率越低。在这种情况下，图像的编码就由原先离散的编码点变成了一条连续的编码分布曲线，如下图所示。

![image-20251211152251087](.\assets\image-20251211152251087.png)

那么上述的这种将图像编码由离散变为连续的方法，就是变分自编码的核心思想。

### (2) VAE 原理

### (3) 对VAE生成能力的理解

![image-20251211161319463](.\assets\image-20251211161319463.png)

（ps: 图上五角星是VAE的编码的隐空间编码，即均值，周围的颜色表示的是正态分布，颜色越深，和原始图像越像。可以这么理解，正态分布在均值附近采样的概率比较大，也就是会说被采样的次数比较多，即对loss的贡献比较大，为了使整体loss下降，概率大的地方要和原始图像越相近）

假设训练数据x1、x2、x3对应的隐空间编码为z1、z2、z3，假设初始的编码空间状态如上图所示，其中x1和x3是相似的数据（比如都是满月），而z2是和他们比较不相似的数据（比如半月）。z_sample1为z1的采样点，受z2的影响会导致数据和z1比较不相似，即loss较大。因此训练过程中为是loss下降，**倾向于将特征相似的数据聚拢到一起。**如z_sample2，当相似数据聚拢到一起后，因为x1和x3是相似的数据，所以z_sample2生成的数据也会和二者相似，从而使Loss变小。

**这个过程可以理解为模型正在学习数据分布的特征**。生成任务也可以看成是数据集里数据特征的一个加权。



## 6. GAN

![img](https://i-blog.csdnimg.cn/blog_migrate/c0b3c588cb7164bf6eb6227715226fb1.jpeg)

>  [生成对抗网络](https://so.csdn.net/so/search?q=生成对抗网络&spm=1001.2101.3001.7020)(GAN, Generative adversarial network)自从2014年被Ian Goodfellow提出以来，掀起来了一股研究热潮。GAN由生成器和判别器组成，生成器负责生成样本，判别器负责判断生成器生成的样本是否为真。生成器要尽可能迷惑判别器，而判别器要尽可能区分生成器生成的样本和真实样本。
>
>  在GAN的原作[1]中，作者将生成器比喻为印假钞票的犯罪分子，判别器则类比为警察。犯罪分子努力让钞票看起来逼真，警察则不断提升对于假钞的辨识能力。二者互相[博弈](https://so.csdn.net/so/search?q=博弈&spm=1001.2101.3001.7020)，随着时间的进行，都会越来越强。那么类比于图像生成任务，生成器不断生成尽可能逼真的假图像。判别器则判断图像是否是真实的图像，还是生成的图像，二者不断博弈优化。最终生成器生成的图像使得判别器完全无法判别真假

GAN由两部分组成：**生成器（Generator）**和 **判别器（Discriminator）**

- **生成器**的任务是生成尽可能接近真实数据的假数据
- **判别器**的任务是区分输入数据是真实数据还是生成器生成的假数据
- 二者通过相互**竞争与对抗**，共同进化，最终生成器能够生成非常**接近真实数据**的样本

当生成器生成的数据越来越真时，判别器为维持住自己的准确性，就必须向判别能力越来越强的方向迭代。当判别器越来越强大时，生成器为了降低判别器的判断准确性，就必须生成越来越真的数据。在这个奇妙的关系中，判别器与生成器同时训练、相互内卷，对损失函数的影响此消彼长，这是真正的零和博弈。

损失函数的形式如下：

![img](https://pic3.zhimg.com/v2-e3f87117db545bd9f53f714563677a38_1440w.png)

**可以证明优化上述目标等价于最小化$P_{data}(x)$ 与$P_G(x)$之间的JS散度**

> ### 1. 固定G，求解令损失函数最大的D
>
> 这对应了第一个原则——**对于判别器来说，尽可能找出生成器生成的数据与真实数据分布之间的差异**
>
> 判别器D的输入 有两部分：
>
> - 一部分是真实数据，设其分布为 
> - 另一部分是生成器生成的数据，参考架构图，生成器接收的数据z服从分布 ，输入z经过生成器的计算生成的数据分布设为 
>
> 这两部分都是判别器D的输入，不同的是，G的输出来自分布 ，而真实数据来自分布 ，所以推导如下：
>
> ![img](https://pic1.zhimg.com/v2-46f58f20d0a34fc9e559fdeebb1f6318_1440w.jpg)
>
> 由于这是D的一元函数，要求最优的D值，所以对D求导，得：
>
> ![img](https://pic2.zhimg.com/v2-3bcd15b232aa9f8ed6b6a2f23e6bdc37_1440w.png)
>
> 令导数为0，得：
>
> ![img](https://pic3.zhimg.com/v2-4b25e3181842b711346c49efdfae596a_1440w.png)
>
> 由于判别器D的输入不是来自真实数据，就是来自生成数据，所以：
>
> ![img](https://pica.zhimg.com/v2-7dcdc44d6b2f5cc7ee452b0d31bf382c_1440w.png)
>
> 则，
>
> ![img](https://pica.zhimg.com/v2-2ba08740a1d3b030ecf604d9b2206a36_1440w.jpg)
>
> 可以看出，固定G，将最优的D带入后，此时 ，也就是 ，实际上是在度量 和 之间的[JS散度](https://zhida.zhihu.com/search?content_id=227890470&content_type=Article&match_order=1&q=JS散度&zhida_source=entity)，同KL散度一样，他们之间的分布差异越大，JS散度值也越大。换句话说：**保持G不变，最大化V(G,D)就等价于计算JS散度！！！，**现在回过头看2.1构造原则中的第一条是不是就更加理解了，**对于判别器来说，尽可能找出生成器生成的数据与真实数据分布之间的差异，这个差异就是JS散度。**
>
> ### **2 固定D，求解令损失函数最小的G**
>
> 这对应了第二个原则——**对于生成器来说，让生成器生成的数据分布接近真实数据分布。**
>
> 现在第一步已经求出了最优解的 ，代入损失函数：
>
> ![img](https://picx.zhimg.com/v2-a140c3d64e5bf17050c53761ef052469_1440w.png)
>
> 可以看出，这一步就是在最小化JS散度，JS散度越小，分部之间的差异越小，正好印证了第二个原则，多么完美的推导，前后逻辑自洽！！！！

具体计算分为两步：

1. **固定G，求损失最大的D，实际上就是在求**$P_{data}(x)$ **与$P_G(x)$** **之间的JS散度，找到分布差异的度量。**
2. **固定D，最小化第一步的结果，就是最小化JS散度，让分布接近。**

**训练过程**具体就是生成器和判别器相互交替，分别更新。

## 7. AR

![image-20251211165807238](.\assets\image-20251211165807238.png)

算法原理：自回归模型是一种基于序列数据的生成模型，它通过预测序列中下一个元素的值来生成数据。给定一个序列$(x_1, x_2, ..., x_n)$，自回归模型试图学习条件概率分布$P(x_t | x_{t-1}, ..., x_1)$，其中$t$表示序列的当前位置。AR模型可以通过循环神经网络（RNN）或 Transformer 等结构实现。

**损失函数：**

![img](https://i-blog.csdnimg.cn/direct/f8035c8c26604eb4a32d43d0884f7e49.png)

如下以 **Transformer** 为例解析。

在深度学习的早期阶段，卷积神经网络（CNN）在图像识别和自然语言处理领域取得了显著的成功。然而，随着任务复杂度的增加，序列到序列（Seq2Seq）模型和循环神经网络（RNN）成为处理序列数据的常用方法。尽管RNN及其变体在某些任务上表现良好，但它们在处理长序列时容易遇到梯度消失和模型退化问题。为了解决这些问题，Transformer模型被提出。而后的GPT、Bert等大模型都是基于Transformer实现了卓越的性能！

![image-20251211170159732](.\assets\image-20251211170159732.png)

## 8. Flow

算法原理：流模型是一种基于可逆变换的深度生成模型。它通过一系列可逆的变换，将简单分布（如均匀分布或正态分布）转换为复杂的数据分布。

![img](https://i-blog.csdnimg.cn/direct/cfcb41535f4a4861b415555a3b6bbb08.png)

核心思想：用 “可逆魔法” 转换分布 流模型就像一个 “数据变形大师”，它的核心是可逆变换。想象你有一团标准形状的橡皮泥（简单分布，如正态分布），通过一系列可逆向操作的手法（比如拉伸、折叠，但随时能恢复原状），把它捏成跟真实数据（如图像、语音）分布一样复杂的形状。这种 “既能变形，又能变回去” 的特性，就是流模型的关键 —— 通过可逆函数，让简单分布 “流动” 成复杂数据分布。

生成过程类比：假设真实数据是 “猫咪图片” 的分布，流模型先从简单的正态分布中采样一个向量z（像随机选一块标准形状的橡皮泥），然后通过生成器G的一系列可逆变换（比如调整颜色、轮廓等操作），把z变成一张猫咪图片x。因为变换可逆，未来也能通过反向操作，从猫咪图片还原出最初的z。

博客：

[流模型 Flow 超详解，基于 Flow 的生成式模型，从思路到基础到公式推导到模型理解与应用（Flow-based Generative Model）_flow模型-CSDN博客](https://blog.csdn.net/m0_56942491/article/details/136346491)

![img](https://i-blog.csdnimg.cn/blog_migrate/4b6a8625ad1f825af92b3e75b3fc185c.png#pic_center)

### 对于一维来说：

假设有两个概率密度函数$p(x)$、$\pi (z)$，存在一个严格单调的函数$f$使得$x=f(z)$，对于区间$[z',z'+\Delta z]$，有对应的区间$[x', x'+\Delta x](其中x'=f(z'), x'+\Delta x = f(z'+\Delta z))$, 则
$$
P(z'\leq z\leq z'+\Delta z) = P(x' \leq x \leq x'+\Delta x)
$$
当$\Delta z$很小的时候，$[z',z'+\Delta z]$和$[x', x'+\Delta x]$可以看成均匀分布，即
$$
\pi (z')\Delta z = p(x') \Delta x 
$$

$$
p(x') = \pi(z') \frac{\Delta z}{\Delta x}
$$

可以写成微分形式，考虑到微分可正可负，我们需要加上绝对值符号：
$$
p(x') = \pi(z')|{\frac{dz}{dx}}|
$$

$$
p(x') = \pi(z')|{\frac{1}{f'(z)}}|=\pi(z')|{h'(x)}|,\\ where \quad z=h(x)=f^{-1}(x)
$$

### 对于二维来说：

![img](https://i-blog.csdnimg.cn/blog_migrate/95b517df47c35bc2ffa1dc94795ccfb6.png#pic_center)

![image-20251211204141312](.\assets\image-20251211204141312.png)

> ps：二维行列式可以表示面积，三维可以表示体积

推导可得：
$$
p(x')|det(J_f)| = \pi (z')\\p(x') = \pi (z')|\frac{1}{det(J_f)}|=\pi (z')|{det(J_{f^{-1}})}|
$$
其中$J_f$为*f*的雅可比矩阵

> **如果两个函数互逆，则它们的雅可比矩阵也互逆。**
> $$
> J_fJ_{f^{-1}}=I
> $$
> **又因为互逆的矩阵的行列式的值互为倒数**，所以
>
> ![image-20251211204638184](.\assets\image-20251211204638184.png)

**上述公式可以推广到多维情况，这就是flow核心公式：**

![img](https://i-blog.csdnimg.cn/blog_migrate/071579e62512c90d0ee6d21dfdac4a2b.png#pic_center)

对于生成器G，我们优化的目标
$$
\begin{equation}
\begin{aligned}
logp(x) &= log(\pi (z)|det(J_{G^{-1}})|)\\
		&= log(\pi(G^{-1}(x))) + log(|det(J_{G^{-1}})|)
\end{aligned}
\end{equation}
$$
由此便得到了我们最终需要最大化的式子，也就是目标函数。不过，要想优化这样的目标函数是有一定的前提的，需要对 G 进行一些限制，使其能满足以下条件：

1. **可以计算 $det(J_G)$：**我们知道了生成器 G，理论上知道了 z 怎么变成 x 就很容易计算其雅可比矩阵的行列式。然而，在现实场景中，G 的输入和输出一般维度都非常高，例如 z 和 x 都是1000维向量，那么$ J_G$ 将是一个 1000*1000 的矩阵，**计算它的行列式值是非常耗费时间的。**
2. **知道 G^-1：**式子中，也有 G^-1 的存在，我们也需要好好设计 G，让其可逆、逆可以计算、方便计算。因此为了让 G 可逆，在 Flow 模型中，输入和输出的维度一般都是一样的。

有了上述的限制，G 就不再是随便一个网络都能胜任的了，这么多限制也使得 G 的能力一定是很有限的。
一个 G 的能力是有限的，那两个 G 呢？三个 G 呢？千千万万个 G 呢？“水流细” 没关系，只要细水长流，日积月累就能至千里。

![img](https://i-blog.csdnimg.cn/blog_migrate/58d18a5e87fc43620e1600430cfe387d.png#pic_center)

这时候有：

![image-20251211210014369](.\assets\image-20251211210014369.png)

![image-20251211210408785](.\assets\image-20251211210408785.png)

## 9. Diffusion

### (1) 初识Diffusion

> ##### 博客
>
> [(45 封私信 / 80 条消息) Diffusion model｜扩散模型的历史 - 知乎](https://zhuanlan.zhihu.com/p/672700039)
>
> 

扩散模型最早由文章**[Deep unsupervised learning using nonequilibrium thermodynamics](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1503.03585)**在2015年提出，**其目的是消除对训练图像连续应用的高斯噪声**，可以将其视为一系列**去噪自编码器**。

在2020年提出的**[Denoising Diffusion Probabilistic Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2006.11239)** 让用扩散模型进行图像生成开始变成主流。大家通常说的扩散模型也就是这个DDPM。

![image-20251213170819787](.\assets\image-20251213170819787.png)

扩散模型的灵感来自非平衡热力学（non-equilibrium thermodynamics），也就是扩散过程，这是一种自然现象，描述了粒子在介质中从高浓度区域向低浓度区域的移动，即**熵增**、信息被破坏的过程。

而作者做的就是，通过定义了一个**扩散步骤的马尔可夫链**，以缓慢地将随机噪声添加到数据中，然后学习反转扩散过程以从噪声中构建所需的数据样本。

> 前向过程（扩散过程），从image到noise；
>
> 反向过程（生成过程，采样过程），从noise到image。

![img](https://picx.zhimg.com/v2-aa12884256a9c6a72f56b32ff66779e1_1440w.png)

![image-20251213170843851](.\assets\image-20251213170843851.png)

· 基础理论

> 1. [DDPM (Denoising Diffusion Probabilistic Models)](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)//Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.
> 2. [SMLP (Score Matching with Langevin Dynamics)](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf)//Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.
> 3. [SGM (Score-Based Generative Modeling)](https://arxiv.org/pdf/2011.13456)//Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.
> 4. [FM (Flow Matching)](https://arxiv.org/pdf/2210.02747)//Lipman Y, Chen R T Q, Ben-Hamu H, et al. Flow matching for generative modeling[J]. arXiv preprint arXiv:2210.02747, 2022.

如今提到扩散模型主要理解为以2020年DDPM为开端及其后续改进的模型。但其实2019年宋飏率先提出了以得分为基础的生成扩散模型SMLP。两者虽然在细节上有些许不同，不过都是基于**噪声模型**和**逐步逼近真实数据分布**的思想建立的，并且宋飏在2021年从一个更高的视角——随机微分方程（Stochastic Differential Equation, SDE）将DDPM和SMLD两者进行了统一。（ps，SGM也涉及常微分方程(ODE)）

扩散模型在生成建模领域取得了显著成功，能够生成高质量的图像、视频和音频内容。然而，这类模型存在一个关键局限性：生成过程需要执行数百个去噪步骤，导致推理效率极低。而Flow Matching的核心思想是通过求解常微分方程(Ordinary Differential Equations, ODE)来学习数据生成过程，而非通过逆向扩散过程。

> 扩散模型： 定义一个从数据到噪声的「加噪」过程，然后学习一个「去噪」过程来反转它。去噪过程通常被建模为一个随机微分方程（SDE） 或常微分方程（ODE）。
>
> 流匹配（Flow Matching）：学习一个从噪声点流动到数据点的速度场，这个过程也被建模为一个 ODE。

·改进

> 1. [DDIM (DENOISING DIFFUSION IMPLICIT MODELS)](https://arxiv.org/pdf/2010.02502)//Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.
> 2. [LDM (Latent Diffusion Model)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)//Rombach R, Blattmann A, Lorenz D, et al. High-resolution image synthesis with latent diffusion models[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 10684-10695.

**DDIM（去噪扩散隐式模型）** 是一种针对扩散模型的关键**加速采样技术**。它突破了原始扩散模型必须依赖数百步随机噪声去除才能生成样本的限制，其核心思想是**将随机性的扩散过程重新定义为确定性的生成过程**。通过巧妙地重新参数化扩散模型的采样方程，DDIM实现了**一种确定性映射**：它直接建模从初始噪声到生成图像的隐式轨迹，从而允许在仅需**数十步甚至十步以内**的极小采样步数下，快速生成高质量图像。这项工作的深远意义在于，它揭示了扩散模型与基于常微分方程的**概率流**之间的深刻联系，极大地提升了扩散模型的实用性，并为后续更高效的生成式方法（如流匹配）奠定了基础。

**LDM（潜扩散模型）** 是扩散模型在**高维数据生成领域取得突破性应用**的核心架构。它的核心创新在于将计算密集的扩散过程从高维的像素空间（如图像空间）转移到低维、高效的**潜空间**中进行。具体而言，LDM首先利用一个预训练的编码器（如VAE或Autoencoder）将图像压缩为潜表示，然后在这个潜空间内执行扩散与去噪过程，最后再用解码器将潜表示重建为图像。这一转变**将计算复杂度降低了数十倍**，使得在有限资源下训练和生成高分辨率图像成为可能。更重要的是，它在潜空间的扩散模型中引入了**交叉注意力机制**，能够无缝整合文本、布局等条件信息，从而实现了强大、高效且**可控的条件图像生成**。**Stable Diffusion** 便是LDM最著名的代表，它彻底推动了文生图等AIGC技术的普及与应用。

### (2) DDPM

![image-20251215090114677](.\assets\image-20251215090114677.png)

#### A. 前向过程（加噪）

##### i. 单步

$$
x_t\textasciitilde q(x_t|x_{t-1}) = N(x_t|\sqrt{1-\beta_t}x_{t-1},\sqrt\beta_tI)
$$

令$\alpha_t = 1-\beta_t$,
$$
x_t\textasciitilde q(x_t|x_{t-1}) = N(x_t|\sqrt{\alpha_t}x_{t-1},\sqrt{1-\alpha_t}I)
\\
% 整个公式只有一个编号
\begin{equation}
x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t,\epsilon_t \textasciitilde N(0,I)
\end{equation}
$$

> 这里的系数$\beta_t$是单步加噪方差的大小，可以是通过学习得到，也可以设置成预设的参数。原文$\beta_t$是通过预设得到的，取值从0.02到0.0001。这里还要说明一件事
>
> ，就说均值的系数为什么设置成$\sqrt{1-\beta_t}$，假设$x_{t-1}$也是个标准正态分布，那这样得到$x_t$方差$(\sqrt{1-\beta_t})^2+\beta_t = 1$，可以保证加噪过程方差稳定。但也有博主认为是以下原因：
>
> ![image-20251215092545800](.\assets\image-20251215092545800.png)
>
> ![image-20251215092514097](.\assets\image-20251215092514097.png)

##### ii. 多步

当我们要求t比较大的$x_t$时，一步一步加噪会比较麻烦，因此我们以下将来推导多步加噪的公式，即从$x_0$一步到$x_t$.
$$
\begin{align}
x_t &= \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-1})+\sqrt{1-\alpha_t}\epsilon_t\\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\epsilon,\epsilon \textasciitilde N(0,I)\\
&...\\
&= \sqrt{\alpha_t\alpha_{t-1}...\alpha_1}x_{0} + \sqrt{1-\alpha_t\alpha_{t-1}...\alpha_1}\epsilon,\epsilon \textasciitilde N(0,I)
\end{align}
$$

> 两个高斯分布相加还是高斯分布：$aN(\mu_1,\sigma_1^2)+bN(\mu_2,\sigma_2^2) = N(a\mu_1+b\mu_2,a^2\sigma_1^2+b^2\sigma_2^2)$

令$\bar{\alpha}_t=\alpha_t\alpha_{t-1}...\alpha_1$，则
$$
x_t = \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon,\epsilon \textasciitilde N(0,I)
$$

#### B. 后向过程（去噪）

我们的后向过程就是从白噪声通过$q(x_{t-1}|x_{t})$一步一步采样得到$x_0$。值得注意的是，当$\beta_t$足够小的时候，$q(x_{t-1}|x_{t})$也是一个高斯分布。

> ![image-20251215160442525](.\assets\image-20251215160442525.png)

但$q(x_{t-1}|x_{t})$不好直接估计，因此我们采用一个网络$p_{\theta}$来估计：
$$
p_{\theta}(x_{0:T}) = p_{\theta}(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_t) \\
p_{\theta}(x_{t-1}|x_t) = N(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t))
$$
文中为了简化设$\Sigma_{\theta}(x_t,t)=\sigma_t^2I$。

![image-20251215100922268](.\assets\image-20251215100922268.png)

> $$
> \begin{align}
> ELBO &= \int_{z\textasciitilde q(z|x)}{q(z|x)log{\frac{p_{\theta}(x,z)}{q(z|x)}}}dz\\
> &= \mathbb{E}_{q(z|x)}[log{\frac{p_{\theta}(x,z)}{q(z|x)}}]
> \end{align}
> $$

整个前向过程是一个后验估计：
$$
q(x_{1:T}|x_0) = \prod_{t=1}^{T}{q(x_t|x_{t-1})}
$$
这里可以把每一步加噪的结果都当成隐空间编码，即$z=x_{1:T}$。因此我们可以从ELBO入手继续推导。
$$
\begin{align}
ELBO &= \mathbb{E}_{q(z|x)}[log{\frac{p_{\theta}(x,z)}{q(z|x)}}]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}}]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})\prod_{t=1}^{T}{p_{\theta}(x_{t-1}|x_{t})}
}{
\prod_{t=1}^{T}{q(x_t|x_{t-1})}
}}]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=2}^{T}{p_{\theta}(x_{t-1}|x_{t})}
}{
q(x_1|x_{0})\prod_{t=2}^{T}{q(x_t|x_{t-1})}
}}]\\
\end{align}
$$
上述式子可以直接推导成蒙特卡洛估计的式子：

> ![image-20251215161523915](.\assets\image-20251215161523915.png)
>
> ![image-20251215161734027](.\assets\image-20251215161734027.png)

此时可以想到$x_0$是确定的，同时因为马尔可夫链的关系，额外增加一项对结果是没有影响的，因此借助下式重写上面的似然优化目标。
$$
\begin{align}
q(x_{t}|x_{t-1}) &= q(x_t|x_{t-1},x_0)\\
&= \frac{q(x_{t-1}|x_{t},x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
\end{align}
$$

$$
\begin{align}
ELBO 

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=2}^{T}{p_{\theta}(x_{t-1}|x_{t})}
}{
q(x_1|x_{0})\prod_{t=2}^{T}{q(x_t|x_{t-1})}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=2}^{T}{p_{\theta}(x_{t-1}|x_{t})}
}{
q(x_1|x_{0})\prod_{t=2}^{T}{q(x_t|x_{t-1},x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})
}{
q(x_1|x_{0})
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_t|x_{t-1},x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})
}{
q(x_1|x_{0})
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
{p_{\theta}(x_{t-1}|x_{t})q(x_{t-1}|x_0)}
}{
{q(x_{t-1}|x_{t},x_0)q(x_t|x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})
}{
q(x_1|x_{0})
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
q(x_{t-1}|x_0)
}{
q(x_t|x_0)
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})
}{
q(x_1|x_{0})
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log
\frac{q(x_{1}|x_0)}{q(x_2|x_0)}
\frac{q(x_{2}|x_0)}{q(x_3|x_0)}
...
\frac{q(x_{T-1}|x_0)}{q(x_T|x_0)}
]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})p_{\theta}(x_{0}|x_{1})
}{
q(x_1|x_{0})
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log
\frac{q(x_{1}|x_0)}{q(x_T|x_0)}
]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\prod_{t=2}^{T}\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]\\

&= \mathbb{E}_{q(x_{1:T}|x_0)}[log
p_{\theta}(x_{0}|x_{1})
]

+
\sum_{t=2}^{T}\mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]
+
\mathbb{E}_{q(x_{1:T}|x_0)}[log{\frac{
p_{\theta}(x_{T})
}{
q(x_T|x_0)
}}]\\

&= \mathbb{E}_{q(x_{1}|x_0)}[log
p_{\theta}(x_{0}|x_{1})
]

+
\sum_{t=2}^{T}\mathbb{E}_{q(x_{t-1}, x_t|x_0)}[log{\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]
+
\mathbb{E}_{q(x_{T}|x_0)}[log{\frac{
p_{\theta}(x_{T})
}{
q(x_T|x_0)
}}]\\

&= \mathbb{E}_{q(x_{1}|x_0)}[log
p_{\theta}(x_{0}|x_{1})
]

+
\sum_{t=2}^{T}\mathbb{E}_{q(x_t|x_0)}\mathbb{E}_{q(x_{t-1}|x_t,x_0)}[log{\frac{
{p_{\theta}(x_{t-1}|x_{t})}
}{
{q(x_{t-1}|x_{t},x_0)}
}}]
+
\mathbb{E}_{q(x_{T}|x_0)}[log{\frac{
p_{\theta}(x_{T})
}{
q(x_T|x_0)
}}]\\
&= \mathbb{E}_{q(x_{1}|x_0)}[log
p_{\theta}(x_{0}|x_{1})
]

-
\sum_{t=2}^{T}\mathbb{E}_{q(x_t|x_0)}[D_{KL}({q(x_{t-1}|x_{t},x_0)}||p_{\theta}(x_{t-1}|x_{t}))]
-
D_{KL}(q(x_T|x_0)||p_{\theta}(x_{T}))
\\
&=L_0+\sum_{t=1}^{T-1}L_t+L_T
\end{align}
$$

1. 对于$L_T$，当T足够大的时候，$x_T$是高斯白噪声，所以$L_T$为常数，可以忽略；
2. 对于$L_t, 1\leq t\leq T-1$，

$$
q(x_{t-1}|x_{t},x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}=\frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

其中
$$
q(x_t|x_{t-1})=N(x_t;\sqrt\alpha_tx_{t-1},(1-\alpha_t)I)\\
q(x_{t-1}|x_{0})=N(x_{t-1};\sqrt{\bar{\alpha}_{t-1}} x_{0},(1-\bar{\alpha}_{t-1})I)\\
q(x_{t}|x_{0})=N(x_{t};\sqrt{\bar{\alpha}_t} x_{0},(1-\bar{\alpha}_t)I)\\
$$
所以
$$
\begin{align}
q(x_{t-1}|x_{t},x_0) &\propto exp\{{-[{
\frac{(x_t-\sqrt\alpha_tx_{t-1})^2}{2(1-\alpha_t)}
+\frac{(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2(1-\bar{\alpha}_{t-1})}
-\frac{(x_{t}-\sqrt{\bar{\alpha}_t} x_{0})^2}{2(1-\bar{\alpha}_{t})}
}]}\}\\
&=exp\{{-[{
(\frac{\alpha_t}{2(1-\alpha_t)}+\frac{1}{2(1-\bar{\alpha}_{t-1})})x_{t-1}^2
+(-\frac{\sqrt{\alpha_t}x_t}{1-\alpha_t}-\frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}})x_{t-1}
+C(x_t,x_0)
}]}\}
\end{align}
$$
因为我们知道高斯分布
$$
N(\mu,\sigma^2)\propto exp[{-\frac{(x-\mu)^2}{2\sigma^2}}]=exp[-(\frac{1}{2\sigma^2}x^2-\frac{\mu}{\sigma^2}x+C)]
$$
所以
$$
\begin{align}
\frac{1}{2\sigma^2} &= \frac{\alpha_t}{2(1-\alpha_t)}+\frac{1}{2(1-\bar{\alpha}_{t-1})}\\
&= \frac{\alpha_t(1-\bar{\alpha}_{t-1})+1-\alpha_t}{2(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\\
&= \frac{\alpha_t-\alpha_t\bar{\alpha}_{t-1}+1-\alpha_t}{2(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\\
&= \frac{1-\alpha_t\bar{\alpha}_{t-1}}{2(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\\
&= \frac{1-\bar{\alpha}_{t}}{2(1-\alpha_t)(1-\bar{\alpha}_{t-1})}
\end{align}
$$

$$
\begin{align} 
-\frac{\mu}{\sigma^2} &= -\frac{\sqrt{\alpha_t}x_t}{1-\alpha_t}-\frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}\\
\end{align}
$$

因此可以求得
$$
\sigma^2 =\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\\
\begin{align}
\mu &= (\frac{\sqrt{\alpha_t}x_t}{1-\alpha_t}+\frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}})\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}}x_0
\end{align}
$$
又因为$x_0$实际上是未知的，所以我们不希望保留它。前面前向过程中我们有这样一个式子
$$
x_t = \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon,\epsilon \textasciitilde N(0,I)
$$
可以得到
$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t- \sqrt{1-\bar{\alpha}_t}\epsilon),\epsilon \textasciitilde N(0,I)
$$
所以
$$
\begin{align}
\mu 
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}}x_0\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}}\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t- \sqrt{1-\bar{\alpha}_t}\epsilon),\epsilon \textasciitilde N(0,I)\\
&= \frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_t}{\sqrt{\alpha_t(1+\bar{\alpha}_t)}}\epsilon,\epsilon \textasciitilde N(0,I)\\
\end{align}
$$
综上，可以求得
$$
q(x_{t-1}|x_{t},x_0)=N(\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_t}{\sqrt{\alpha_t(1+\bar{\alpha}_t)}}\epsilon, \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}})
$$

> 两个高斯函数的KL散度
>
> ![image-20251215213045861](.\assets\image-20251215213045861.png)

因为方差为常数，所以
$$
\begin{align}
\mathop{\arg\min}\limits_{\theta}D_{KL}({q(x_{t-1}|x_{t},x_0)}||p_{\theta}(x_{t-1}|x_{t})) 
&= \mathop{\arg\min}\limits_{\theta}\frac{1}{2\sigma_t^2}||{\mu-\mu_{\theta}(x_t,t)}||_2^2
\end{align}
$$
为了计算方便，我们可以将$\mu_{\theta}(x_t,t)$定义成和$\mu$相同的形式
$$
\mu_{\theta}(x_t,t) = \frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_t}{\sqrt{\alpha_t(1+\bar{\alpha}_t)}}\epsilon_\theta(x_t,t)
$$
所以
$$
\begin{align}
\mathop{\arg\min}\limits_{\theta}D_{KL}({q(x_{t-1}|x_{t},x_0)}||p_{\theta}(x_{t-1}|x_{t})) 
&= \mathop{\arg\min}\limits_{\theta}\frac{1}{2\sigma_t^2}||{\mu-\mu_{\theta}(x_t,t)}||_2^2\\
&= \mathop{\arg\min}\limits_{\theta}\frac{(1-\alpha_t)^2}{2\sigma_t^2\alpha_t(1+\bar{\alpha}_t)}||{\epsilon-\epsilon_\theta(x_t,t)}||_2^2\\
&= \mathop{\arg\min}\limits_{\theta}\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1+\bar{\alpha}_t)}||{\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon,t)}||_2^2
\end{align}
$$
所以
$$
L_t =\mathbb{E}_{q(x_t|x_0)}[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1+\bar{\alpha}_t)}||{\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon,t)}||_2^2]
$$
上式还可以简化为
$$
L_{simple} =\mathbb{E}_{q(x_t|x_0)}[||{\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon,t)}||_2^2]
$$


3. 对于$L_0$

原文说为了确保变分下界是离散数据是无损码长，对其做$L_0$进行特殊处理。

![image-20251215221803134](.\assets\image-20251215221803134.png)

> 但很多时候好像经常不管这个问题，把这一项并入中间项。

训练过程如下

![image-20251215222029676](.\assets\image-20251215222029676.png)

至于方差取多少，文中是这么说的：

![image-20251217150810335](.\assets\image-20251217150810335.png)

![image-20251215222039283](.\assets\image-20251215222039283.png)

![image-20251215222127318](.\assets\image-20251215222127318.png)

### (3) SMLD

#### 1.什么是分数匹配

a. 分数匹配为了解决什么问题

![image-20251216102707613](.\assets\image-20251216102707613.png)

b.如何解决

Score Matching的巧妙之处在于，它避免直接计算 $Z(\boldsymbol{\theta})$，转而通过匹配模型和数据的“得分函数”来估计参数$\boldsymbol{\theta}$。得分函数（score function）定义为对数密度关于数据向量的梯度：
$$
s_{\theta}(x) = \nabla_xp(x;\theta) = \nabla_xq(x,\theta) + \nabla_xZ(\theta) = \nabla_xq(x,\theta)
$$
Score Matching的目标是最小化模型得分函数和数据得分函数之间的期望平方距离：

![image-20251216111101339](.\assets\image-20251216111101339.png)

c. 分数的本质

**对数概率密度函数对于输入数据的梯度**($\frac{\partial log{p(x)}}{\partial x}$ )，全称应该是 [Stein Score](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1602.03253.pdf) ，这就是分数。而我们玩分数模型，就是要**训练它让它学会估计(预测)分数**。

我们知道，数据往往是多维的。由分数的定义以及从数学的角度出发来看，它应当是一个**“矢(向)量场”(vector field)**。既然是向量，那么就有方向，这个方向就是：**对于输入数据(样本)来说，其对数概率密度增长最快的方向**。

![img](https://pic3.zhimg.com/v2-c7f7108866509fe241fcf0b240f4dc60_1440w.jpg)

**如果在采样过程中沿着分数的方向走，就能够走到数据分布的高概率密度区域，最终生成的样本就会符合原数据分布。**

d. score matching 训练方式（求解）

上述训练目标因为中数据的概率密度是未知的，所以不能直接求解，因此需要进行转换。可以证明上述目标可以等价于：

![image-20251216111540045](.\assets\image-20251216111540045.png)

其中$\nabla_xs_{\theta}(x)$是雅可比矩阵。当数据维度比较高的时候，这个式子计算量非常大，因此有人提出了几种简化的方法：**Sliced Score Matching(切片分数匹配)**，**Denoising Score Matching(去噪分数匹配)**。

这里我们就只介绍Denoising Score Matching(去噪分数匹配)

![image-20251216112505239](.\assets\image-20251216112505239.png)

![image-20251216112552345](.\assets\image-20251216112552345.png)

这时$\frac{\partial log q_{\sigma}(\tilde{x}|x)}{\partial \tilde{x}}$可以直接求解
$$
\frac{\partial log q_{\sigma}(\tilde{x}|x)}{\partial \tilde{x}} = \frac{\partial (-\frac{(\tilde{x}-x)^2}{2\sigma^2})}{\partial \tilde{x}}
= -\frac{\tilde{x}-x}{\sigma^2}
$$

#### 2. **What is the Langevin Dynamics?**

Langevin dynamics 本指朗之万动力学方程，它是描述物理学中布朗运动(悬浮在液体/气体中的微小颗粒所做的无规则运动)的微分方程，借鉴到这里作为一种生成样本的方法。

概括地来说，该方法首先从先验分布随机采样一个初始样本，然后利用模型估计出来的分数逐渐将样本向数据分布的高概率密度区域靠近。**为保证生成结果的多样性，我们需要采样过程带有随机性。**正好！布朗运动就是带有随机性的，朗之万动力学方程因此也带有随机项。

![image-20251216102024947](.\assets\image-20251216102024947.png)

#### 3.SMLD 要解决什么问题

1. loss不收敛

流形假设认为，**生活中的真实数据大部分都倾向于分布在低维空间中**，这也是大名鼎鼎的 **“流形学习”(mainfold learning)** 的大前提。也就是说，我们的编码空间(通过神经网络等一系列方法)的维度可能很广泛，但是数据经过编码后在许多维度上都存在冗余，实际上用一部分维度就足以表示它，说明**实质上它仅分布在整个编码空间中的一部分(低维流形)，并没有“占满”整个编码空间。**

首先，**分数这个东西是针对整个编码空间定义的**，然而当我们的数据仅分布在编码空间的低维流形时，分数就会出现“没有定义” (undefined) 的情况。

另外，**score matching 的训练目标(见公式** **) 是基于数据“占满”了整个编码空间这个前提下而推出来的**。当数据仅分布在低维流形中时，score matching 的方法就不适用了。

2. 分数估计不准

除了 loss 不收敛，作者还发现有时会出现模型估计分数不准的问题。这是因为，**属于低概率密度区域的数据**，会由于**没有足够的样本来供模型训练学习**，从而导致这部分的分数估计不准，实质上就是训练不充分。

3. 生成结果偏差较大

当数据分布是由多个分布按一定比例混合而成时，**在朗之万动力学采样的玩法下，即使用真实的分数(对应公式** **)，采样生成出来的结果也可能不能反应出各个分布之间的比例关系。**

![img](https://pic2.zhimg.com/v2-c62ace0813719314b0943cafe00e5831_1440w.jpg)

![image-20251216114028306](.\assets\image-20251216114028306.png)

#### 3.SMLD 怎么解决（提出了NCSN网络）

**方法也就是给数据添加一个扰动。**

1. NCSN 用了高斯噪声去扰动数据，而**高斯噪声是分布在整个编码空间的，因此，扰动后的数据就不会仅“驻扎”在低维流形**，而是能够有机会出现在整个编码空间的任意部分，从而避免了分数(score)没有定义(undefined)的情况，同时使得 score matching适用，这就破解了“loss 不收敛”的困局。
2. 在扰动数据的时候，**尺度较大的噪声会有更多机会将原来的数据“踢”到低概率密度区域，于是原来的低概率密度区域就能获得更多的监督信号**，从而提升了分数估计的准确性，这，破解了“分数估计不准”的困局。
3. 从下图可以看到，**在低概率密度区域，分数几乎是 50% 的机率指向两个分布。**那么**在使用朗之万动力学采样生成时，由于是根据分数来迭代生成的，若初始点落在了这些区域，就会造成有均等机会走向两个分布的方向**，因此生成的整体分布所包含的两个分布的样本数量就会很接近，而理想的是其中一个分布的样本数应该要多一些。而在**加噪扰动后(噪声尺度要稍微大些)，原本“稀疏”的低概率密度区域就能够被“填满”**，并且，自然地，**两个分布中混合比例高的那个分布所占这部分(“填满”后的区域)的比例会更高**

![img](https://pica.zhimg.com/v2-2ef510be793845e0d6a642d0e6e21cee_1440w.jpg)

![img](https://pica.zhimg.com/v2-1dc1258ce1a3e99a53c1d2892988dacc_1440w.jpg)

#### 4.NCSN噪声怎么设计

![image-20251216114605091](.\assets\image-20251216114605091.png)

#### 5.损失函数

![image-20251216115353979](.\assets\image-20251216115353979.png)

![image-20251216115414837](.\assets\image-20251216115414837.png)

其中$\lambda(\sigma_i)取\sigma_i^2$,

#### 6.采样生成：退火朗之万动力学

![image-20251216115619669](.\assets\image-20251216115619669.png)

使用一种称为 **“退火朗之万动力学”(Annealed Langevin Dynamics)** 的方法，这种方法实质上是**在噪声强度递减的情况下使用朗之万动力学采样**，在**每个噪声级别下都有一个朗之万动力学采样生成的过程**。

**首先从最高强度的噪声级别开始**使用朗之万动力学进行一定步数的采样生成，**这个阶段结束后生成的样本会作为下一个噪声级别的初始样本**，然后继续进行朗之万动力学的采样生成过程。重复这种做法，直至在最小的噪声级别下也完成了朗之万动力学的采样生成过程，就得到了最终生成的样本。

这种方式之所以叫“退火”的朗之万动力学，是因为噪声级别不断地减小，而每个噪声级别都在进行朗之万动力学采样，这期间采样生成的样本也不断靠近原数据分布的高概率密度区域，还蛮形象的吧~

### (4) SGM

#### 1. 回顾SMLD和DDPM

##### a. SMLD

对于不同噪声等级$\sigma_{min} = \sigma_1 < \sigma_2 <  ... <  \sigma_N = \sigma_{max}$，我们有

加噪过程
$$
\tilde{x} = x+\sigma_i z, z\textasciitilde N(0,I)\\
p_{\sigma_i}(\tilde{x}|x) = N(\tilde{x};x,\sigma^2I)
$$
优化目标
$$
\begin{align}
\theta^* &= \mathop{\arg\min}\limits_{\theta} \sum \sigma_i^2\mathbb{E}_{p_{data}(x)}\mathbb{E}_{p_{\sigma_i}(\tilde{x}|x)}
[||
{s_{\theta}(\tilde{x},\sigma_i)}-\nabla_{\tilde{x}}p_{\sigma_i}(\tilde{x}|x)
||_2^2]
\end{align}
$$
采样过程
$$
x_i^{m-1} = x_i^{m}+\epsilon_is_{\theta^*}(x_i^m,\sigma_i) + \sqrt{2\epsilon_i}z_i^m, z_i^m \textasciitilde N(0,I)
$$

##### b.DDPM

对于不同噪声强度$0 < \beta_1,\beta_2,...,\beta_N<0$，我们有

加噪过程
$$
x_{i+1} = \sqrt{1-\beta_i} x_i+\sqrt{\beta_i}z_i, z_i\textasciitilde N(0,I)\\
x_{i} = \sqrt{\bar{\alpha}_i}x_0+\sqrt{1-\bar{\alpha}_i}z_i, z_i\textasciitilde N(0,I)
$$
优化目标
$$
\begin{align}
\theta^* &= \mathop{\arg\min}\limits_{\theta} \sum \mathbb{E}_{p_{data}(x_0)}\mathbb{E}_{p_{\sigma_i}(x_i|x_0)}
[||
{z_{\theta}(x_i,i)}-z_i
||_2^2]
\end{align}
$$
采样过程
$$
x_{i-1}=\frac{1}{\sqrt{\alpha_i}}(x_i-\frac{1-\alpha_i}{\sqrt{1-\bar{\alpha}_i}}z_{\theta^*}(x_i,i)) + \sigma_iz_i,z_i\textasciitilde N(0,I)
$$

##### c. SMLD和DDPM等价

$$
\begin{align}
&对于DDPM,因为\\&x_{i} = \sqrt{\bar{\alpha}_i}x_0+\sqrt{1-\bar{\alpha}_i}z_i, z_i\textasciitilde N(0,I)\\&p(x_i|x_0) = N(x_i;\sqrt{\bar{\alpha}_i}x_0,(1-\bar{\alpha}_i)I)\\
&所以\nabla_{x_i}logp(x_i|x_0) = - \frac{x_i-\sqrt{\bar{\alpha}_i}x_0}{1-\bar{\alpha}_i}\\
&又因为 z_i = \frac{x_i-\sqrt{\bar{\alpha}_i}x_0}{\sqrt{1-\bar{\alpha}_i}}\\
&所以优化目标可以等价于\\
&\theta^* = \mathop{\arg\min}\limits_{\theta} \sum (1-\bar{\alpha}_i)\mathbb{E}_{p_{data}(x_0)}\mathbb{E}_{p_{\sigma_i}(x_i|x_0)}
[||\frac{z_{\theta}(x_i,i)}{\sqrt{1-\bar{\alpha}_i}}+\nabla_{x_i}logp(x_i|x_0)||_2^2]\\
&令s_{\theta}(x_i,t)=\frac{z_{\theta}(x_i,i)}{\sqrt{1-\bar{\alpha}_i}},则\\
&\theta^* = \mathop{\arg\min}\limits_{\theta} \sum (1-\bar{\alpha}_i)\mathbb{E}_{p_{data}(x_0)}\mathbb{E}_{p_{\sigma_i}(x_i|x_0)}
[||s_{\theta}(x_i,t)+\nabla_{x_i}logp(x_i|x_0)||_2^2]\\
&采样过程\\
&x_{i-1}=\frac{1}{\sqrt{\alpha_i}}[x_i-(1-\alpha_i)s_{\theta^*}(x_i,t)] + \sigma_iz_i,z_i\textasciitilde N(0,I)\\
&或 x_{i-1}=\frac{1}{\sqrt{1-\beta_i}}(x_i-\beta_is_{\theta^*}(x_i,t)) + \sigma_iz_i,z_i\textasciitilde N(0,I)\\

\end{align}
$$

#### 2. 随机微分方程描述扩散过程

![image-20251217152807884](.\assets\image-20251217152807884.png)

![image-20251217152837013](.\assets\image-20251217152837013.png)

![img](https://ai-studio-static-online.cdn.bcebos.com/5092c98eb9c14ea3aaebf0791f3bed7940361d1537a24b308a14314f7497cd89)

> SMLD和DDPM可以看成上述理论的离散表示形式。

##### a. 随机微分方程的实践（一）-SMLD与VE

1) 迭代公式

$$
x_i = x+\sigma_i^2 \bar{z}_i,\bar{z}_i\textasciitilde N(0,I)\\
x_{i-1} = x+\sigma_i^2 \bar{z}_{i-1},\bar{z}_{i-1}\textasciitilde N(0,I)\\
$$

两式相减可得
$$
x_i = x_{i-1}+\sqrt{\sigma_i^2 - \sigma_{i-1}^2}z_{i-1},{z}_{i-1}\textasciitilde N(0,I)
$$

2. 连续化

$$
\{x_i\}_{i=0}^N \rightarrow x(t),x(\frac{i}{N}) = x_i\\
\{\sigma_i\}_{i=0}^N \rightarrow \sigma(t),\sigma(\frac{i}{N}) = \sigma_i\\
\{z_i\}_{i=0}^N \rightarrow z(t),z(\frac{i}{N}) = z_i\\
$$

$$
\begin{align}
	x(t+\Delta t) &= x(t) + \sqrt{\sigma^2(t+\Delta t)-\sigma^2(t)}z(t)\\
	&= x(t) + \sqrt{\frac{\sigma^2(t+\Delta t)-\sigma^2(t)}{\Delta t}{\Delta t}}z(t)\\
\end{align}\\
$$

$$
x(t+\Delta t) - x(t)=  \sqrt{\frac{\sigma^2(t+\Delta t)-\sigma^2(t)}{\Delta t}{\Delta t}}z(t)
$$

当$\Delta t \rightarrow 0$时，
$$
dx=  \sqrt{\frac{d[\sigma^2(t)]}{dt}}dw
$$

> 其中 $dw = dt z(t),z(t)\textasciitilde N(0,I)$

$$
f(x,t)=0;\\
g(t) = \sqrt{\frac{d[\sigma^2(t)]}{dt}}
$$

3. 写出反向SDE

$$
\begin{align}
dx &= [f(x,t)-g^2(t)\nabla_xlogp_t(x)]dt + g(t)dw\\
&=-\frac{d[\sigma^2(t)]}{dt}\nabla_xlogp_t(x)dt+\sqrt{\frac{d[\sigma^2(t)]}{dt}}dw
\end{align}
$$

4. 再次离散化得到采样公式

$$
\begin{align}
	x(t) - x(t-\Delta t) &= -\frac{\sigma^2(t)-\sigma^2(t-\Delta t)}{\Delta t}\times\Delta t\nabla_xlogp_t(x)
	+\sqrt{\frac{\sigma^2(t)-\sigma^2(t-\Delta t)}{\Delta t}{\Delta t}}z(t)\\
	&= (\sigma^2(t-\Delta t)-\sigma^2(t))\nabla_xlogp_t(x)
	+\sqrt{\sigma^2(t)-\sigma^2(t-\Delta t)}z(t)\\
\end{align}\\
$$

$$
\begin{align}
	x(t-\Delta t)=x(t)+(\sigma^2(t)-\sigma^2(t-\Delta t))\nabla_xlogp_t(x)
	-\sqrt{\sigma^2(t)-\sigma^2(t-\Delta t)}z(t)
\end{align}\\
$$

$$
\begin{align}
	x_{i-1}&=x_i+(\sigma^2_{i}-\sigma^2_{i-1})\nabla_xlogp_t(x)
	-\sqrt{\sigma^2_i-\sigma^2_{i-1}}z_i\\
	&=x_i+(\sigma^2_{i}-\sigma^2_{i-1})\nabla_xlogp_t(x)
	+\sqrt{\sigma^2_i-\sigma^2_{i-1}}z_i\\
\end{align}\\
$$

替换成网络可得
$$
\begin{align}
	x_{i-1}=x_i+(\sigma^2_{i}-\sigma^2_{i-1})s_{\theta^*}(x_i,\sigma_i)
	+\sqrt{\sigma^2_i-\sigma^2_{i-1}}z_i
\end{align}\\
$$

##### b. 随机微分方程的实践（二）-DDPM与VP

1. 迭代公式

$$
x_{i+1} = \sqrt{1-\beta_i} x_i+\sqrt{\beta_i}z_i\\
x_{i+1} = \sqrt{1-\frac{\bar{\beta_i}}{N}} x_i+\sqrt{\frac{\bar{\beta_i}}{N}}z_i
$$

2. 连续化

$$
\{x_i\}_{i=0}^N \rightarrow x(t),x(\frac{i}{N}) = x_i\\
\{\bar{\beta_i}=N\beta_i\}_{i=0}^N \rightarrow \beta(t),\beta(\frac{i}{N}) = \bar{\beta_i}=N\beta_i\\
\{z_i\}_{i=0}^N \rightarrow z(t),z(\frac{i}{N}) = z_i\\
\frac{1}{N} \rightarrow \Delta t
$$

> 引入1/N的目的只要是想引入$\Delta t$, 然后凑出$dtz=dw$

$$
\begin{align}
x(t+\Delta t) &= \sqrt{1-\beta(t)\Delta t}x(t)+\sqrt{\beta(t)\Delta t}z(t)\\
& \approx (1-\frac{1}{2}\beta(t)\Delta t)x(t)+\sqrt{\beta(t)\Delta t}z(t)\\
& = x(t)-\frac{1}{2}\beta(t)\Delta tx(t)+\sqrt{\beta(t)\Delta t}z(t)\\
\end{align}
$$

$$
\begin{align}
x(t+\Delta t) -x(t)
& = -\frac{1}{2}\beta(t)\Delta tx(t)+\sqrt{\beta(t)\Delta t}z(t)\\
\end{align}
$$

> 当x很小时$\sqrt{1-x}\approx1-\frac{1}{2}x$

当$\Delta t \rightarrow 0$时，
$$
dx = -\frac{1}{2}\beta(t)xdt+\sqrt{\beta(t)}dw
$$

$$
f(t,x)=-\frac{1}{2}\beta(t)x\\
g(t)=\sqrt{\beta(t)}
$$

3. 写出反向SDE

$$
\begin{align}
dx &= [f(x,t)-g^2(t)\nabla_xlogp_t(x)]dt + g(t)dw\\
&=[-\frac{1}{2}\beta(t)x-\beta(t)\nabla_xlogp_t(x)]dt+\sqrt{\beta(t)}dw
\end{align}
$$

4. 再次离散化得到采样公式

$$
\begin{align}
x(t) - x(t-\Delta t) &=[-\frac{1}{2}\beta(t)x(t)-\beta(t)\nabla_xlogp_t(x)]\Delta t+\sqrt{\beta(t)\Delta t}z(t)\\

x(t-\Delta t) &=x(t) + [\frac{1}{2}\beta(t)x(t)+\beta(t)\nabla_xlogp_t(x)]\Delta t+\sqrt{\beta(t)\Delta t}z(t)\\
&=(1+\frac{1}{2}\beta(t)\Delta t)x(t) + \beta(t)\Delta t\nabla_xlogp_t(x)+\sqrt{\beta(t)\Delta t}z(t)\\
&=(2-(1-\frac{1}{2}\beta(t)\Delta t)x(t) + \beta(t)\Delta t\nabla_xlogp_t(x)+\sqrt{\beta(t)\Delta t}z(t)\\
&\approx (2-\sqrt{1-\beta(t)\Delta t}x(t) + \beta(t)\Delta t\nabla_xlogp_t(x)+\sqrt{\beta(t)\Delta t}z(t)\\
\end{align}
$$

可得
$$
x_{i-1} = (2-\sqrt{1-\beta_i})x_i + \beta_i\nabla_xlogp_t(x)+\sqrt{\beta_i}z_i
$$
即
$$
x_{i-1} = (2-\sqrt{1-\beta_i})x_i + \beta_is_{\theta^*}(x_i,i)+\sqrt{\beta_i}z_i
$$

##### c. 总结

![image-20251217174757724](.\assets\image-20251217174757724.png)

#### 3. ODE

![image-20251217175958168](.\assets\image-20251217175958168.png)

和SDE有什么区别？

有什么作用？

### (5) FM

### (6) DDIM

### **(7) LDM**