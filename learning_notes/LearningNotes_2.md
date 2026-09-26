# 学习笔记-过程记录2

我的笔记之二，就是我在学习过程中对深度学习的一些名词的理解，具体到激活函数、损失函数的选择，还有各种模型突出的特点与重要的思想。事实上，我的深度学习，大部分都是跟随AI学习的。不得不说，我没有办法在如此之短的时间内总结出这样一篇学习笔记，但利用AI可以做到：它不仅可以给我概念的总结，也可以给我我想要的形式（先说名词再说解释）、还有精确表达的数学公式，流程图等。

见到这么庞大的只是笔记，我是否存在完全AI代劳而本人啥也不做的情况？我的回答是：没有。因为我曾在大一期间就了解过深度学习，并做过相关小项目。比如公开到我的GitHub上的groundingdino，还有未曾公开的yolov8n，因此我具有一定的基础。这份笔记虽然也是AI整理，但我是让他根据与我的聊天记录进行的整理，因此我可以说：绝大部分的内容都是我看过读过的。但我能否具体的记下来并形成体系？不好说，毕竟在这之中的某些知识我也是囫囵吞枣，但我相信我重复的深度学习相关知识越多，零碎的知识点终有一日会穿成一条线，到那时我也能形成自己的知识体系。

此外，代码实现也是我的硬伤。不过相较于大一初次接触深度学习相关内容的时候，我已经有了一定的基础和耐心，起码我在这个项目里面可以耐下心来看过我的MLP的代码。当然其他网络的代码我是囫囵吞枣的，更多的还是了解模型的理论构成等等。因此在这方面，我还有待学习，道阻且长。

# 深度学习知识笔记

学习路线：

**MLP → CNN → AlexNet / VGG / ResNet → U-Net**

---

# 一、神经网络基础与 MLP

**神经网络（Neural Network）：** 神经网络可以理解为一个由大量可学习参数组成的函数，它接收输入 $x$，经过一系列数学变换得到输出 $\hat{y}$。所谓“训练神经网络”，本质上就是先进行前向传播得到预测结果，再用损失函数衡量预测结果与真实答案之间的差距，然后通过反向传播计算各个参数应该如何变化，最后由优化器更新参数。这个过程不断重复，使网络表示的函数逐渐从一个初始的随机映射变成适合当前任务的映射。可以把整个训练过程概括为：

$$
x
\xrightarrow{\text{Forward}}
\hat{y}
\xrightarrow{\text{Loss}}
L
\xrightarrow{\text{Backward}}
\nabla L
\xrightarrow{\text{Optimizer}}
\text{更新参数}
$$

---

**张量（Tensor）：** Tensor 是 PyTorch 中存储和处理数据的基本结构，可以把它理解为能够进行高效数值计算的多维数组。一个数字可以看作 0 维 Tensor，一个向量可以看作 1 维 Tensor，一个矩阵可以看作 2 维 Tensor，而一批图像通常可以表示成更高维 Tensor。在图像任务中常见的数据形状为：

$$
[B,C,H,W]
$$

其中 $B$ 表示 Batch Size，$C$ 表示 Channel，$H$ 和 $W$ 分别表示图像的高度和宽度。Tensor 不仅负责保存数据，还可以记录计算过程以及梯度，因此是 PyTorch 自动求导和神经网络训练的基础。

---

**MNIST：** MNIST 是经典的手写数字图像数据集，每张图片表示 $0\sim9$ 中的一个数字，原始图片大小为 $28\times28$，属于灰度图，因此只有一个通道。每个样本可以看作一对数据：

$$
(x,y)
$$

其中 $x$ 是手写数字图片，$y$ 是对应的数字标签。在 MLP 中，二维的 $28\times28$ 图片通常需要先展平成长度为 784 的一维向量：

$$
28\times28=784
$$

然后才能输入全连接网络。

---

**数据预处理（Data Preprocessing）：** 数据预处理是模型训练之前对原始数据进行必要变换的过程，例如把图片转换为 Tensor、调整数据范围、归一化或改变尺寸。预处理的目的不是让模型直接“知道答案”，而是把原始数据转换成更适合神经网络处理的形式，并尽量使训练过程更加稳定。需要注意训练集、验证集和测试集的处理规则应保持合理一致，同时不能通过预处理把测试集答案等信息泄露给训练过程。

---

**归一化（Normalization）：** 归一化是对输入数据的数值范围或分布进行调整，使不同样本的数据尺度更加统一，从而让网络训练更加稳定。图像原始像素通常可以表示为 $0\sim255$，转换为 Tensor 后常先缩放到 $0\sim1$；还可以进一步按照均值 $\mu$ 和标准差 $\sigma$ 进行标准化：

$$
x'=\frac{x-\mu}{\sigma}
$$

这样可以避免输入数值尺度过大或分布不合理对优化过程造成影响。这里的数据归一化与后面网络内部使用的 Batch Normalization 不是同一个概念。

---

**Dataset：** Dataset 用来描述“数据集里有什么以及如何取得一个样本”。在 PyTorch 中，一个 Dataset 通常能够根据索引返回一个样本及其标签，例如：

$$
(x_i,y_i)
$$

它解决的是数据如何组织和读取的问题。MNIST 本身就可以通过 PyTorch 提供的数据集接口读取，而在更复杂的任务中，也可以自己编写 Dataset 来定义图片路径、标签读取和预处理方式。

---

**DataLoader：** DataLoader 建立在 Dataset 之上，负责按照 Batch 将样本取出，并提供打乱数据、并行加载等功能。Dataset 更关注“一个样本怎么得到”，DataLoader 更关注“训练时怎样一批一批地把样本交给模型”。如果 Batch Size 为 64，那么一次迭代通常会得到 64 个样本，而不是一次只训练一张图片。

---

**Batch / Batch Size：** Batch 是一次送入网络进行前向传播和反向传播的一组样本，Batch Size 表示这一组中包含多少个样本。例如 Batch Size 为 64，就表示模型一次使用 64 个样本计算预测结果和 Loss，再根据这一批数据得到的梯度更新参数。Batch 太小，梯度波动通常更明显；Batch 太大则会占用更多显存，因此它既是训练超参数，也会影响训练效率和优化行为。

---

**Epoch：** 一个 Epoch 表示模型完整地遍历一次训练集。例如训练集有 55,000 个样本，Batch Size 为 64，那么一个 Epoch 中会包含许多个 Batch；每处理一个 Batch 通常都会进行一次参数更新。训练多个 Epoch 的目的，就是让模型多次观察训练数据并不断调整参数，但 Epoch 并不是越多越好，训练过久可能产生过拟合。

---

**训练集（Training Set）：** 训练集是直接用于更新模型参数的数据。模型对训练集执行前向传播、计算 Loss、反向传播并由优化器修改参数，因此模型会直接从训练集中学习。

**验证集（Validation Set）：** 验证集不用于直接更新参数，而是在训练过程中检查模型在未参与当前参数更新的数据上的表现，可以用来观察模型是否正在改善、是否出现过拟合以及选择较好的模型或超参数。验证集虽然不参与梯度更新，但如果反复根据验证结果选择模型和调整方案，就已经间接参与了开发过程。

**测试集（Test Set）：** 测试集主要用于模型开发完成后的最终评估，不应该参与模型训练、参数更新和超参数选择。保持测试集独立，可以让最终测试结果更接近模型对真正未知数据的泛化能力。

---

**数据泄露（Data Leakage）：** 数据泄露指本来不应该在训练阶段被模型或开发流程利用的信息进入了训练过程，例如把测试集样本混入训练集、根据测试集表现反复选择模型，或者在划分数据之前以不恰当方式利用了全体数据的信息。数据泄露会让测试结果虚高，使最终准确率不能真实反映模型面对未知数据时的能力，因此训练集、验证集和测试集必须保持合理隔离。

---

**参数（Parameter）：** 参数是神经网络通过训练自动学习的数值，最典型的是 Weight 和 Bias。网络刚创建时，这些参数通常经过随机或特定规则初始化，因此未训练模型的输出往往没有实际意义；训练过程通过反向传播和优化器不断修改这些参数，使网络输出逐渐接近正确结果。神经网络所谓“学到了东西”，最终就是这些可学习参数发生了有意义的变化。

---

**权重（Weight）：** Weight 决定输入中的不同信息对下一层输出产生多大的影响。以最基本的线性层为例：

$$
z=Wx+b
$$

其中 $x$ 是输入，$W$ 就是权重矩阵。训练过程中，网络会根据 Loss 对 $W$ 的梯度不断调整它，使有用的输入特征获得合适的影响程度。

---

**偏置（Bias）：** Bias 是在线性变换中额外加入的可学习参数：

$$
z=Wx+b
$$

其中 $b$ 就是偏置。它使神经元的输出不必被限制为完全由输入与权重乘积决定，相当于允许线性关系进行平移，从而提高模型表示不同函数的灵活性。Bias 与 Weight 一样，会通过反向传播得到梯度并由优化器更新。

---

**神经元（Neuron）：** 神经元可以理解为神经网络中的一个基本计算单元，它接收上一层的一组输入，对这些输入进行加权求和并加入 Bias，之后通常再经过激活函数。一个简单神经元可以表示为：

$$
z=\sum_{i=1}^{n}w_i x_i+b
$$

再经过激活函数：

$$
a=f(z)
$$

单个神经元的能力有限，但大量神经元按照多层结构组合起来后，就可以表示复杂的函数关系。

---

**全连接层（Fully Connected Layer / Linear Layer）：** 全连接层表示当前层的每个输出神经元都与上一层的所有输入相连接，其核心计算为：

$$
y=Wx+b
$$

在 PyTorch 中通常使用 `nn.Linear` 实现。全连接层能够对输入特征进行重新组合，是 MLP 的核心组成部分；但对于图像来说，它没有显式利用像素之间的二维空间关系，这也是后面 CNN 要解决的重要问题。

---

**Flatten（展平）：** Flatten 是把多维数据按照一定顺序转换成一维特征向量的操作。MNIST 图片原本是 $28\times28$ 的二维空间结构，在输入 MLP 前可以变为：

$$
28\times28\rightarrow784
$$

这样才能作为 784 维输入送入全连接层。Flatten 本身不会学习任何参数，只负责改变数据形状。它使 MLP 能处理图片，但同时也弱化了“哪些像素原本彼此相邻”这样的空间结构信息，因此对于图像任务存在明显局限。

---

**MLP（Multi-Layer Perceptron，多层感知机）：** MLP 是由多个全连接层和非线性激活函数组成的前馈神经网络。它将输入经过一层层线性变换和非线性变换后得到最终输出。在本项目的 MNIST 任务中，$28\times28$ 图片先被 Flatten 成 784 维向量，然后经过隐藏层，最后输出 10 个数，对应数字 $0\sim9$ 的分类信息。项目中的基本结构可以概括为：

$$
784\rightarrow128\rightarrow10
$$

MLP 建立了最基本的神经网络训练思路：输入数据经过网络得到输出，再通过 Loss、Backward 和 Optimizer 不断调整参数。但 MLP 将图片展平后处理，没有专门利用图像中局部像素之间的空间关联，这也是从 MLP 进一步学习 CNN 的主要原因。

---

**隐藏层（Hidden Layer）：** 隐藏层指位于输入层和输出层之间的网络层。之所以叫“隐藏”，不是因为它无法观察，而是因为它既不是原始输入，也不是最终任务输出。隐藏层会逐步把原始输入转换成更适合完成任务的内部特征表示。例如 `784 → 128 → 10` 中，128 维这一层就属于隐藏层。

---

**激活函数（Activation Function）：** 激活函数通常作用在线性层或卷积层的输出之后，最重要的作用是给神经网络引入非线性。如果多层网络只有线性变换：

$$
y=W_2(W_1x+b_1)+b_2
$$

那么展开后仍然可以写成：

$$
y=W'x+b'
$$

也就是说，无论堆叠多少个纯线性层，本质上仍然只是一个线性变换。加入非线性激活函数以后，多层网络才有能力表示复杂的非线性关系。因此“增加网络层数”要真正产生更强的表达能力，通常必须配合非线性激活。

---

**ReLU（Rectified Linear Unit）：** ReLU 是常用的激活函数，其数学表达式为：

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

也就是当 $x>0$ 时保留原值，当 $x\leq0$ 时输出 0。ReLU 计算简单，同时为网络引入非线性，因此被广泛应用于深度神经网络。与早期常用的 Sigmoid、Tanh 相比，ReLU 在正数区域的梯度不会随着输入增大而趋近于 0，因此在深层网络训练中通常更容易优化。不过当某个神经元长期落在负数区域时，梯度可能一直为 0，这也是所谓的 “Dead ReLU” 问题。

---

**Sigmoid：** Sigmoid 是经典的非线性激活函数，它会把任意实数输入压缩到 $0\sim1$ 之间，其数学表达式为：

$$
\operatorname{Sigmoid}(x)=\frac{1}{1+e^{-x}}
$$

当 $x$ 很大时，Sigmoid 的输出逐渐接近 1；当 $x$ 很小时，输出逐渐接近 0；当 $x=0$ 时，输出为 0.5。由于输出范围类似概率，Sigmoid 常用于二分类模型的输出处理中，也常用于一些需要控制信息通过比例的网络结构。但 Sigmoid 在输入绝对值较大时曲线会趋于平坦，此时梯度接近 0，反向传播经过很多层后容易出现梯度消失（Vanishing Gradient），使深层网络前面的参数难以有效更新。因此在现代深层网络的隐藏层中，通常更多使用 ReLU 等激活函数，而 Sigmoid 更多出现在具有特定含义的输出或门控结构中。

---

**Tanh（Hyperbolic Tangent，双曲正切）：** Tanh 也是经典的非线性激活函数，它会把任意实数输入压缩到 $-1\sim1$ 之间，其数学表达式为：

$$
\operatorname{Tanh}(x)
=
\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

当 $x$ 很大时，Tanh 输出逐渐接近 1；当 $x$ 很小时，输出逐渐接近 -1；当 $x=0$ 时，输出为 0。与 Sigmoid 的 $0\sim1$ 输出不同，Tanh 的输出以 0 为中心，因此在一些情况下更有利于后续层进行优化。Tanh 同样能够为网络引入非线性，但当输入绝对值较大时也会进入饱和区域，此时梯度接近 0，因此同样存在梯度消失问题。它曾广泛用于传统神经网络和循环神经网络中，但在许多现代深层 CNN 的隐藏层中通常也更常使用 ReLU。

---

**前向传播（Forward / Forward Propagation）：** 前向传播指输入数据按照网络定义的方向从输入层逐层计算到输出层的过程。假设一个简单 MLP 为：

$$
x\rightarrow Linear_1\rightarrow ReLU\rightarrow Linear_2\rightarrow z
$$

那么可以写成：

$$
h=\operatorname{ReLU}(W_1x+b_1)
$$

$$
z=W_2h+b_2
$$

前向传播只回答“按照当前参数，这个输入会得到什么输出”。训练开始时参数还没有学好，因此结果可能很差；随着参数不断更新，同样的前向计算会逐渐产生更合理的结果。

---

**Logits：** Logits 是分类网络最后一层直接输出、尚未转换成概率的一组实数。例如 MNIST 有 10 个类别，因此模型对一张图片通常输出 10 个 Logits：

$$
z=[z_0,z_1,\dots,z_9]
$$

每个值对应一个类别的相对得分。Logit 可以为正数、负数，也不要求总和为 1。在 PyTorch 中使用 `CrossEntropyLoss` 时，一般应该直接把原始 Logits 传入损失函数，而不是提前手动进行 Softmax。

---

**Softmax：** Softmax 可以把一组 Logits 转换为总和为 1 的概率分布。对于第 $i$ 个类别：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

其中 $z_i$ 是第 $i$ 个类别的 Logit，$K$ 是类别数量。Softmax 不只是简单地把数值缩放到 $0\sim1$，而是通过指数运算突出较大的 Logit，并使所有类别概率满足：

$$
\sum_{i=1}^{K}p_i=1
$$

在推理阶段可以用它理解模型对各类别的相对置信程度；但使用 PyTorch 的 `CrossEntropyLoss` 训练多分类模型时通常不需要自己先调用 Softmax，因为损失函数内部已经完成了相应计算。

---

**损失函数（Loss Function）：** 损失函数用于把模型预测结果与真实答案之间的差距转换成一个可以优化的数值 $L$。前向传播告诉我们“模型输出了什么”，Loss 则告诉我们“这个输出有多不好”。训练的目标通常可以写成：

$$
\min_{\theta}L(\theta)
$$

其中 $\theta$ 表示模型所有可学习参数。Loss 本身不会自动修改模型参数，它首先提供优化目标，然后反向传播根据这个目标计算梯度，最后由优化器真正更新参数。

---

**L1 Loss（Mean Absolute Error，平均绝对误差）：** L1 Loss 使用预测值与真实值之间差的绝对值衡量误差。对于 $N$ 个元素，其常见形式为：

$$
L_{\mathrm{L1}}
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值。L1 对误差大小呈线性惩罚，即误差扩大两倍，对 Loss 的贡献也大致扩大两倍，因此相比平方误差，它不容易被少量特别大的误差完全主导，对异常值相对更鲁棒。在图像重建任务中，L1 可以直接约束输出图片与目标图片之间的像素差异，并且通常比单独使用 MSE 更不容易产生过度平滑的结果，因此在图像到图像任务中十分常见。本项目 Level 4 的组合损失中也使用了 Global L1 来约束整体图像重建效果。

---

**L2 Loss：** L2 Loss 根据预测值和真实值之间误差的平方进行惩罚，常见的平方形式可以写为：

$$
L_{\mathrm{L2}}
=
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

平方操作使较大的误差受到更强的惩罚，例如误差从 1 增加到 2 时，对平方误差的贡献会从 1 增加到 4，因此 L2 对大误差比 L1 更敏感。需要注意，“L2 Loss”这个名称在不同资料中可能表示平方 L2 范数，也可能表示欧氏距离本身，因此实际阅读代码时应该确认具体定义。神经网络中常见的 MSE 就可以看作对平方误差进行平均的一种形式。

---

**MSE Loss（Mean Squared Error，均方误差）：** MSE 计算预测值与真实值之间误差平方的平均值：

$$
L_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

MSE 本质上关注预测值与目标值之间的数值距离，因为使用平方惩罚，所以较大的预测误差会对最终 Loss 产生更大的影响。它常用于回归和图像重建等连续值预测任务。在图像任务中，可以直接比较预测图片与 Target 对应像素之间的差异。本项目 Level 4 同时使用了 Global MSE 和 Changed-Region MSE，使模型既关注整张图片的重建，也更重视发生变化区域中的较大误差。

---

**L1 Loss 与 MSE 的区别：** 两者都可以衡量预测值和真实值之间的差距，但惩罚误差的方式不同。若单个位置的误差为：

$$
e=y-\hat{y}
$$

那么 L1 关注：

$$
|e|
$$

而 MSE 关注：

$$
e^2
$$

因此 L1 对误差进行线性惩罚，对异常的大误差相对不那么敏感；MSE 对误差进行平方惩罚，会更加重视较大的误差。在图像重建任务中，两者并不存在绝对的“谁更好”，可以根据任务特点进行选择，也可以组合使用，让模型同时获得不同的优化约束。

---

**Cross Entropy Loss（交叉熵损失）：** Cross Entropy 是单标签多分类任务中最常见的损失函数之一，例如 MNIST 中每张图片只属于 $0\sim9$ 中的一个类别。模型首先输出每个类别对应的 Logit，Softmax 可以将这些 Logits 转换成概率。对于真实类别 $y$，如果模型给该类别的概率为 $p_y$，单个样本的交叉熵可以表示为：

$$
L_{\mathrm{CE}}
=
-\log(p_y)
$$

当模型给正确类别较高概率时，Loss 较小；当正确类别概率很低时，Loss 会明显增大，因此训练过程会推动正确类别获得更高的相对得分。在 PyTorch 中，`nn.CrossEntropyLoss` 通常直接接收模型输出的原始 Logits 和整数类别标签，内部已经完成了与 Softmax 对应的计算，因此训练时通常不要自己先对输出调用 Softmax。它主要解决“多个类别中选择一个正确类别”的分类问题，也是本项目 MNIST 分类中使用的 Loss。

---

**Binary Cross Entropy Loss（BCE，二元交叉熵）：** BCE 主要用于二分类问题，也可以用于多个标签彼此独立的多标签分类问题。对于真实标签 $y\in\{0,1\}$ 和模型预测为正类的概率 $p$，其基本形式为：

$$
L_{\mathrm{BCE}}
=
-\left[
y\log(p)
+
(1-y)\log(1-p)
\right]
$$

当真实标签 $y=1$ 时，Loss 会推动 $p$ 接近 1；当 $y=0$ 时，则推动 $p$ 接近 0。因此 BCE 可以理解为专门衡量二元概率预测是否正确的损失函数。实际使用 PyTorch 时通常更推荐 `BCEWithLogitsLoss`，让模型直接输出 Logit，并由损失函数内部完成 Sigmoid 和 BCE 的组合，这样数值计算通常更加稳定。

---

**组合损失（Combined Loss）：** 实际任务中，一个 Loss 往往只能强调模型输出的某一个方面，因此可以将多个损失函数按照不同权重组合起来：

$$
L
=
\lambda_1L_1
+
\lambda_2L_2
+
\cdots
+
\lambda_kL_k
$$

其中 $\lambda_i$ 用于控制不同损失项的重要程度。组合 Loss 的意义不是简单地认为“Loss 越多越好”，而是让不同损失项分别约束任务中的不同目标。本项目 Level 4 最终使用的组合损失可以表示为：

$$
L
=
L_{\mathrm{Global\ L1}}
+
0.5L_{\mathrm{Global\ MSE}}
+
2.0L_{\mathrm{Changed\ Region\ MSE}}
$$

其中 Global L1 和 Global MSE 约束整张输出图片与 Target 的差异，而 Changed-Region MSE 进一步提高发生变化区域在优化目标中的重要程度。这样设计的原因是手写痕迹在整张文档中通常只占一部分区域，如果只计算全局平均误差，大量本来就相似的背景区域可能会削弱模型对真正需要修改区域的关注。

---

**交叉熵损失（Cross Entropy Loss）：** Cross Entropy 是多分类任务中常用的损失函数。若真实类别为 $y$，模型经过 Softmax 后对正确类别给出的概率为 $p_y$，单个样本的交叉熵可以直观写成：

$$
L=-\log(p_y)
$$

如果模型给正确类别很高的概率，那么 $p_y$ 接近 1，此时 Loss 较小；如果正确类别概率很低，那么 Loss 就会增大。因此它会推动网络提高正确类别相对于错误类别的得分。PyTorch 的 `nn.CrossEntropyLoss` 通常直接接收原始 Logits 和整数类别标签，不需要提前手动进行 Softmax。

---

**自动求导（Autograd / Automatic Differentiation）：** 自动求导是 PyTorch 自动记录 Tensor 运算关系并根据计算图计算梯度的机制。前向传播过程中，PyTorch 会记录参与梯度计算的运算；当调用 `loss.backward()` 时，它会根据这些运算关系和链式法则，从 Loss 开始向前反向计算各个参数的梯度。自动求导解决的是“复杂神经网络的导数如何高效、正确地计算”的问题，使我们不需要手动推导并编写每一个参数的偏导数。

---

**计算图（Computational Graph）：** 计算图描述一次前向计算中各个变量和运算之间的依赖关系。例如：

$$
x\rightarrow h\rightarrow z\rightarrow L
$$

表示输入 $x$ 经过若干运算得到隐藏表示 $h$、输出 $z$ 和最终 Loss $L$。反向传播会沿着这张图的反方向，根据链式法则把 Loss 对输出的影响逐层传回参数。PyTorch 的 Autograd 正是利用计算图完成自动求导。

---

**梯度（Gradient）：** 梯度表示 Loss 对某个参数变化的敏感程度。对于参数 $w$：

$$
\frac{\partial L}{\partial w}
$$

描述当 $w$ 发生微小变化时，Loss 会如何变化。梯度不是“参数应该直接变成什么值”，而是为优化器提供局部方向信息。对于包含大量参数的神经网络，可以把所有参数的偏导数组合成梯度：

$$
\nabla_{\theta}L
$$

优化器利用这些梯度决定如何更新模型参数。

---

**链式法则（Chain Rule）：** 神经网络由很多层函数复合而成，因此要计算前面某个参数对最终 Loss 的影响，需要使用链式法则。若：

$$
x\rightarrow h\rightarrow y\rightarrow L
$$

那么：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial h}
\frac{\partial h}{\partial x}
$$

反向传播本质上就是从最终 Loss 出发，利用链式法则把梯度逐层向前传递，因此即使参数位于网络较前的位置，也能知道它对最终误差产生了怎样的影响。

---

**反向传播（Backpropagation / Backward）：** 反向传播是在完成前向传播并得到 Loss 后，从 Loss 开始沿计算图反方向计算各个可学习参数梯度的过程。它解决的不是“直接从答案反推出输入”这个问题，而是计算：

$$
\frac{\partial L}{\partial \theta}
$$

也就是“如果稍微改变某个参数，Loss 会怎样变化”。因此前向传播负责根据当前参数得到结果，反向传播负责根据结果与目标之间的误差计算参数的调整依据。在 PyTorch 中，核心操作通常是：

```python
loss.backward()
```

但 `backward()` 主要负责**计算梯度**，并不负责真正修改参数。

---

**优化器（Optimizer）：** 优化器根据反向传播得到的梯度更新模型参数。可以把训练过程理解为：Loss 给出目标，Backward 计算方向，Optimizer 真正执行参数更新。一个最基本的梯度下降更新可以表示为：

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}L
$$

其中 $\theta_t$ 是当前参数，$\eta$ 是学习率，$\nabla_{\theta}L$ 是当前梯度。PyTorch 中常见优化器包括 SGD 和 Adam。

---

**学习率（Learning Rate）：** 学习率通常记为 $\eta$，决定每次参数更新的步长：

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}L
$$

学习率太大时，参数可能一次跨得太远，使 Loss 剧烈震荡甚至无法收敛；学习率太小时，训练又可能非常缓慢。因此学习率是神经网络训练中非常重要的超参数。

---

**SGD（Stochastic Gradient Descent）：** SGD 是经典优化方法。实际训练时通常不会每次都使用整个训练集计算梯度，而是利用一个 Mini-batch 估计当前梯度，再更新参数：

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}L_{\text{batch}}
$$

由于不同 Batch 得到的梯度存在一定波动，因此它被称为随机梯度下降。SGD 结构简单，是理解神经网络参数更新最直接的优化器之一。

---

**Adam（Adaptive Moment Estimation）：** Adam 是常用的自适应优化器，它不仅使用当前梯度，还维护梯度的一阶矩和二阶矩估计，并根据不同参数的梯度情况自适应调整实际更新幅度。相比最基础的 SGD，Adam 在许多任务中能够较快得到稳定结果，因此工程实践中非常常见。其核心思想可以理解为：SGD 主要根据当前梯度统一迈步，而 Adam 会结合历史梯度的信息，为不同参数调整更新方式。

---

**梯度清零（`optimizer.zero_grad()`）：** PyTorch 默认会把多次反向传播得到的梯度进行累加，因此常规训练时每处理一个新的 Batch，都需要先清除上一次留下的梯度。如果不清零，本轮计算出的梯度会继续叠加到旧梯度上，从而改变原本预期的参数更新过程。典型训练顺序因此通常为：

```python
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

其中 `zero_grad()` 清除旧梯度，Forward 得到输出，Loss 计算误差，`backward()` 计算新梯度，`step()` 最终更新参数。

---

**`nn.Module`：** `nn.Module` 是 PyTorch 中构建神经网络模型的基础类。通常在 `__init__()` 中定义网络包含哪些层，在 `forward()` 中定义数据按照什么顺序流过这些层。这样模型中的可学习参数就能够被 PyTorch 统一管理，并可以方便地进行训练、保存、加载和推理。MLP、CNN、ResNet、U-Net 虽然结构完全不同，但在 PyTorch 中通常都可以作为 `nn.Module` 的子类进行组织。

---

**训练模式（`model.train()`）：** `model.train()` 将模型切换到训练模式。对于普通 Linear 和 ReLU 来说训练模式与评估模式可能没有明显区别，但 Dropout、BatchNorm 等层在训练和推理时行为不同，因此养成训练前调用 `model.train()` 的习惯非常重要。它并不是“调用之后模型就自动训练”，真正的训练仍然需要 Forward、Loss、Backward 和 Optimizer。

---

**评估模式（`model.eval()`）：** `model.eval()` 将模型切换到评估模式，使 Dropout、BatchNorm 等具有训练/推理差异的模块采用推理时的行为。它同样不会自动完成验证或测试，只是在告诉模型“现在处于评估阶段”。验证集、测试集和实际推理时通常应该使用评估模式。

---

**`torch.no_grad()`：** `torch.no_grad()` 表示在其作用范围内不记录用于反向传播的梯度信息。验证和推理阶段不需要更新模型参数，因此通常可以关闭梯度计算，以减少额外的内存占用和计算开销。`model.eval()` 和 `torch.no_grad()` 解决的是两个不同问题：前者改变部分网络层的运行模式，后者关闭梯度记录，因此推理时经常一起使用。

---

**训练循环（Training Loop）：** 训练循环就是不断从 DataLoader 中读取 Batch，并重复“梯度清零 → 前向传播 → 计算 Loss → 反向传播 → 更新参数”的过程。可以概括为：

$$
\text{Batch}
\rightarrow
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Update}
$$

整个训练集遍历一次构成一个 Epoch，再重复多个 Epoch，使模型参数逐渐收敛。训练循环是 MLP、CNN、ResNet、U-Net 等模型共同使用的基本训练框架，后面变化的主要是模型结构、数据形式和 Loss，而这一核心流程基本保持不变。

---

**验证循环（Validation Loop）：** 验证循环使用验证集检查当前模型效果，但不进行 `loss.backward()` 和 `optimizer.step()`，因为验证数据不能用于直接更新模型参数。验证阶段通常会使用 `model.eval()` 和 `torch.no_grad()`，然后计算 Validation Loss、Accuracy 或其他任务指标。训练 Loss 持续下降但 Validation Loss 开始上升，是判断过拟合的重要信号之一。

---

**准确率（Accuracy）：** Accuracy 是分类任务中最直观的评价指标，表示预测正确的样本数占全部样本数的比例：

$$
\operatorname{Accuracy}
=
\frac{\text{预测正确的样本数}}
{\text{总样本数}}
$$

MNIST 各类别相对均衡，因此 Accuracy 能比较直观地反映分类性能。但 Accuracy 与 Loss 并不是同一个概念：Accuracy 只关心最终类别是否预测正确，而 Loss 还会反映模型对类别得分的具体情况，并且 Loss 通常需要具有适合优化的数学性质。

---

**Loss Curve（损失曲线）：** Loss Curve 是记录 Loss 随 Epoch 或 Step 变化得到的曲线，可以用于观察模型是否正在学习、训练是否稳定以及是否可能出现过拟合。例如 Training Loss 和 Validation Loss 同时下降通常说明模型正在有效学习；如果 Training Loss 持续下降而 Validation Loss 明显上升，则可能说明模型越来越贴合训练集，但对未见数据的泛化能力正在下降。

---

**过拟合（Overfitting）：** 过拟合指模型在训练数据上表现越来越好，但对没有参与训练的数据表现没有同步改善甚至变差。可以理解为模型过度适应了训练数据中的具体细节，而没有学到足够具有泛化能力的规律。典型现象是：

$$
L_{\text{train}}\downarrow
\qquad
L_{\text{val}}\uparrow
$$

缓解过拟合的方法包括增加数据、数据增强、正则化、Dropout、控制模型复杂度、提前停止训练等。具体采用哪种方法取决于任务和实验情况。

---

**欠拟合（Underfitting）：** 欠拟合指模型连训练数据本身都没有学好，训练集和验证集表现通常都较差。原因可能包括模型表达能力不足、训练时间不够、学习率不合理等。与过拟合相比，欠拟合更接近“还没有学会”，而过拟合则是“对训练数据学得过于具体”。

---

**泛化能力（Generalization）：** 泛化能力表示模型面对没有参与训练的新数据时仍然能够正确完成任务的能力。训练神经网络的最终目的不是记住训练集，而是从训练数据中学习能够推广到未知数据的规律，因此 Validation Set 和 Test Set 的存在本质上都是为了评价模型的泛化能力。

---

**超参数（Hyperparameter）：** 超参数是训练开始前由开发者设定、而不是模型通过反向传播直接学习得到的配置，例如 Learning Rate、Batch Size、Epoch 数、隐藏层大小等。Weight 和 Bias 属于模型参数，而 Learning Rate 和 Batch Size 属于超参数。超参数会显著影响模型训练速度、稳定性以及最终效果。

---

**推理（Inference）：** 推理指使用已经训练好的模型处理新的输入并得到预测结果，此时只进行前向传播，不再通过反向传播修改模型参数。例如 MNIST 单张图片推理时，需要按照训练时相同或兼容的方式对图片进行预处理，然后送入模型得到 10 个 Logits，再选择得分最高的类别：

$$
\hat{y}=\arg\max_i z_i
$$

因此训练和推理最大的区别之一是：训练需要根据 Loss 更新参数，而推理只使用已经学好的参数进行 Forward。

---

**模型权重保存（Model Weight Saving）：** 模型训练完成后，需要把已经学习到的参数保存下来，否则程序结束后训练结果就无法直接复用。在 PyTorch 中通常保存模型的 `state_dict`，其中记录各层的 Weight、Bias 等参数。保存权重使训练和实际使用可以分离：模型只需要训练一次，之后可以多次加载进行测试和推理。

---

**模型权重加载（Model Weight Loading）：** 加载模型时，需要先创建与训练时结构一致的网络，然后把保存的 `state_dict` 加载进去。网络结构决定“参数应该放在哪里”，权重文件提供“这些参数具体是多少”。因此仅有 `.pth` 权重文件并不等于完整的模型定义，如果网络结构不匹配，参数通常无法正确加载。

---

**Checkpoint：** Checkpoint 是训练过程中保存的模型状态。最简单的 Checkpoint 可以只保存模型权重，也可以进一步保存 Optimizer 状态、当前 Epoch、Loss 等信息。保存 Checkpoint 的意义是保留某个训练阶段的状态，例如保存 Validation Accuracy 最好的模型，或者在长时间训练中断后继续训练。

---

**模型训练的完整逻辑：** 对于一个监督学习分类任务，可以把整个过程统一理解为：Dataset 提供样本 $(x,y)$，DataLoader 将样本组成 Batch，模型接收 $x$ 进行 Forward 得到 Logits，通过 Loss Function 比较 Logits 与真实标签 $y$ 得到损失 $L$，Autograd 根据计算图和链式法则执行 Backward 得到参数梯度，Optimizer 根据梯度更新 Weight 和 Bias；训练过程中使用 Validation Set 检查泛化情况，训练完成后在独立 Test Set 上进行最终评价，保存得到的模型权重，最后加载模型对新图片执行 Inference。这个过程不仅适用于 MLP，也是后面 CNN、AlexNet、VGG、ResNet 乃至 U-Net 训练流程的基础。

---

**MLP 的局限：** MLP 可以完成 MNIST 分类，但它首先需要把二维图片 Flatten 成一维向量，这意味着网络本身并没有显式利用“相邻像素通常具有更强关联”这一图像先验。例如图片中一个局部笔画在空间上移动后，在 Flatten 后会对应完全不同的一组输入位置，全连接层并不会天然把它们视为同一种局部特征。同时，全连接层的参数量会随着输入尺寸快速增长。为了解决图像空间结构利用不足的问题，下一步需要引入专门针对局部空间信息设计的 **卷积神经网络 CNN**。

---


训练的本质可以进一步压缩成：

$$
\boxed{
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Update}
}
$$

MLP 已经建立了神经网络训练的基本框架，但对于图像而言，它没有充分利用二维空间中相邻像素之间的关系，因此引入了 CNN。


# 二、CNN

**CNN（Convolutional Neural Network，卷积神经网络）：** CNN 是主要用于处理图像等具有空间结构数据的神经网络。与 MLP 一开始就把图片 Flatten 成一维向量不同，CNN 会保持图像的二维空间结构，并通过卷积操作优先处理局部区域。它利用了图像中一个重要特点：相邻像素之间通常具有较强关联，例如一条边缘、一个笔画或者一个纹理都是由附近的一组像素共同构成的。CNN 通过局部连接和权重共享，让同一个特征检测器可以在图片不同位置寻找相似的局部模式，因此相比直接使用全连接层，更符合图像本身的结构特点。CNN 通常由 Convolution、Activation、Pooling 等结构逐步提取特征，最后再通过全连接层或其他输出结构完成具体任务。

---

**卷积（Convolution）：** 卷积是 CNN 最核心的操作，它使用一个较小的卷积核在输入图像或 Feature Map 上滑动，每移动到一个位置，就将卷积核与当前位置覆盖的局部数据进行对应元素相乘并求和，从而得到输出 Feature Map 中的一个值。以二维单通道情况为例，可以粗略表示为：

$$
Y(i,j)
=
\sum_m\sum_n
X(i+m,j+n)K(m,n)
$$

其中 $X$ 表示输入，$K$ 表示卷积核，$Y$ 表示输出 Feature Map。实际深度学习框架中的卷积通常还会同时处理多个 Channel 并加入 Bias。卷积最重要的意义并不只是完成一种数学运算，而是让网络能够从局部区域中提取特征，同时保持图片的空间关系。

---

**卷积核（Kernel / Filter）：** 卷积核是 CNN 中用于提取局部特征的一组可学习参数，例如常见的 $3\times3$ 卷积核。卷积核会在输入图像或上一层 Feature Map 上滑动，在每个局部区域执行加权计算，从而产生新的 Feature Map。可以把一个卷积核理解成一个通过训练自动学习出来的“局部特征检测器”：浅层卷积可能逐渐对边缘、方向、简单纹理等局部模式产生响应，更深层则可以在已有特征基础上组合出更加复杂的模式。卷积核的数值并不是人为提前规定的，而是和 MLP 中的 Weight 一样，通过反向传播和优化器自动学习得到。

---

**卷积核大小（Kernel Size）：** Kernel Size 表示卷积核在空间上的尺寸，例如 $3\times3$、$5\times5$。更大的卷积核一次能够观察更大的局部区域，因此单层具有更大的感受范围，但同时需要更多参数和计算；较小卷积核一次观察的范围更小，但可以通过连续堆叠多层卷积逐渐扩大感受野，并在中间加入更多非线性变换。现代 CNN 中 $3\times3$ 卷积非常常见，例如连续两个 $3\times3$ 卷积就能够逐渐获得接近 $5\times5$ 的有效感受范围，同时可以在两层之间加入激活函数。卷积核大小属于重要超参数，不存在对所有任务都最好的固定选择，需要根据输入大小、任务和网络结构综合决定。

---

**局部连接（Local Connectivity）：** CNN 中一个输出位置通常只与输入中的一个局部区域直接相连，而不像全连接层那样与所有输入建立连接。例如 $3\times3$ 卷积在计算某个位置时，只直接观察周围 $3\times3$ 的区域。这种局部连接来自图像本身的特点：很多视觉特征首先表现为局部模式，例如边缘、笔画和纹理，因此没有必要让最浅层的每个神经元一开始就同时观察整张图片。局部连接既利用了图像的局部空间关系，也减少了不必要的连接。

---

**权重共享（Weight Sharing）：** 权重共享指同一个卷积核在图片的不同空间位置重复使用同一组参数。例如一个 $3\times3$ 卷积核有一组已经学习到的参数，当它从图片左上角滑动到右下角时，卷积核本身的参数不会因为位置变化而重新定义。这样一个卷积核如果学会检测某种局部模式，就可以在整张图片的不同位置寻找这种模式。相比 MLP 中不同输入位置通常拥有独立连接权重，权重共享显著减少了参数数量，同时让 CNN 更适合识别可能出现在不同位置的相似局部特征。

---

**通道（Channel）：** Channel 表示数据在同一个空间位置上具有多少组不同的信息。灰度 MNIST 图片通常只有 1 个输入 Channel，而 RGB 彩色图片通常有 3 个输入 Channel，分别对应红、绿、蓝三个颜色分量。在 CNN 的中间层中，Channel 不再简单代表颜色，而可以理解为网络学习出的不同类型特征。比如某一层输出 32 Channels，就表示这一层在每个空间位置上产生了 32 组不同的特征响应。因此随着网络加深，经常会看到空间尺寸 $H\times W$ 逐渐减小，而 Channel 数逐渐增加，让网络从较大的原始空间信息逐步转变为更加丰富的特征表示。

---

**输入通道（Input Channels）与输出通道（Output Channels）：** 一个卷积层通常需要指定输入通道数和输出通道数。例如：

```python
nn.Conv2d(1, 32, kernel_size=3)
```

可以理解为：

```text
1 个输入 Channel
        ↓
32 个不同的可学习卷积核组
        ↓
32 个输出 Channel
```

对于多通道输入，一个输出通道对应的卷积核实际上会覆盖所有输入通道，然后把各输入通道的卷积结果组合起来形成一个输出 Feature Map。因此如果输入通道数为 $C_{\text{in}}$、输出通道数为 $C_{\text{out}}$、Kernel Size 为 $K_h\times K_w$，普通卷积层的 Weight 参数量为：

$$
C_{\text{out}}
\times
C_{\text{in}}
\times
K_h
\times
K_w
$$

如果每个输出通道还有一个 Bias，则总参数量为：

$$
C_{\text{out}}
\times
C_{\text{in}}
\times
K_h
\times
K_w
+
C_{\text{out}}
$$

---

**特征图（Feature Map）：** Feature Map 是输入经过卷积核处理后得到的特征表示。它仍然保留空间维度，因此某个位置的数值可以表示网络在图片对应区域检测到某种特征的响应程度。如果一个卷积层输出 32 Channels，那么可以理解为产生了 32 张 Feature Maps，每一张由不同的可学习卷积核生成，用来表达不同类型的特征。随着网络不断加深，Feature Map 会从接近原始像素的信息逐渐转化为更加抽象、更加适合最终任务的表示。

---

**Stride（步幅）：** Stride 表示卷积核每次在输入上移动多少个像素。当 Stride 为 1 时，卷积核每次移动一个位置；当 Stride 为 2 时，每次跨过两个位置，因此输出空间尺寸会明显减小。对于一维尺寸，可以用下面的公式计算卷积后的输出大小：

$$
H_{\text{out}}
=
\left\lfloor
\frac{H_{\text{in}}+2P-K}{S}
\right\rfloor
+1
$$

其中 $H_{\text{in}}$ 是输入尺寸，$K$ 是 Kernel Size，$P$ 是 Padding，$S$ 是 Stride。Stride 不仅决定卷积核如何移动，也会影响 Feature Map 的尺寸、计算量以及保留空间信息的多少。

---

**Padding（填充）：** Padding 指在输入 Feature Map 的边缘额外补充像素，最常见的是补 0。卷积核在图片边缘时没有足够的邻域，如果完全不进行 Padding，Feature Map 每经过卷积都会缩小，而且边缘区域参与卷积的次数更少。通过适当 Padding 可以控制输出空间尺寸。例如对于 Kernel Size 为 3、Stride 为 1 的卷积，使用 Padding 为 1 时：

$$
H_{\text{out}}=H_{\text{in}}
$$

因此可以在完成卷积的同时保持 Feature Map 的高度和宽度不变。Padding 的意义主要是控制空间尺寸，并使边缘信息能够更充分地参与卷积计算。

---

**感受野（Receptive Field）：** 感受野表示网络中某个特征点在原始输入上能够受到多大区域的影响。单个 $3\times3$ 卷积层只能直接观察局部 $3\times3$ 区域，但随着卷积层不断堆叠，后面的神经元会间接整合越来越大的输入区域，因此感受野会逐渐扩大。例如两个连续的 $3\times3$、Stride 为 1 的卷积，在不考虑边界影响时可以获得约 $5\times5$ 的有效感受范围。CNN 正是通过这种方式从局部特征逐渐组合出更大范围、更复杂的特征。

---

**池化（Pooling）：** Pooling 是对 Feature Map 的局部区域进行聚合，从而降低空间尺寸的操作。与卷积不同，常见 Pooling 通常没有需要训练的 Weight。池化可以减少后续计算量，同时让后面的特征对应更大的原图区域，但下采样也会丢失一部分精确的空间信息。因此 Pooling 并不是“越多越好”，特别是在需要精确恢复图像空间结构的任务中，过度下采样会带来问题，这一点在后面的 U-Net 中尤其重要。

---

**最大池化（Max Pooling）：** Max Pooling 在一个局部窗口中只保留最大的数值。例如对于 $2\times2$ 区域：

$$
\begin{bmatrix}
1 & 5\\
2 & 3
\end{bmatrix}
\rightarrow
5
$$

如果使用 `2×2` Max Pooling 且 Stride 为 2，那么 Feature Map 的高度和宽度通常都会减半：

$$
H\times W
\rightarrow
\frac{H}{2}\times\frac{W}{2}
$$

可以把它理解为在缩小 Feature Map 的同时保留局部区域中响应最明显的特征。CNN 中常通过 Pooling 逐渐降低空间分辨率、扩大后续特征的有效感受范围并减少计算量。

---

**下采样（Downsampling）：** 下采样泛指降低 Feature Map 空间分辨率的操作，例如：

$$
28\times28
\rightarrow
14\times14
\rightarrow
7\times7
$$

Max Pooling 和 Stride 大于 1 的卷积都可以实现下采样。下采样可以减少计算量、扩大后续神经元对应的感受范围，并让网络逐渐从精细的空间信息转向更抽象的特征。但它也会丢失一部分位置信息，因此分类任务和图像生成/分割任务对下采样的处理方式会有所不同。

---

**卷积层参数量：** CNN 的一个重要特点是卷积层的参数数量主要由 Kernel Size、输入 Channel 和输出 Channel 决定，而不是直接由图片的高度和宽度决定。普通二维卷积层的参数量为：

$$
\text{Parameters}
=
C_{\text{out}}
\times
C_{\text{in}}
\times
K_h
\times
K_w
+
C_{\text{out}}
$$

例如一个 `1 → 32`、Kernel Size 为 $3\times3$ 的卷积层，如果包含 Bias：

$$
32\times1\times3\times3+32
=
320
$$

这说明卷积层可以用相对少量的共享参数扫描整张图片。不过“CNN 一定比 MLP 参数更少”并不成立，因为完整 CNN 还可能包含较大的全连接层，实际参数量必须根据整个网络结构计算。在本项目中，Level 2 CNN 的总参数量实际上高于 Level 1 MLP，因此应该把“卷积层具有参数共享优势”和“整个模型最终参数量是多少”区分开。

---

**卷积后的尺寸变化：** CNN 中必须随时关注 Tensor Shape。对于普通二维卷积，一个方向上的输出尺寸为：

$$
H_{\text{out}}
=
\left\lfloor
\frac{H_{\text{in}}+2P-K}{S}
\right\rfloor+1
$$

宽度同理。例如输入为 $28\times28$，使用 $3\times3$ 卷积、Padding 为 1、Stride 为 1：

$$
\frac{28+2\times1-3}{1}+1=28
$$

因此空间尺寸仍然为：

$$
28\times28
$$

如果随后使用 `2×2`、Stride 为 2 的 Max Pooling，则变成：

$$
14\times14
$$

理解尺寸变化非常重要，因为后面的卷积层、Flatten 和 Fully Connected Layer 都依赖上一层输出的 Shape。

---

**CNN 中的 Flatten：** CNN 前面的卷积层负责提取具有空间结构的 Feature Maps，而传统分类 CNN 在进入全连接分类器之前，通常仍然需要把多通道 Feature Maps 展平成一维向量。例如：

$$
64\times7\times7
\rightarrow
3136
$$

再送入 Linear Layer。这里的 Flatten 与 MLP 开头直接 Flatten 图片不同：MLP 在提取特征之前就丢掉了显式二维结构，而 CNN 是先经过多层卷积充分利用空间结构提取特征，最后才为了分类器进行 Flatten。

---

**CNN 中的全连接层（Fully Connected Layer）：** 在传统 CNN 分类网络中，卷积部分负责提取图像特征，而后面的 Fully Connected Layer 根据这些特征完成最终分类。因此可以粗略理解为：

$$
\text{Image}
\rightarrow
\text{Convolutional Feature Extraction}
\rightarrow
\text{Flatten}
\rightarrow
\text{Fully Connected Classifier}
\rightarrow
\text{Logits}
$$

CNN 并不是完全抛弃 MLP，而是把“直接使用原始像素分类”改成“先通过卷积提取更适合图像的特征，再完成分类”。

---

**特征提取（Feature Extraction）：** 特征提取指网络把原始像素逐步转换成更适合任务的内部表示。CNN 不需要人为告诉模型“这里是横线”“这里是圆弧”，而是通过训练自动调整卷积核，使能够降低最终 Loss 的局部模式被逐渐保留下来。浅层通常更直接地响应局部低级模式，深层则在浅层 Feature Maps 的基础上组合更加复杂的特征。因此深度学习中的“特征”通常不是人工固定设计的，而是模型根据任务从数据中学习得到的。

---

**平移等变性（Translation Equivariance）：** 卷积具有一定的平移等变性质，可以粗略理解为：如果输入中的某个局部模式发生位置移动，那么由同一个卷积核检测到的响应也会相应移动，而不需要为图片每一个位置重新学习一套完全不同的参数。这来自卷积的局部连接和权重共享。它使 CNN 比普通全连接网络更自然地处理“同一种特征可能出现在图片不同位置”的情况。需要注意，CNN 并不是天然对任意平移完全不变，Pooling、Stride、边界处理以及后续网络结构都会影响最终行为。

---

**CNN 相比 MLP 的核心优势：** MLP 把图片 Flatten 后，把每个像素主要当作向量中的独立位置处理，而 CNN 明确利用了图像的二维空间结构。CNN 通过局部连接利用相邻像素之间的关联，通过权重共享让同一个特征检测器能够在不同位置重复使用，通过多层卷积逐渐从局部简单特征组合出更复杂的特征。因此 CNN 对图像具有更合适的结构先验。在本项目 MNIST 实验中，Level 1 MLP 的 Test Accuracy 为 96.36%，Level 2 CNN 为 98.70%，提高了 2.34 个百分点，这个实验结果与 CNN 更适合提取图像局部空间特征的特点是一致的，但不能因此简单认为 CNN 在所有数据集和所有配置下都一定优于 MLP。

---

**CNN 的超参数：** CNN 中需要人为设定而不是通过反向传播直接学习的配置包括 Kernel Size、Stride、Padding、输出 Channel 数、Pooling 方式、Learning Rate、Batch Size 等。这些超参数会影响模型的感受野、Feature Map 尺寸、参数量、计算量和最终性能。例如更大的 Kernel 可以一次观察更大范围，但增加计算；更多 Channels 可以提供更丰富的特征表示，但也增加参数和计算；更大的 Stride 可以快速降低空间尺寸，但可能损失更多细节。因此分析 CNN 不能只看最终 Accuracy，还应该结合网络结构和计算代价。

---

**收敛（Convergence）：** 收敛描述训练过程中模型的 Loss 和性能逐渐趋于稳定的现象。比较 MLP 和 CNN 时，“收敛速度”不能只理解成总训练时间，因为一个 Epoch 的计算量可能不同；还可以观察经过多少 Epoch 或 Step 后 Validation Loss、Validation Accuracy 达到较稳定水平。更复杂的网络可能在较少 Epoch 内达到较高 Accuracy，但由于单次 Forward 和 Backward 计算更多，实际运行时间仍然更长，因此 Epoch、Step 和实际时间应该区分。

---

**错误样本（Misclassified Samples）：** 错误样本是分类模型预测错误的输入，例如真实数字是 5，但模型预测成 3。只看 Accuracy 只能知道模型“错了多少”，观察错误样本则可以进一步分析“模型在什么情况下容易错”。例如字迹模糊、数字形状相似、书写方式特殊等都可能导致错误。错误样本分析能够帮助判断模型真正的弱点，而不是只依赖一个总体准确率数字。

---

**CNN 的基本数据流：** 一个典型的 CNN 分类模型可以概括为：

$$
\text{Image}
\rightarrow
\text{Conv}
\rightarrow
\text{ReLU}
\rightarrow
\text{Pooling}
\rightarrow
\text{Conv}
\rightarrow
\text{ReLU}
\rightarrow
\text{Pooling}
\rightarrow
\text{Flatten}
\rightarrow
\text{FC}
\rightarrow
\text{Logits}
$$

其中 Conv 负责提取局部特征，ReLU 引入非线性，Pooling 逐步降低空间尺寸，后面的 Fully Connected Layer 根据已经提取出的特征完成分类。训练方式与 MLP 并没有本质变化，仍然是：

$$
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Optimizer}
$$

真正发生变化的是网络处理图像和提取特征的方式。

---

**从 CNN 到更深的 CNN：** 基础 CNN 已经解决了 MLP 没有充分利用图像空间结构的问题，但随着任务变复杂，仅使用少量卷积层能够提取的特征仍然有限。一个自然思路就是继续增加网络深度和特征通道，让模型学习更加复杂的层次化特征。但网络变深之后，又会出现训练成本增加、过拟合、梯度传播困难以及深层网络优化困难等问题。因此 CNN 的下一步发展并不是简单地“无限增加 Conv”，而是开始研究怎样更合理地设计深层网络结构，由此产生了 AlexNet、VGG、ResNet 等经典架构。



# 三、AlexNet、VGG 与 ResNet

**深层卷积神经网络（Deep CNN）：** 基础 CNN 已经能够利用局部连接和权重共享提取图像特征，而增加网络深度可以让模型进行更多层次的特征变换：前面的卷积层学习较局部、较简单的特征，后面的卷积层在这些特征基础上继续组合，形成更复杂、更抽象的表示。因此 AlexNet、VGG、ResNet 都可以看作对“如何构建更深的 CNN”这一问题的不同回答。但网络并不是简单地越深越好，深度增加会同时带来计算量增加、过拟合风险以及优化困难等问题，所以经典 CNN 架构的发展重点逐渐从“使用 CNN”转向“怎样合理地把 CNN 做深”。

---

**AlexNet：** AlexNet 是深度 CNN 发展中的经典网络，它通过多层卷积、ReLU、Pooling 和全连接层构成较深的图像分类网络，并结合 Dropout、数据增强等方法改善训练效果。相比更早期的浅层网络，AlexNet 展示了深层 CNN 在图像任务上的强大能力，也推动了 ReLU 在深层网络中的广泛使用。可以把 AlexNet 的基本思路概括为：

$$
\text{Image}
\rightarrow
\text{Conv / ReLU / Pooling}
\rightarrow
\cdots
\rightarrow
\text{Flatten}
\rightarrow
\text{FC}
\rightarrow
\text{Classification}
$$

在本项目中实现的是针对 MNIST 尺寸进行调整的 AlexNet-style 网络，而不是直接照搬原始 AlexNet：使用 5 个卷积层逐渐将 Channel 从 $1$ 提高到 $64、192、384、256、256$，随后使用全连接分类器完成 $0\sim9$ 的数字分类。该实验的 Test Accuracy 为 99.33%。

---

**ReLU 在深层网络中的作用：** ReLU 的数学表达式仍然是：

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

相比 Sigmoid、Tanh 等在输入绝对值较大时容易进入饱和区域的激活函数，ReLU 在正数区域的导数为 1：

$$
\frac{d}{dx}\operatorname{ReLU}(x)
=
\begin{cases}
0, & x<0\\
1, & x>0
\end{cases}
$$

因此正数区域的梯度不会因为激活函数本身随着输入增大而逐渐趋近于 0，这使深层网络通常更容易进行梯度传播和优化。AlexNet 对 ReLU 的成功使用也是其重要特点之一。不过 ReLU 在负数区域梯度为 0，因此仍然存在 Dead ReLU 问题。

---

**Dropout：** Dropout 是一种常用的正则化方法，在训练过程中按照给定概率随机将一部分神经元的输出置为 0，使网络不能长期过度依赖某几个固定神经元，而需要学习更加分散、更加具有鲁棒性的特征表示，从而缓解过拟合。假设 Dropout Probability 为 $p$，可以直观理解为训练过程中每个神经元都有概率 $p$ 暂时不参与当前这次计算。需要注意，Dropout 的随机丢弃主要发生在训练阶段，推理时不会继续随机关闭神经元，因此使用包含 Dropout 的模型时必须正确切换 `model.train()` 和 `model.eval()`。本项目 AlexNet-style 和 VGG-style 的分类器中都使用了 Dropout。

---

**正则化（Regularization）：** 正则化泛指为了减少模型过度拟合训练数据、提高泛化能力而采用的方法。深层网络参数较多，表达能力很强，因此也更容易把训练数据中的偶然细节学进去。Dropout、Weight Decay、数据增强等都可以从不同角度起到正则化作用。正则化通常不是为了让 Training Loss 尽可能低，而是希望模型在没有见过的数据上仍然具有较好的表现，因此应该结合 Validation Set 和 Test Set 判断效果。

---

**数据增强（Data Augmentation）：** 数据增强是在不额外人工收集大量新数据的情况下，对已有训练样本进行合理变换，从而产生更多训练样本的方法，例如图像任务中的随机裁剪、翻转、旋转、颜色变化等。它的核心假设是这些变换不应该改变样本真正的语义，例如一张猫的图片进行轻微裁剪后仍然是猫。数据增强可以增加训练数据的多样性，使模型不容易只记住训练集中的固定形式，从而提高泛化能力。具体采用哪些增强必须根据任务决定，例如手写数字如果进行不合理的大角度旋转，就可能改变数字本身的含义。

---

**VGG：** VGG 是经典的深层 CNN 架构，其重要特点不是设计大量不同类型的复杂卷积操作，而是使用结构统一的小卷积核，并通过重复堆叠卷积层和 Pooling 构建更深的网络。VGG 常使用 $3\times3$ 卷积，在相同空间尺度上连续进行多次卷积，再通过 Pooling 下采样，可以概括为：

$$
\text{Conv}
\rightarrow
\text{ReLU}
\rightarrow
\text{Conv}
\rightarrow
\text{ReLU}
\rightarrow
\text{Pooling}
$$

然后不断重复类似模块。这样的设计使网络结构规则、模块化程度高，也说明复杂的视觉特征可以通过重复堆叠简单的小卷积逐步学习。本项目实现的同样是针对 MNIST 调整后的 VGG-style 网络，最终 Test Accuracy 为 99.06%。

---

**小卷积核堆叠（Stacked Small Convolutions）：** VGG 的代表性思想之一是重复使用 $3\times3$ 小卷积，而不是每一层都使用很大的卷积核。两个连续的 $3\times3$、Stride 为 1 的卷积可以获得大约 $5\times5$ 的有效感受范围：

$$
3\times3
\rightarrow
3\times3
\quad
\Longrightarrow
\quad
5\times5\ \text{Receptive Field}
$$

三个连续的 $3\times3$ 卷积则可以获得大约 $7\times7$ 的有效感受范围。这样不仅能够逐渐扩大感受野，还能在卷积层之间加入更多 ReLU，使网络拥有更多次非线性变换。因此 VGG 展示了一种重要的深层网络设计思想：不一定需要不断设计新的复杂操作，也可以通过规则地重复简单模块建立深层网络。

---

**网络模块化（Modular Design）：** 网络模块化指把经常重复出现的一组网络操作组织成一个相对独立的模块，然后通过重复堆叠模块构建完整网络。VGG 中连续卷积加 Pooling 就体现了这种思想。模块化可以减少网络结构设计的混乱，使代码更容易理解、修改和复用。这个思想在后面的 ResNet 和 U-Net 中会更加明显，例如 ResNet 使用重复的 Residual Block，而 U-Net 则可以把 Encoder Block 和 Decoder Block 作为基本模块。

---

**网络深度（Depth）：** 网络深度通常表示模型中参与特征变换的层数。更深的网络理论上可以通过更多层次的变换表示更复杂的函数和特征，但深度增加并不意味着性能一定提高。除了更高的计算成本，深层网络还会使梯度需要经过更长的传播路径，优化变得更加困难。VGG 说明了“规则地堆叠更多卷积层”可以建立很深的 CNN，而 ResNet 接下来进一步解决的是“网络继续加深以后怎样保持可训练性”。

---

**梯度消失（Vanishing Gradient）：** 梯度消失指反向传播过程中，梯度经过很多层连续相乘后逐渐变得非常小，使靠近网络前部的参数几乎得不到有效更新。根据链式法则：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial h_n}
\frac{\partial h_n}{\partial h_{n-1}}
\cdots
\frac{\partial h_1}{\partial x}
$$

如果其中很多局部导数的绝对值都小于 1，连续相乘后梯度可能迅速趋近于 0。Sigmoid、Tanh 的饱和区域尤其容易造成这种问题。ReLU、合理初始化、Batch Normalization、Residual Connection 等技术都可以从不同角度改善深层网络的训练。

---

**梯度爆炸（Exploding Gradient）：** 梯度爆炸与梯度消失相反，指反向传播时多个较大的梯度因子连续相乘，使梯度变得非常大，从而导致参数更新剧烈、Loss 不稳定甚至出现数值问题。梯度消失和梯度爆炸都说明：随着网络变深，梯度需要经过越来越长的路径，单纯增加层数并不一定容易训练，因此深层网络需要更加合理的结构和训练方法。

---

**Batch Normalization（BatchNorm / BN）：** Batch Normalization 是深层网络中常用的归一化方法，它在训练过程中利用一个 Mini-batch 的统计量对中间特征进行标准化，然后再通过可学习参数 $\gamma$ 和 $\beta$ 进行缩放和平移。简化形式可以表示为：

$$
\mu_B
=
\frac{1}{m}
\sum_{i=1}^{m}x_i
$$

$$
\sigma_B^2
=
\frac{1}{m}
\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

$$
\hat{x}_i
=
\frac{x_i-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y_i
=
\gamma\hat{x}_i+\beta
$$

BN 可以改善中间特征的数值尺度和训练稳定性，使深层网络通常更容易优化，也常允许使用较大的学习率。它与输入图片预处理中的 Normalization 不同：输入归一化发生在数据进入网络之前，而 BatchNorm 是网络内部的层，并且具有可学习参数。BN 在训练和推理阶段的行为也不同，因此包含 BatchNorm 的模型同样需要正确使用 `model.train()` 和 `model.eval()`。

---

**ResNet（Residual Network，残差网络）：** ResNet 的核心思想是引入残差连接，让网络中的某些层不必直接学习完整映射 $H(x)$，而是学习相对于输入的残差 $F(x)$，最终输出为：

$$
H(x)=F(x)+x
$$

其中 $x$ 通过 Shortcut 直接传到后面，与卷积分支学习到的 $F(x)$ 相加。这样如果某些新增层暂时没有学到有用变换，只要 $F(x)$ 接近 0，整个 Block 就可以接近恒等映射：

$$
H(x)\approx x
$$

这使非常深的网络相比单纯堆叠卷积层更容易优化，同时 Shortcut 也为信息和梯度传播提供了更直接的路径。因此 ResNet 的意义并不是简单地“网络更深”，而是让网络能够更加稳定地扩展到更深的结构。

---

**残差学习（Residual Learning）：** 普通网络希望若干层直接学习目标映射：

$$
H(x)
$$

而残差学习把目标改写成：

$$
F(x)=H(x)-x
$$

因此：

$$
H(x)=F(x)+x
$$

也就是说，卷积分支只需要学习“输入还需要改变多少”，原始输入则通过 Shortcut 直接保留下来。如果最合适的变换接近 Identity，那么让 $F(x)$ 接近 0 往往比让多层卷积重新学习完整的 Identity Mapping 更容易。残差学习是 ResNet 能够有效训练更深网络的核心思想。

---

**Residual Block / BasicBlock：** Residual Block 是 ResNet 的基本组成模块。以经典 BasicBlock 为例，主分支通常包含两个 $3\times3$ 卷积以及 BatchNorm 等操作，同时输入 $x$ 通过 Shortcut 分支绕过这些卷积，最后两条路径相加：

$$
y
=
F(x)+x
$$

再进行后续激活。可以粗略表示为：

```text
                 ┌──────────── Shortcut ────────────┐
                 │                                  │
x ──→ Conv ──→ BN ──→ ReLU ──→ Conv ──→ BN ──→ Add ──→ ReLU
                 │                                  ↑
                 └──────────────────────────────────┘
```

多个 Residual Blocks 可以继续堆叠形成更深的 ResNet。本项目实现的 ResNet-18-style 使用 `[2, 2, 2, 2]` 的 Block 配置，并逐步增加 Channel。

---

**Shortcut / Skip Connection（捷径连接 / 跳跃连接）：** Shortcut 指让输入绕过若干中间层，直接传递到网络后面的连接。在 ResNet 中，Shortcut 的结果与主分支通常采用逐元素相加：

$$
y=F(x)+x
$$

它提供了一条比连续通过所有卷积层更直接的信息和梯度传播路径，使深层网络更加容易优化。后面的 U-Net 同样会使用 Skip Connection，但目的和连接方式并不完全相同：ResNet 主要通过 Addition 学习残差，而 U-Net 通常通过 Concatenation 把 Encoder 的高分辨率特征直接提供给 Decoder。

---

**Identity Mapping（恒等映射）：** Identity Mapping 指输入不经过改变直接得到相同输出：

$$
H(x)=x
$$

ResNet 中当输入和主分支输出具有相同 Shape 时，Shortcut 可以直接使用 Identity：

$$
x\rightarrow x
$$

不需要额外的卷积参数。残差结构的一个重要优势就是：如果某些层没有必要对输入进行复杂变换，那么网络可以通过让 $F(x)$ 接近 0，使整个 Block 自然接近 Identity Mapping，而不必让多层卷积自己重新学习“什么都不改变”。

---

**1×1 Convolution：** $1\times1$ 卷积的空间 Kernel Size 只有一个像素，因此它不会像 $3\times3$ 卷积那样直接观察周围邻域，但它可以在同一个空间位置上对不同 Channels 进行线性组合，并改变 Channel 数。例如：

$$
C_{\text{in}}
\rightarrow
C_{\text{out}}
$$

在 ResNet 中，如果主分支因为下采样或 Channel 改变导致输出 Shape 与输入不同，就不能直接执行：

$$
F(x)+x
$$

此时可以在 Shortcut 上使用 $1\times1$ 卷积和合适的 Stride，将输入变换到与主分支相同的 Shape：

$$
y=F(x)+W_sx
$$

其中 $W_s$ 表示 Shortcut 上的投影变换。这样既保留了残差连接，又解决了尺寸或 Channel 不匹配的问题。

---

**Addition（逐元素相加）：** ResNet 中 Shortcut 与主分支的融合通常采用逐元素相加，因此两个 Tensor 必须具有相同 Shape。对于：

$$
A,B\in\mathbb{R}^{C\times H\times W}
$$

相加后仍然得到：

$$
A+B\in\mathbb{R}^{C\times H\times W}
$$

因此 Addition 不会因为融合两条分支而直接增加 Channel 数。这个特点与后面 U-Net 常用的 Concatenation 不同，是理解 ResNet Skip Connection 和 U-Net Skip Connection 区别的关键。

---

**Global Average Pooling（GAP，全局平均池化）：** GAP 会对每个 Channel 的整个空间区域取平均值。例如输入 Feature Map 为：

$$
C\times H\times W
$$

经过 GAP 后可以得到：

$$
C\times1\times1
$$

第 $c$ 个 Channel 可以表示为：

$$
y_c
=
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
x_{cij}
$$

这样可以把每个 Channel 的空间信息压缩成一个数值，再交给最终分类层。与直接 Flatten 大尺寸 Feature Map 后连接巨大 Fully Connected Layer 相比，GAP 可以显著减少分类器参数，同时对输入空间尺寸更加灵活。本项目的 ResNet-18-style 使用 `AdaptiveAvgPool2d((1,1))`，然后将得到的 256 维特征输入最终分类层。

---

**Adaptive Average Pooling：** Adaptive Average Pooling 与固定 Kernel Size 的 Pooling 不同，它直接指定希望得到的输出尺寸，再由框架根据输入尺寸自动决定池化区域。例如：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

无论前面的 Feature Map 具体空间尺寸是多少，最终都会得到：

$$
C\times1\times1
$$

因此它常用于 CNN 分类网络的末尾，把不同空间尺寸的 Feature Map 转换成固定长度的 Channel 特征。

---

**Scaling：** Scaling 可以理解为扩大模型规模，例如增加网络深度、宽度或其他模型容量。理论上更大的模型能够表示更加复杂的函数，但简单地增加层数并不保证效果提升，因为深层网络本身可能越来越难优化。ResNet 的重要意义之一就是通过 Residual Connection 改善深层网络的信息和梯度传播，使网络能够更加有效地向更深的结构扩展。因此题目中“残差跳跃使得模型更适合 Scaling”，重点并不是“ResNet 越深一定越好”，而是残差结构让增加深度这件事变得更加可行。

---

**AlexNet、VGG 与 ResNet 的关系：** 三种网络都属于 CNN，但代表了不同阶段的深层网络设计思想。AlexNet 展示了深层 CNN 配合 ReLU、Dropout、数据增强等方法在图像任务中的有效性；VGG 强调使用规则的 $3\times3$ 小卷积和重复模块构建更深、更统一的网络；ResNet 则进一步面对深层网络越来越难优化的问题，通过 Residual Connection 和 Shortcut 改善信息与梯度传播。因此它们可以粗略形成这样一条发展路线：

$$
\text{AlexNet}
\rightarrow
\text{Deep CNN}
$$

$$
\text{VGG}
\rightarrow
\text{Stack Small Convolutions}
$$

$$
\text{ResNet}
\rightarrow
\text{Residual Learning}
$$

它们并不是简单的“后一种全面替代前一种”，而是分别体现了 CNN 架构发展中的重要设计思想。

---

**参数量（Parameters）与计算量（Computation）：** 模型参数量表示需要学习和存储多少 Weight、Bias 等参数，而计算量描述完成 Forward / Backward 需要进行多少运算，两者有关但不是同一个概念。一个模型参数较少，并不意味着训练一定更快，因为卷积操作可能在较大的 Feature Maps 上重复执行很多次，网络深度、Channel 数、BatchNorm、内存访问等都会影响实际速度。本项目中 ResNet-18-style 的参数量约为 2.80M，小于 AlexNet-style 的约 3.56M，但实际训练时间约 101.42 s，反而明显长于 AlexNet-style 的约 58.28 s，这正说明“参数更少”和“计算更快”不能画等号。

---

**模型复杂度与准确率：** 更深、更复杂的网络具有更强的表示能力，但最终准确率还受到数据集难度、模型结构、优化器、学习率、训练 Epoch、正则化等许多因素影响，因此不能认为模型越复杂 Accuracy 就一定越高。本项目中 AlexNet-style、VGG-style 和 ResNet-18-style 的 Test Accuracy 分别为 99.33%、99.06% 和 99.18%，三个模型都超过 99%，但结果并没有按照网络深度或结构复杂度严格排序。这说明网络架构需要结合具体任务分析，而不能只根据“更深”判断模型效果。

---

**从分类网络到 U-Net：** AlexNet、VGG 和 ResNet 的目标仍然主要是图像分类，也就是把整张图片最终压缩成一个类别结果：

$$
\text{Image}
\rightarrow
\text{Features}
\rightarrow
\text{Class}
$$

但 Level 4 的手写痕迹去除任务要求输入一张图片，并输出另一张与原图具有对应空间结构的图片：

$$
\text{Image}
\rightarrow
\text{Image}
$$

这意味着网络不仅需要知道“图片里有什么”，还需要知道“这些信息具体位于哪里”，并最终恢复高分辨率空间结构。传统分类 CNN 不断通过 Pooling 和 Downsampling 压缩空间信息的方式已经不能直接满足这个要求，因此需要从分类网络进一步发展到 Encoder-Decoder，并最终引出 U-Net。



# 四、U-Net 与图像到图像任务

**图像分类（Image Classification）：** 图像分类的目标是输入一张图片，最终输出一个类别。例如 MNIST 中输入一张 $28\times28$ 的手写数字图片，输出 $0\sim9$ 中的一个类别：

$$
\text{Image}
\rightarrow
\text{Class}
$$

因此分类网络通常可以不断通过卷积和下采样压缩空间信息，最后把整张图片转换成一个特征向量进行分类。AlexNet、VGG 和 ResNet 最经典的应用都属于这种任务。

---

**图像分割（Image Segmentation）：** 图像分割与分类不同，它不是只给整张图片一个类别，而是需要对图像中的不同像素或区域进行预测，因此输出必须保留空间位置。例如语义分割可以对每个像素预测其所属类别：

$$
\text{Image}
\rightarrow
\text{Pixel-wise Prediction}
$$

这意味着网络不能只提取“图片中有什么”，还需要知道“这些内容位于哪里”。U-Net 最初就是为生物医学图像分割提出的经典网络，因此它特别重视高分辨率空间信息的恢复。

---

**图像到图像任务（Image-to-Image）：** Image-to-Image 指输入是一张图片，输出仍然是一张具有对应空间结构的图片：

$$
\text{Input Image}
\rightarrow
\text{Output Image}
$$

图像去噪、图像修复、超分辨率、图像风格转换以及本项目的手写痕迹去除都可以看作 Image-to-Image 问题。与分类任务不同，这类任务不仅要求模型理解图像内容，还要求输出中的空间位置与输入保持正确对应，因此通常需要同时完成“提取上下文信息”和“恢复空间细节”两个过程。

---

**U-Net：** U-Net 是经典的 Encoder-Decoder 卷积神经网络，其结构整体类似字母 U，因此得名 U-Net。左侧 Encoder 不断通过卷积和 Downsampling 提取特征并压缩空间尺寸，中间 Bottleneck 得到较深层的特征表示，右侧 Decoder 再通过 Upsampling 逐步恢复空间分辨率。同时，U-Net 使用 Skip Connection 将 Encoder 中较高分辨率的特征直接传递给 Decoder，使 Decoder 在恢复图片时能够重新利用前面保留下来的空间细节。整体结构可以概括为：

$$
\text{Input}
\rightarrow
\text{Encoder}
\rightarrow
\text{Bottleneck}
\rightarrow
\text{Decoder}
\rightarrow
\text{Output}
$$

同时还存在：

$$
\text{Encoder Features}
\xrightarrow{\text{Skip Connection}}
\text{Decoder}
$$

因此 U-Net 同时兼顾了深层语义/上下文信息和浅层空间细节，非常适合需要输出高分辨率空间结果的任务。

---

**Encoder（编码器）：** Encoder 是 U-Net 左侧负责特征提取的部分。它通常由多层卷积和 Downsampling 组成，随着网络逐渐深入，Feature Map 的高度和宽度不断减小，而 Channel 数通常不断增加。例如可以出现：

$$
256\times256
\rightarrow
128\times128
\rightarrow
64\times64
\rightarrow
32\times32
$$

空间尺寸减小使后面的神经元能够整合更大范围的上下文信息，而增加 Channel 则允许网络学习更多种类的特征。Encoder 可以理解为逐渐把原始像素转换成更加抽象的特征表示。

---

**Downsampling（下采样）：** Downsampling 是降低 Feature Map 空间分辨率的过程，可以通过 Max Pooling 或 Stride 大于 1 的卷积实现。例如：

$$
256\times256
\rightarrow
128\times128
$$

下采样能够减少计算量并扩大后续特征的感受范围，使网络逐渐从局部像素信息转向更大范围的上下文信息。但下采样也会不可避免地损失一部分精确的空间位置和细节，因此对于 Image-to-Image 任务，不能只进行下采样，还需要 Decoder 和 Skip Connection 来恢复这些信息。

---

**Bottleneck（瓶颈层）：** Bottleneck 位于 U-Net 最深处，连接 Encoder 和 Decoder。此时 Feature Map 的空间尺寸通常最小，而 Channel 数较多，因此每个特征能够看到较大的输入区域。Bottleneck 可以理解为 Encoder 对输入进行多次压缩和特征提取之后得到的高层表示，也是网络从“不断压缩”转向“逐步恢复”的位置。

---

**Decoder（解码器）：** Decoder 是 U-Net 右侧负责恢复空间分辨率的部分。它会逐层进行 Upsampling，把较小的 Feature Map 恢复到更大的空间尺寸，并结合 Encoder 通过 Skip Connection 提供的高分辨率特征，最终生成与目标图像尺寸对应的输出。因此 Encoder 更偏向“理解和压缩”，Decoder 更偏向“恢复和重建”。两者结合，使网络能够完成像素级预测或 Image-to-Image 任务。

---

**Upsampling（上采样）：** Upsampling 是增加 Feature Map 空间尺寸的过程，例如：

$$
64\times64
\rightarrow
128\times128
$$

它与 Downsampling 的方向相反。常见方法包括插值后卷积、转置卷积等。需要注意，上采样并不是把下采样过程中已经丢失的信息凭空恢复回来，它主要负责扩大空间尺寸；真正的细节恢复还需要网络已经学习到的特征以及 Encoder 通过 Skip Connection 提供的信息。

---

**转置卷积（Transposed Convolution）：** 转置卷积是一种可学习的上采样方式，可以在增加 Feature Map 空间尺寸的同时学习特征变换。它常通过 `ConvTranspose2d` 等结构实现。与普通插值不同，转置卷积包含可学习参数，因此模型能够根据训练数据学习如何进行上采样。不过转置卷积如果 Kernel Size、Stride 等设置不合理，也可能产生棋盘格状伪影，因此实际网络设计中也常使用插值加普通卷积作为另一种上采样方案。

---

**Skip Connection（跳跃连接）：** U-Net 的 Skip Connection 会把 Encoder 某一层得到的高分辨率 Feature Map 直接传递到 Decoder 对应层，使 Decoder 在恢复图像时能够重新利用 Encoder 中保留的空间细节。因为深层特征经过多次 Downsampling 后虽然包含较强的上下文信息，但精确的位置和边缘信息会逐渐损失，所以仅依赖 Bottleneck 很难恢复细节。Skip Connection 相当于给 Decoder 提供了一条“细节捷径”，让高分辨率特征不必全部经过最深层压缩后再恢复。

---

**Concatenation（拼接）：** U-Net 的 Skip Connection 通常使用 Concatenation 将 Encoder Feature 和 Decoder Feature 沿 Channel 维度拼接。假设两个 Feature Maps 都具有：

$$
C\times H\times W
$$

拼接之后得到：

$$
2C\times H\times W
$$

因此 Concatenation 会增加 Channel 数，让后续卷积同时看到 Encoder 的高分辨率特征和 Decoder 当前已经恢复出的特征。随后再通过卷积将这些信息进行融合。

---

**ResNet Skip Connection 与 U-Net Skip Connection：** 两种网络都使用 Skip Connection，但目的和融合方式不同。ResNet 通常采用逐元素 Addition：

$$
y=F(x)+x
$$

重点是学习 Residual，并为深层网络提供更直接的信息和梯度传播路径；U-Net 通常使用 Concatenation：

$$
F_{\text{decoder}}
=
\operatorname{Concat}
(
F_{\text{encoder}},
F_{\text{upsampled}}
)
$$

重点是把 Encoder 中的高分辨率空间特征提供给 Decoder，帮助恢复细节。因此可以简单记为：

$$
\boxed{
\text{ResNet：Add，帮助深层网络优化}
}
$$

$$
\boxed{
\text{U-Net：Concat，帮助恢复空间细节}
}
$$

---

**Paired Data（成对数据）：** Image-to-Image 的监督学习通常需要输入图片和目标图片一一对应，可以表示为：

$$
(x_i,y_i)
$$

其中 $x_i$ 是需要处理的 Input，$y_i$ 是期望模型输出的 Target。模型学习的是：

$$
f_\theta(x_i)\approx y_i
$$

本项目中，Input 是包含手写痕迹的文档图片，Target 是对应的干净文档图片。Input 和 Target 必须正确配对并在空间上合理对应，否则同一个像素位置表达的内容不一致，像素级 Loss 就无法提供正确监督。

---

**数据清洗（Data Cleaning）：** 数据清洗是训练前检查和处理数据质量的过程，例如检查 Input 与 Target 是否正确配对、图片是否损坏、尺寸是否合理、是否存在明显异常样本等。对于 Image-to-Image 任务，数据质量尤其重要，因为模型直接根据成对图片之间的差异学习映射。如果配对错误或 Target 本身存在问题，模型就可能学习到错误的变化规律。

---

**标注规范（Annotation Convention）：** 标注规范指明确规定什么样的内容应该保留、什么样的内容应该删除，以及 Target 应该满足什么标准。对于传统分割任务，标注可能是一张像素级 Mask；对于本项目这样的成对图像恢复任务，Target 本身就构成监督信号。无论使用哪种方式，标注规则都需要保持一致，否则相似输入在训练集中可能对应互相矛盾的目标，使模型难以学习稳定映射。

---

**Train / Validation / Test Split：** Image-to-Image 数据同样需要划分训练集、验证集和测试集。Training Set 用于更新参数，Validation Set 用于训练过程中评价模型和选择 Checkpoint，Test Set 用于最终评价模型在未参与开发的数据上的泛化效果。划分时需要注意数据之间是否存在高度相关或重复样本，避免相似图片同时出现在 Train 和 Test 中造成数据泄露。

---

**文档图像预处理（Document Image Preprocessing）：** 文档图片与普通自然图像不同，常见问题包括光照不均、纸张背景变化、阴影、扫描或拍摄产生的亮度差异等。因此可以在进入网络前进行适当的背景归一化、灰度处理或文档增强，使模型更集中地学习文字和手写痕迹等真正与任务有关的信息。本项目中使用了从 RGB 到灰度，再进行 Background Normalization 和 Document Enhancement 的预处理流程。

---

**数据增强（Data Augmentation）：** 在文档 Image-to-Image 任务中，数据增强需要同时作用于 Input 和 Target 的几何结构，否则二者会失去像素对应关系。例如如果对 Input 进行了裁剪或旋转，那么 Target 必须执行完全相同的几何变换。合理的数据增强能够提高样本多样性和泛化能力，但增强操作不能破坏文档内容本身的语义，也不能让 Input 和 Target 错位。

---

**Patch（图像块）：** Patch 是从较大的原始图片中裁剪出来的一小块局部图像。例如从高分辨率文档中提取：

$$
256\times256
$$

的局部区域作为一个训练样本。对于高分辨率文档，如果直接把整张图片送入 U-Net，会产生很大的 Feature Maps 和显存占用，因此 Patch 可以在保留局部文字细节的同时控制训练成本。

---

**Patch Training：** Patch Training 指训练时不直接使用整张高分辨率图片，而是从原图中裁剪较小的 Patch 送入模型。本项目使用 $256\times256$ Patch。这样做有两个主要原因：一是不同原始文档尺寸不完全一致，而固定 Patch 可以统一输入 Shape；二是 U-Net 中间层会保存大量 Feature Maps，直接训练高分辨率整图会消耗大量显存。相比直接把整张高分辨率文档 Resize 到 $256\times256$，Patch Training 还能更好地保留文字笔画和细线等局部细节。

---

**为什么不能直接把整张文档缩小到 256×256：** 高分辨率文档中很多文字、笔画和细线本身只占少量像素，如果直接将整张图片 Resize 到很小的分辨率，这些细节可能被严重压缩甚至消失。模型即使成功学习，也无法恢复输入中已经被缩放破坏的信息。因此本项目选择保持原图分辨率，在训练时裁剪 $256\times256$ Patch，在推理时再通过滑动 Patch 处理完整图片。

---

**Difference Mask（差异区域掩码）：** 本项目根据 Input 与 Target 的像素差异构造 Difference Mask，用于近似表示哪些区域发生了明显变化。可以粗略表示为：

$$
M_i=
\begin{cases}
1, & |x_i-y_i|>\tau\\
0, & |x_i-y_i|\leq\tau
\end{cases}
$$

本项目使用的阈值为：

$$
\tau=0.1
$$

Difference Mask 可以让 Loss 或指标更加关注 Input 和 Target 真正存在明显差异的区域。但它并不是人工标注的“语义手写 Mask”，因为 Input 与 Target 的像素差异除了手写内容之外，也可能来自对齐误差、亮度变化、背景变化等因素。因此更准确的理解是“像素变化区域”，而不是严格的“手写区域”。

---

**L1 Loss：** L1 Loss 使用预测图像与 Target 之间绝对误差的平均值：

$$
L_{\mathrm{L1}}
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{y}_i-y_i|
$$

L1 对误差进行线性惩罚，对特别大的误差相对没有 MSE 那么敏感。在图像重建任务中，L1 常用于约束预测图片整体接近 Target，同时相比单独使用平方误差通常不容易过度强调少量极大误差。本项目最终 Loss 中使用 Global L1 作为整体图像重建约束之一。

---

**L2 Loss 与 MSE：** L2 通常表示基于平方误差的距离，而 MSE 是最常见的平方误差平均形式：

$$
L_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

由于误差被平方，较大的误差会受到更强的惩罚。需要注意不同资料中的 “L2 Loss” 定义可能存在差异，有时表示平方 L2 范数，有时表示欧氏距离，因此实际使用时应该确认具体公式。MSE 的定义则相对明确。

---

**Global Loss（全局损失）：** Global Loss 在整张预测图像或整个 Patch 上计算误差，因此能够约束模型整体输出接近 Target。例如 Global L1：

$$
L_{\mathrm{Global\ L1}}
=
\frac{1}{N}
\sum_i
|\hat{y}_i-y_i|
$$

这种 Loss 可以保证模型不会只关注手写变化区域而忽略其他背景和印刷内容。但本项目中真正发生变化的像素只占图片的一部分，如果完全依赖 Global Loss，大量本来就很相似的背景像素会在平均过程中占据较大比例。

---

**Changed-Region Loss（变化区域损失）：** Changed-Region Loss 只重点计算 Difference Mask 标记出的变化区域。若 $M_i$ 表示 Difference Mask，则 Changed-Region MSE 可以写成：

$$
L_{\mathrm{changed}}
=
\frac{
\sum_i
M_i(\hat{y}_i-y_i)^2
}{
\sum_i M_i+\epsilon
}
$$

这样可以提高真正发生明显像素变化区域在优化目标中的权重，避免模型仅仅通过“保持大部分背景不变”就得到较低的平均 Loss。

---

**Weighted / Combined Loss（加权组合损失）：** 当一个任务同时具有多个优化目标时，可以把多个 Loss 按照不同权重组合：

$$
L
=
\lambda_1L_1
+
\lambda_2L_2
+
\lambda_3L_3
$$

本项目最终采用的组合可以表示为：

$$
L
=
L_{\mathrm{Global\ L1}}
+
0.5L_{\mathrm{Global\ MSE}}
+
2.0L_{\mathrm{Changed\ Region\ MSE}}
$$

其中 Global L1 约束整体像素差异，Global MSE 对较大的全局误差进行更强惩罚，而 Changed-Region MSE 进一步强调发生明显变化的区域。组合 Loss 的核心不是简单增加 Loss 数量，而是让不同 Loss 分别约束任务中不同的重要目标。

---

**训练成本控制（Training Cost Control）：** 深度学习实验不应该每修改一次代码就直接进行完整训练，因为完整数据集、较多 Epoch 和大模型可能消耗大量时间与计算资源。更加合理的方式是先使用少量数据、较少 Epoch 或较小规模配置验证整个 Pipeline 能否正常运行，包括数据读取、Forward、Loss、Backward、Checkpoint 和 Inference；确认流程正确后再开始正式训练。这样可以避免因为简单代码错误而浪费大量 GPU 时间。

---

**Pipeline Verification（训练流程验证）：** Pipeline Verification 指正式训练前先确认完整训练链路能够正确运行。例如先使用几个 Batch 或少量数据完成：

$$
\text{Data}
\rightarrow
\text{Model}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Checkpoint}
\rightarrow
\text{Inference}
$$

并检查 Tensor Shape、Loss 是否正常、模型是否能够保存加载以及推理输出是否合理。对于 U-Net 这种结构较复杂、训练成本较高的任务，先验证 Pipeline 再正式训练尤其重要。

---

**PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）：** PSNR 是图像重建任务中常见的评价指标，它基于预测图像和 Target 之间的 MSE 计算：

$$
\operatorname{PSNR}
=
10\log_{10}
\left(
\frac{MAX_I^2}
{\operatorname{MSE}}
\right)
$$

其中 $MAX_I$ 表示图像允许的最大像素值，如果图像已经归一化到 $0\sim1$，通常可以取 $MAX_I=1$。因为 MSE 越小，PSNR 越高，所以通常 PSNR 越高表示预测图像在像素误差意义上越接近 Target。但 PSNR 主要基于逐像素误差，并不能完全反映人眼对图像结构和视觉质量的感受，因此通常需要结合 SSIM 和实际可视化结果一起判断。

---

**SSIM（Structural Similarity Index，结构相似性）：** SSIM 是衡量两张图像结构相似程度的常用指标，它不仅直接比较像素误差，还综合考虑亮度、对比度和结构信息。其经典形式可以表示为：

$$
\operatorname{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)
(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}
$$

其中 $\mu_x,\mu_y$ 表示局部平均亮度，$\sigma_x,\sigma_y$ 与 $\sigma_{xy}$ 描述局部对比度和结构关系，$C_1,C_2$ 用于保持数值稳定。SSIM 通常越接近 1，表示两张图像在结构上越相似。相比只基于 MSE 的 PSNR，SSIM 更强调结构信息，但它同样不能完全代替实际视觉检查。

---

**指标与视觉结果：** 图像生成或恢复任务不能只依赖一个数字指标判断模型好坏。例如 PSNR 较高说明像素误差较小，但并不保证文字边缘、细线和视觉质量一定最好；Validation Loss 更低也不意味着最终生成结果一定更符合实际任务需求。因此应该综合考虑 Loss、PSNR、SSIM 和实际可视化结果。本项目中也对不同 Checkpoint 的输出进行了视觉比较，而不是只根据某一个 Loss 数值选择最终模型。

---

**异常输出（Degenerate Output）：** Image-to-Image 模型有时可能产生明显异常的结果，例如整张图片几乎全白、全黑或趋于某个固定值。这通常说明数据范围、输出激活、Loss、归一化、模型加载或推理 Pipeline 等环节可能存在问题。因此正式评价指标之前，应先检查输出是否在合理数值范围内，并通过可视化确认模型确实完成了有意义的图像映射。

---

**Sliding Window Inference（滑动窗口推理）：** 模型训练时使用固定大小 Patch，但实际使用时需要处理完整高分辨率图片，因此可以让 $256\times256$ 的窗口在整张图片上逐步滑动，对每一个 Patch 分别执行 U-Net 推理，最后把这些 Patch 重新组合成完整图片。这种方法避免了整张高分辨率图片一次性进入 GPU，同时能够保持训练和推理时局部输入尺寸的一致性。

---

**Overlapping Patches（重叠图像块）：** 如果推理时把整张图片切成完全不重叠的 Patch，每个 Patch 边缘缺少周围上下文，拼接时可能出现明显的网格边界。因此可以让相邻 Patch 保留一定重叠区域。本项目 Patch Size 为 256，Stride 为 192，因此相邻 Patch 之间存在：

$$
256-192=64
$$

像素的重叠区域。这样同一个位置可能被多个 Patch 预测，再通过融合减少 Patch 边缘造成的不连续。

---

**Weighted Blending（加权融合）：** Weighted Blending 用于将多个重叠 Patch 的预测结果平滑地组合起来。如果某个像素同时被多个 Patch 覆盖，可以根据它在各 Patch 中的位置给予不同权重，然后进行加权平均：

$$
\hat{y}(i,j)
=
\frac{
\sum_k
w_k(i,j)\hat{y}_k(i,j)
}{
\sum_k
w_k(i,j)
}
$$

通常可以降低 Patch 边缘预测结果对最终图片的影响，使相邻 Patch 之间过渡更加平滑。它解决的是“怎样把多个局部预测重新组成一张完整图片”的问题，并不会改变 U-Net 本身的网络参数。

---

**Padding 与完整图像推理：** 原始图片的高度和宽度不一定刚好能够被 Patch Size 和 Stride 完整覆盖，因此在进行 Sliding Window Inference 前可以先对图片边缘进行适当 Padding，使所有区域都能够被 Patch 覆盖。完成推理和 Weighted Blending 后，再把额外 Padding 的区域裁掉，恢复原始图片尺寸。因此本项目完整推理流程可以概括为：

$$
\text{Original Image}
\rightarrow
\text{Padding}
\rightarrow
\text{Overlapping Patches}
\rightarrow
\text{U-Net}
\rightarrow
\text{Weighted Blending}
\rightarrow
\text{Crop}
\rightarrow
\text{Final Image}
$$

---

**Checkpoint Selection（模型选择）：** 训练过程中可以保存多个 Checkpoint，但最终模型不一定简单选择 Training Loss 最低的那个。更合理的方式是结合 Validation Loss、PSNR、SSIM、任务目标以及实际输出图片进行判断。本项目还尝试过 Changed-MSE Finetune，虽然对应验证目标有所改善，但最终视觉效果并没有优于原来的 Weighted-Loss Checkpoint，因此最终选择 `best_unet_weighted.pth` 作为正式模型。这说明图像恢复任务中的模型选择应该围绕最终任务效果，而不是只追求某一个数值指标。

---

**GAN（Generative Adversarial Network，生成对抗网络）：** GAN 由 Generator 和 Discriminator 两部分组成。Generator 负责生成结果，Discriminator 负责判断结果更像真实数据还是模型生成的数据，两者在对抗过程中共同训练。可以粗略表示为：

$$
G:x\rightarrow\hat{y}
$$

$$
D:y\rightarrow
\text{Real / Fake}
$$

对于 Image-to-Image 任务，可以让 Generator 根据输入图片生成目标图片，再利用 Discriminator 约束生成结果具有更真实的图像分布和局部细节。相比单纯使用 L1 或 MSE，GAN Loss 有可能改善生成结果的视觉真实性，但训练过程通常更加复杂和不稳定，需要平衡重建目标和对抗目标。

---

**Diffusion Model（扩散模型）：** Diffusion Model 是近年来常见的生成模型，其核心思想可以直观理解为：训练阶段学习如何从逐渐加入噪声的数据中恢复原始数据，生成阶段则从噪声或带条件的信息开始，通过多步去噪逐渐得到最终结果。对于条件图像生成或 Image-to-Image 任务，可以将原始图片作为条件，引导模型逐步生成目标图片。Diffusion Model 通常具有较强的生成能力和细节建模能力，但训练和推理成本也往往比普通 U-Net 前向一次得到结果更高。值得注意的是，U-Net 本身也经常作为 Diffusion Model 中预测噪声或相关目标的骨干网络，因此 U-Net 不仅用于分割和图像恢复，也与现代生成模型存在紧密联系。

---

**U-Net 完整信息流：** U-Net 可以从信息流的角度统一理解为：

$$
\text{Input Image}
\rightarrow
\text{Encoder}
\rightarrow
\text{Downsampling}
\rightarrow
\text{Bottleneck}
\rightarrow
\text{Upsampling}
\rightarrow
\text{Decoder}
\rightarrow
\text{Output Image}
$$

同时：

$$
\text{Encoder Feature}
\xrightarrow{\text{Skip / Concat}}
\text{Decoder}
$$

Encoder 负责逐渐扩大感受范围并提取上下文特征，Decoder 负责恢复空间分辨率，而 Skip Connection 将浅层较精细的空间特征直接提供给 Decoder。因此 U-Net 的关键不是单纯“先缩小再放大”，而是：

$$
\boxed{
\text{深层上下文信息}
+
\text{浅层空间细节}
}
$$

两者共同决定最终的像素级输出。

---

**从 MLP 到 U-Net 的整体学习路线：** 整个项目中的网络结构并不是彼此独立的，而是可以形成一条逐渐发展的思路。MLP 建立了 Forward、Loss、Backward 和 Optimizer 的基本神经网络训练框架；CNN 在此基础上加入局部连接和权重共享，更合理地处理图像空间结构；AlexNet 和 VGG 进一步探索如何构建更深的 CNN；ResNet 通过 Residual Connection 解决深层网络越来越难优化的问题；U-Net 则从单纯的图像分类进一步走向像素级预测和 Image-to-Image，通过 Encoder-Decoder 与 Skip Connection 同时利用上下文信息和空间细节。可以概括为：

$$
\boxed{
\text{MLP}
\rightarrow
\text{CNN}
\rightarrow
\text{AlexNet / VGG}
\rightarrow
\text{ResNet}
\rightarrow
\text{U-Net}
}
$$

对应的理解路线则是：

$$
\boxed{
\text{学习参数}
\rightarrow
\text{提取局部特征}
\rightarrow
\text{构建深层网络}
\rightarrow
\text{让深层网络更容易优化}
\rightarrow
\text{恢复像素级空间输出}
}
$$