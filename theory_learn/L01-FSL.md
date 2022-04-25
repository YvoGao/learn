# 小样本学习

[[1904.05046\] Generalizing from a Few Examples: A Survey on Few-Shot Learning (arxiv.org)](https://arxiv.org/abs/1904.05046)

## 综述

### 问题定义

- 机器学习定义：A computer program is said to learn from experience E with respect to some classes of task T and performance measure P if its performance can improve with E on T measured by P.

  > 计算机程序可以通过使用方法P在任务T中获得经验E来使它的表现变好。但是总是需要大量的数据，这是比较困难的。

- 小样本学习：Few-Shot Learning(FSL) is a type of machine learning problems (specified by E, T and P), where E contains only a limited number of examples with supervised information for the target T.

使用小样本学习典型的几种场景

- 字符生成：学习（E）由给定示例和监督信息以及预先训练的概念（如零件和关系）组成的知识，作为先验知识。生成的字符通过视觉图灵测试（P）的通过率进行评估，该测试可区分图像是由人类还是机器生成的。
- 罕见案例学习：当不能获得充足的训练集来进行训练时，如，考虑一个药物发现任务（T），它试图预测一个新分子是否具有毒性作用。正确分配为有毒或无毒（P）的分子百分比随着（E）的增加而提高，（E）通过新分子的有限分析和许多类似分子的分析作为先验知识获得。
- 减轻样本收集的负担：考虑少量镜头图像分类任务（T）。图像分类精度（P）通过为每个类别的target提取一些标记图像，以及从其他类别（如原始图像）提取先验知识（E）来提高。成功完成此任务的方法通常具有较高的通用性。因此，它们可以很容易地应用于许多样本的任务。

例如下表

![image-20211120202445018](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211120202445018.png)



> *Remark 1.When there is only one example with supervised information inE, FSL is calledone-shot
> learning[14,35,138]. When E does not contain any example with supervised information for the
> targetT, FSL becomes azero-shot learningproblem (ZSL). As the target class does not contain
> examples with supervised information, ZSL requires E to contain information from other modalities
> (such as attributes, WordNet, and word embeddings used in rare object recognition tasks), so as to
> transfer some supervised information and make learning possible.*
>
> 当只有一个有监督信息的样本称为单样本学习，没有办法从监督学习获得信息的时候成为0样本学习，0样本要求从其他地方获得信息。

### 相关的领域

- Weakly supervised learning弱监督学习：仅从包含弱监督（如不完整、不精确、不准确或有噪声的监督信息）的经验中学习。根据人工干预的不同又分为以下几类：

  - Semi-supervised learning半监督学习：从少量有标签数据和大量无标签数据，通常应用文本分类和网页分类。还有一种Positive-unlabeled learning正未学习，只判断样本是未知的还是正向。
  - Active learning主动学习，它选择信息性的未标记数据来查询oracle的输出。这通常用于注释标签昂贵的应用程序，如行人检测。

  > FSL也包括强化学习问题，只有当先验知识是未标记数据且任务是分类或回归时，FSL才成为弱监督学习问题。

- Imbalanced learning不平衡学习：不平衡学习是从经验中学习的，它的分布是偏态的。在欺诈检测和灾难预测应用程序中，当一些值很少被采用时，就会发生这种情况。

- 迁移学习：将知识从训练数据丰富的源域/任务转移到训练数据稀缺的目标域/任务。它可以用于跨域推荐、跨时间段、跨空间和跨移动设备的WiFi定位等应用。

  > 小样本学习中经常使用迁移学习的方法

- 元学习：元学习者在任务中逐渐学习通用信息（元知识），学习者通过任务特定信息概括元学习者的新任务

  >元学习者被视为指导每个特定FSL任务的先验知识。

### 核心问题

#### **经验风险最小化**（Empirical Risk Minimization）

假设一个任务h，我们想最小化他的风险R，损失函数用$p(x,y)$进行计算。得到如下公式

$$R(h)=\int \ell(h(x),y)dp(x,y)=\mathbb{E}[\ell(h(x),y)]$$

因为$p(x,y)是未知的，经验风险在有I个样本的训练集上的平均值$来代理经验风险值$R_I(h)$

$$R_I(h)= \frac{1}I\sum_{i=1}^i \ell(h(x_i),y_i)$$

为方便说明做以下三种假设，

- $\hat{h} = arg {\ } min_h(R(h))$期望最小值函数
- $h^* = arg{\ }min_{h \in \mathcal{H}}R(h)$在$\mathcal{H}$中期望最小值函数
- $h_I=arg {\ }min_{h\in\mathcal{H}}R_I(h)$在$\mathcal{H}$中经验最小值函数

因为$\hat{h}$是未知的，但是在$\mathcal{H}$中$h^*$是$\hat{h}$最好的近似值，所以可以得到误差为

$$\mathbb{E}[R(h_I)-R(\hat h)]=\underbrace{\mathbb{E}[R(h^*)-R(\hat h)]}_{\xi_{app}(\mathcal H)}+\underbrace{\mathbb{E}[R(h_I)-R( h^*)]}_{\xi_{est}(\mathcal H,I)}$$

$\xi_{app}(\cal H)$计算的是在$\cal H$能多接近期望最小是$\hat h， \xi_{est}(\cal H,I)$计算的是经验风险可以多接近在$\cal H$上的期望风险。

#### 不可靠的经验风险最小化（Unreliable Empirical Risk Minimizer）

$\hat h， \xi_{est}(\cal H,I)$可以通过增大I来进行减少，但是在小样本学习中I很小，所以经验风险离期望风险很远，这就是小样本学习中的核心问题，用下图进行表示。

![image-20211126105455171](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126105455171.png)

## 解决方法

根据上面的误差计算公式，我们可以发现，减少误差有三种方法

1. 增大I样本数量
2. 改善模型，缩小$\cal H$的范围
3. 改进算法，使搜索$h_I \in \cal H$更优，初始化$h^*$更接近$\hat h$

![image-20211126105921626](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126105921626.png)

下表为文章中总结的方法

![Empirical Risk Minimization](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211120205637469.png)

### 数据增强

- 从训练集中转换样本

  > - 早期的方法，通过迭代地将每个样本与其他样本对齐，从类似的类中学习一组几何变换。将学习到的转换应用于每个（xi，yi）以形成一个大型数据集，然后可以通过标准机器学习方法学习这些数据集。
  > - 从类似的类中学习一组自动编码器，每个编码器代表一个类内可变性。通过添加学习的变化toxi生成新样本。
  > - 通过假设所有类别在样本之间共享一些可转换的可变性，学习单个转换函数，以将从其他类别学习的样本对之间的变化转换为（xi，yi）
  > - 从大量场景图像中学习的一组独立属性强度回归器将每个样本转换为多个样本，并将原始样本的标签指定给这些新样本。

- 从弱标记或者无标记的数据集中转换样本

  > - 为每个训练集的目标标签学习一个样本SVM，然后用于预测弱标签数据集中样本的标签。然后将具有目标标签的样本添加到训练集中。
  > - 直接使用标签传播来标记未标记的数据集。
  > - 使用渐进策略选择信息性未标记样本。然后为选定的样本指定伪标签，并用于更新CNN。

- 从相似的样本中转换样本

  > 该策略通过聚合和调整来自相似但较大数据集的输入-输出对来增强性能。

选择使用哪种增强策略取决于应用程序。有时，目标任务（或类）存在大量弱监督或未标记的样本，但由于收集注释数据和/或计算成本高，因此小样本学习是首选的。现有的方法主要是针对图像设计的，因为生成的图像可以很容易地由人类进行视觉评估。相比之下，文本和音频涉及语法和结构，更难生成。

### 模型

- 多任务学习：
  - 参数共享。此策略在任务之间直接共享一些参数。eg：两个任务网络共享通用信息的前几层，并学习不同的最终层以处理不同的输出。
  - 参数绑定：正则化对齐不同任务。
  
- 嵌入学习：将每一个例子embed（嵌入）一个低维，这样相似的样本靠的很近，而不同的样本则更容易区分。同时可以构造更小的假设空间$\cal H$。嵌入学习主要从先验知识中学习。

  根据嵌入函数和参数是否随任务改变，将FSL分为三种

  >- 特定于任务的嵌入模型
  >
  >- 任务不变了嵌入模型
  >
  >- 混合嵌入模型
  >
  >  ![image-20211126223422834](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126223422834.png)

- 用外部记忆学习：使用额外的存储器从训练集中学习知识并保存起来（key-value的键值槽）。与嵌入学习不同的是，测试集不直接用这种方式表示，只基于额外存储的内存的相似性，进行预测。![image-20211126224157226](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126224157226.png)

- 生成模型：从先验知识中观察到的x估计的概率分布P(x)。

  ![image-20211126225306394](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126225306394.png)

### 算法

假设$\theta$是在$\cal H$上能获得最好的$h^{*}$，算法通过（i）提供良好的初始化参数$θ_0$，或（ii）直接学习优化器以输出搜索步骤，使用先验知识来影响θ的获取方式。根据先验知识对搜索策略的影响，分为以下三类

- 细化现存参数

  >- 通过正则化微调现有参数
  >
  >  - 早停
  >
  >  - 选择性更新$\theta_0$：仅更新一部分$\theta_0$防止过拟合
  >
  >  - 一起更新$\theta_0$相关部分：可以将$θ_0$的元素分组（例如深层神经网络中的神经元），并使用相同的更新信息对每组进行联合更新。
  >
  >  - 使用模型回归网络：捕获了任务不可知变换，该变换映射了通过对几个示例进行训练获得的参数值。
  >
  >  ![image-20211126233024794](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126233024794.png)
  >
  >- 聚合一组参数：有时，我们没有一个合适的θ0开始。相反，我们有许多从相关任务中学习的模型。例如，在人脸识别中，我们可能已经有了眼睛、鼻子和耳朵的识别模型。因此，可以将这些模型参数聚合到一个合适的模型中，然后直接使用该模型或通过训练集进行细化![image-20211126233009988](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126233009988.png)
  >
  >- 使用新参数微调现有参数：使用新参数微调现有参数。预先训练的θ0可能不足以完全编码新的FSL任务。因此，使用一个附加参数δ来考虑特殊性。具体来说，该策略将模型参数扩展为θ={θ0，δ}，并在学习δ的同时微调θ0。![image-20211126232959714](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126232959714.png)

- 细化元学习参数：使用元学习来细化参数$\theta_0$，它持续被元学习器更新。![image-20211126233220869](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126233220869.png)

  > 一种代表性的方法是模型不可知元学习（MAML）
  >
  > - 包含特定于任务的信息：一般MAML为所有任务提供相同的初始化。但是，这忽略了特定于任务的信息，只有当任务集非常相似时才适用。
  > - 使用元学习$θ_0$建模不确定性：通过几个例子学习不可避免地会导致模型具有更高的不确定性。因此，所学习的模型可能无法以高置信度对新任务执行预测。测量这种不确定性的能力为主动学习和进一步的数据收集提供了提示。
  > - 改进精炼程序：通过几个梯度下降步骤进行细化可能不可靠。正则化可用于纠正下降方向。

- 学习优化器：不使用梯度下降，而是学习一个优化器，该优化器可以直接输出更新。这样就不需要调整步长α或找到搜索方向，因为学习算法会自动完成这项工作。![image-20211126233717798](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211126233717798.png)

  在第t次迭代中，这一行的工作学习一个元学习器，它接受在（t-1）处计算的错误信号，并直接输出更新$\triangle\phi_{t-1}$,更新特定于任务的参数$\phi_{t}=\phi_{t-1}+\triangle\phi_{t-1}$

## 未来工作

### 问题

大多使用数据增强的方式，

可能的方向是多模态在FSL中的设计

### 技术

元学习

避免灾难性遗忘

自动机器学习（Automated machine learning）

### 应用

1. 计算机视觉（除了字符识别和图像分类外，还考虑了其他图像应用。其中包括物体识别[35,36,82]、字体样式转换[7]、短语基础[162]、图像检索[130]、物体跟踪[14]、图像中的特定物体计数[162]、场景位置识别[74]、手势识别[102]、部分标记[24]、图像生成[34,76,107,109]、跨域图像翻译[12]，三维对象的形状视图重建[47]，以及图像字幕和视觉问答[31]。FSL还成功地应用于视频应用，包括运动预测[50]、视频分类[164]、动作定位[152]、人员重新识别[148]、事件检测[151]和对象分割）


2. 机器人学:机器仿生，模仿人的动作等。

3. 自然语言处理（解析[64]、翻译[65]、句子完成（使用从提供的集合中选择的单词填空）[97138]、简短评论中的情感分类[150157]、对话系统的用户意图分类[157]、刑事指控预测[61]、词语相似性任务，如nonce定义[56125]和多标签文本分类[110]。最近，发布了一个名为FewRel[52]的新关系分类数据集。这弥补了自然语言处理中FSL任务基准数据集的不足）

4. 声音信号处理：最近的努力是语音合成。一项流行的任务是从用户的几个音频样本中克隆语音

5. 其他：曲线拟合，医疗，推理方面



### 理论

通过考虑一种特定的元学习方法，在中考察了将一项任务中训练的模型转移到另一项任务中的风险。然而，到目前为止，只有少数方法得到了研究。还有很多理论问题需要探讨。

元学习者学习深层网络的较低层，而学习者学习最后一层，全部使用梯度下降。对元学习方法的收敛性进行更全面的分析将非常有用



## 论文总结

这篇文章总结了近年来小样本领域的各项工作，取得的成就，研究的多种方法，并介绍了未来的发展和研究难点，让我对小样本学习产生了浓厚的兴趣，以上内容纯属自己记录，如有不对请读者指出，如有同志欢迎一起积极探讨。
