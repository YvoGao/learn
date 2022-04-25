# Meta-Learning Adversarial Domain Adaptation Network for Few-Shot

小样本的文本分类主流研究方法

- 元学习
- 迁移学习

## 少样本文本分类

### 之前的方法

1. BERT： Pre-training of Deep Bidirectional Transformers for Language Understanding（2019）**微调参数的方法**
2. XLNet: Generalized Autoregressive Pretraining for Language Understanding（2019）**微调参数的方法**
3. Few-Shot Transfer Learning for Text Classification With Lightweight Word Embedding Based Models（2019）**改进的基于预训练词嵌入的层次化知识池策略**
4. Effective Few-Shot Classification with Transfer Learning（2020）**二进制分类器**

## 元学习对抗性领域自适应网络(MLADA)

将GAN的思想引入小样本学习，增加了元知识生成器和鉴别器结构，来提高小样本分类的效果。

### 网络结构

![](F:\DLUT\我爱敲代码\笔记\papper\png\MLADA.png)

**单词表示层（Word Representation Layer）：**使用预训练的*FastText.zip*来达到对单词进行embedding的作用。

**域判别器（Domain Discriminator）：**源域：支持集和查询集。构建一个三层前馈神经网络可以识别输入是否来自源域。

**元知识生成器（Meta-knowledge Generator）：**主要由双向LSTM和一个全连接层构成。利用双向神经网络多每一个时间步的内容进行embedding。

**交互层（Interaction Layer）：**可转移的特征：由元生成器生成的向量；句子特定的特征：单词的embedding。交互层用两种特征来生产输出，用来作为最后分类器的输入。

**分类器（Classifier）：**使用岭回归作为分类器。（1.使用神经网络，支持集太少不能充分训练；2.而且岭回归允许闭环解决方案，并且减少过拟合。）

### 训练过程

整个算法中引入了GAN的思想所以在训练过程中也有所不同。从模型结构有三个参数需要进行训练。

>- ${元知识生成器的参数：}\beta$
>- $ {域判别器参数：}\mu$
>- ${分类器参数：}\theta$

**训练步骤**

- Step1：在每一个情境中，我们首先固定生成器和判别器的参数，使用支持集，均方差来训练分类器参数。

  $$\mathcal L^{RR}(\theta)= \frac 1 {2m}\sum_{i=1}^m[((f_\theta(x^{(i)}-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2)]$$

  - m表示支持集样本数量
  - $f_\theta$表示岭回归
  - $\sum_{j=1}^n\theta_j^2$表示F-范数
  - $\lambda$表示正则化系数

- Step2：固定生成器和分类器的参数，使用查询集和源域，交叉熵损失来进行训练判别器的参数。

  $$\mathcal L^D(\mu) = - \frac 1 {2m}\sum_{i=1}^{2m}[y_d^{(i)}logD_\mu(k^{(i)})+(1-y_d^{(i)})log(1-D_\mu(k^{(i)}))] $$

  - m表示查询集或源域的样本数量
  - k表示元知识向量

- Step3：固定判别器和分类器的参数，利用查询集和源域，`最后分类结果的交叉熵损失和判别器的相反的损失`来更新生成器的参数。

  $$\mathcal L^G(\beta)=CELoss(f(W·G_\beta(W)),y)-\mathcal L^D$$

  - $G_\beta(W)$表示生成器生成的元知识特征向量
  - W表示词向量矩阵
  - $W·G_\beta(W)$表示句子特征和元知识特征的融合
  - $f(W·G_\beta(W))$表示岭回归结果
  - -表示判别器的相反损失，因为我们希望生成器得到的结果是鉴别器分不出来的，所以这个地方希望鉴别器的loss值变大。

*元知识生成器在所有训练集上都进行了优化，而分类器则是针对每个情景从头开始训练的。*这应该比较好理解，我们希望可以得到适应于广泛任务的元知识，所以利用所有的数据对元知识生成器进行提取，而针对不同情景，分类的分布可能是不同的，所以从头开始训练。

### 算法效果

![](F:\DLUT\我爱敲代码\笔记\papper\png\7d33035eb569fbe6f9415a72dc85035e_5_Table_2.png)

MLADA在这四个常见的数据集上都取得了最好的效果，可能是因为使用了100个情景的原因，而Reuters样本数最少所以在Reuters上效果特别好。

