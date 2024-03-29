# Few-shot Text Classification with Distributional Signatures

在计算机视觉中，低水平的模式是可以跨学习任务迁移的。然而，直接将这一方法应用于文本是具有挑战性的-词汇特征对一项任务具有很高的信息量，对另一项任务可能无关紧要。本文的模型不仅从单词中学习，还利用了它们的分布签名，这些签名编码了相关的单词并发模式。我们的模型在元学习框架内进行训练，将这些签名映射为注意力分数，然后用这些分数对单词的词汇表示进行加权。

## 网络结构

### 注意力生成器

关注单词统计的函数，根据大的资源库来统计一般词的重要性，利用支持集来估计特定词的重要性，生成的注意力机制构建下游分类的表示。

- 根据文献中的记载，频繁出现的词不太可能是信息性的，（因为这里the会比较容易出现），所以要降低频繁词的权重，增加稀有词的权重，本文选用了Arora的一种既定的方法。

  $$s(x_i):=\frac \epsilon {\epsilon + P（x_i)}$$

  > 这里$\epsilon是10^{-3}，x_i是第i个单词，P（x_i)是x_i在源域的重要性。$

- 支持集中有区别的词在查询集中也有可能有区别。所以定义了所以如下的统计数据来反应单词重要性。

  $$t(x_i):=\mathcal H(P(y|x_i))^{-1}$$

  > $(P(y|x_i))是支持集上的极大似然，\mathcal H(.)是熵算子，t(.)根据频率分布高度加权。$

- 直接使用统计量可能不是效果很好原因如下

  - 两个统计量可能互补，不清楚如何结合。

  - 这些统计量对于分类来说单词重要性噪声近似。使用双向LSTM融合输入信息，使用点积注意力来预测单词$x_i$的注意力分数。

    $\alpha_i :=\frac {exp(v^Th_i)} {\sum_jexp(v^Th_j)}$

    $h_i$是i处的双向LSTM输出，v是可学习的向量。

### 岭回归

在注意生成器下，岭回归者在看了几个例子后很快就学会了做出预测。首先，对于给定情节中的每个例子，我们构建一个词汇表征，重点放在重要的单词上，由注意力分数表示。下一步，给出这些词汇表示，我们从零开始训练支持集上的岭回归。最后，我们进行预测。

- 构建表征：根据不同的词的重要程度进行词汇表征。

  $$\phi(x):=\sum_i\alpha·f_{ebd}(x_i)$$

  > $f_{ebd}(·)$是预训练后对x的嵌入表示。

- 从支持集训练：岭回归允许模型进行端到端的闭合解可以减少过拟合。

  $$W = \phi^T_S(\phi_S\phi_S^T+\lambda I)^-1Y_S$$

  > I是定义的矩阵，W是权重矩阵，$\lambda$是正则化系数。

- 在查询集的推理：使用$\hat Y_Q$推理$\hat P_Q$，使用交叉熵损失类更新参数。

  $\hat Y_Q = \alpha\phi_QW + b$

### 理论分析

- 为了针对提高输入扰动的鲁棒性，设（P, S, Q)集合，P是原集，Q是查询集， S是支持集，对于任何S和Q的交集，注意力生成器会产生词的重要性。

  $$\alpha = AttGen(x|S,P)$$

  因为重要的单词可能会被常见的单词替换（如the，a经常出现），所以我们使用$\sigma(w)$表示单词的扰动，使用$P(W)=P(\sigma(W))$来表示w，这样可以使单词映射到相同的特征空间。


## 训练过程

了解完这篇文章的网络结构，下面根据几副来好好理解一下训练过程

### 元训练

- 在元训练的每个episode中。首先从训练集拿出N个类样本，在从N类样本中进行支持集和查询集的抽取，在这一episode中将剩下的样本作为源池库。
- 元测试中，从测试集抽取支持集和查询集，将训练集作为源池库。

> 源池库的作用是为了让对查询集分布更加的合理，减少过拟合。

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/8844e22d053582755e9f325f1930badb_2_Figure_3.png)

在每一个episode中，可以将数据集分成以下的部分。

![](F:\DLUT\我爱敲代码\笔记\papper\png\ds.2.png)

文章的整个思路根据下图再进行一次梳理就显得很清楚了，

- 首先根据支持集和源池库中的样本中单词分布通过双向LSTM得到注意力分数$\alpha$，
- $\alpha$和FastText进行从词嵌入相乘构造特征向量$\phi(x)$。
- 在支持集中训练，利用$Y_S$和对支持集的特征向量$\phi(x)$得到W参数。
- 最后利用得到的W和对查询集进行特征表征后得到的$\phi(x)$计算查询集的概率值，计算误差进行参数更新。

![fig-metalearning](E:\code_git\few-shot\Distributional-Signatures-master\assets\model.png)

