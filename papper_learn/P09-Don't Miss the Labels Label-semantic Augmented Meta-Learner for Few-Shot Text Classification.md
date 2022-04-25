# Don't Miss the Labels Label-semantic Augmented Meta-Learner for Few-Shot Text Classification

> 本篇文章的主要思想是，利用训练集中标签信息来提高模型的特征提取能力。


![](F:\DLUT\我爱敲代码\笔记\papper\png\LSAML1.png)

在训练的过程中使用，将句子的输入从[CLS]句子[SEP]，变成[CLS]句子[SEP]类名[SEP]。将下一句的变成标签，作者将这种方法称为标签语义增强特征提取(Label-语义AugededFeature Extraction)

> BERT 的输入可以包含一个句子对 (句子 A 和句子 B)，也可以是单个句子。此外还增加了一些有特殊作用的标志位：
>
> - [CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
> - [SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
> - [UNK]标志指的是未知字符
> - [MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。

![](F:\DLUT\我爱敲代码\笔记\papper\png\LSAML2.png)

*GAP是全局平均池化操作*



