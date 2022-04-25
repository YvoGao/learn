

@[toc](小样本学习记录————MAML用于深度网络快速适应的模型不可知元学习)

# 相关概念
## 小样本学习（Few-Shot Learning）
>Few-Shot Learning(FSL) is a type of machine learning problems (specified by E, T and P), where E contains only a limited number of examples with supervised information for the target T.  

简单来说就是使用少量样本数据进行训练完成目标任务的一种机器学习方法。具体有关小样本学习的介绍可以看我的上一篇博客[https://blog.csdn.net/yunlong_G/article/details/121570804](https://blog.csdn.net/yunlong_G/article/details/121570804)

**N-way K-shot**

这是小样本学习中常用的数据，用以描述一个任务：它包含N个分类，每个分类只有K张图片。K越小，N越大越难实现。

**Support set and Query set**

Support set指的是参考集，Query set指的是查询集。其实就是训练集和测试集。

**eg:** 
>用人识别动物种类，有5种不同的动物，每种动物2张图片，总计10张图片给人做参考。另外给出5张动物图片，让人去判断各自属于那一种类。那么10张作为参考的图片就称为Support set，5张要分类的图片就称为Query set。这可以说是一个5-way-2-shot的任务。





## 元学习（Meta-Learning）
>Meta-learning, also known as“learning to learn”, refers to improving the learning ability of a model through multiple training episodes so that it can learn new tasks or adapt to new environments quickly with a few training examples.

元学习又称“学会学习”，是指通过多次训练来提高模型的学习能力，使其能够通过几个训练实例快速学习新任务或适应新环境。现有的方法主要分为两类：
- (1)基于优化的方法，包括开发一个元学习器作为优化器，直接为每个学习者输出搜索步骤，以及学习模型参数的优化初始化，这些参数稍后可以通过几个梯度下降步骤来适应新任务
- (2)基于度量的方法，包括Matching Network、PROTO、Relation Network、TapNet和Induction Network，旨在学习适当的距离度量，以将验证点与训练点进行比较，并通过匹配训练点进行预测。

现在小样本学习的主流方法主要是基于元学习或者迁移学习的，而MAML是元学习中特别经典的一篇论文，所以在此将自己的阅读收获分享给大家。

# MAML思想
**算法目标：** 是一个模型可以经过比较少的训练快速迭代到最好的效果。
>为了达到这一目的，模型需要大量的先验知识来不停修正初始化参数，使其能够适应不同种类的数据。这里需要借助李宏毅老师课堂的PPT图来理解一下MAML和预训练的区别。
********
我们定义初始化参数为 $\phi$，其初始化参数为 $\phi_0$ ，定义在第n个测试任务上训练之后的模型参数为 ${\hat{\theta}}^n$ ，于是总的损失函数为 $L(\phi)=\sum_{n=1}^Nl^n( \hat{\theta}^n )$ 。pre-training的损失函数是$L(\phi)=\sum_{n=1}^Nl^n(\phi)$，直观上理解是MAML所评测的损失是在任务训练之后的测试loss，而pre-training是直接在原有基础上求损失没有经过训练。
![在这里插入图片描述](https://img-blog.csdnimg.cn/abc11fa38cff4c51b8ee81853e5aa33c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LqR5rqq6b6Z,size_20,color_FFFFFF,t_70,g_se,x_16)
用论文中图片来说就是找到一个$\phi$，在训练后可以让所有任务上loss都能下降到最优。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e6a135e2b8444718a861db57a46e5857.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LqR5rqq6b6Z,size_20,color_FFFFFF,t_70,g_se,x_16)
而model pre-training的初衷是寻找一个从一开始就让所有任务的损失之和处于最小状态 $\phi$，它并不保证所有任务都能训练到最好的${\hat{\theta}}^n$ ，如上图所示， loss可能会收敛到局部最优。

## MAML算法
$P(T)$用来表示任务的分布，$\beta，\alpha$是训练的超参数，表示子任务内的学习率和任务间的学习率，$f_\theta$表示训练的模型。
1.	随机初始化模型参数$\theta$，这个$\theta$就是前文李宏毅老师所讲的$\phi_0$其实。
2.	每一次训练从中提取一个子任务$T_i$。
3.	在$T_i$任务里，我们使用公式$\theta'_i=\theta - \alpha\nabla_\theta L_{T_i}(f\theta)$ 来更新任务内的$\theta'_i$参数。这个地方就是利用loss函数对于$\theta$的梯度更新。
4.	当$T_i$任务训练完之后，就在根据公式$\theta'_i=\theta - \beta\nabla_\theta \sum_{T_i-P(T)}L_{T_i}(f\theta'_i)$利用在所有任务上的loss和的梯度更新最后的$\theta$。
5.直到所有的子任务都训练完毕，得到最后的$\theta$。

![在这里插入图片描述](https://img-blog.csdnimg.cn/da7c487477e34328b5f13ceca9537446.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LqR5rqq6b6Z,size_20,color_FFFFFF,t_70,g_se,x_16)
## 论文代码
论文有两个实验例子，一个是监督学习中的应用，一个是强化学习应用的例子，本人只是简单跑了一下监督学习的例子。作者仓库[https://github.com/cbfinn/maml](https://github.com/cbfinn/maml)，要运行作者代码也很简单，在readme和main.py有明确的说明，只要耐心阅读即可。最后得到可运行项目文件结构如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/a016280ef8514dc590d0fe9f50372130.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LqR5rqq6b6Z,size_20,color_FFFFFF,t_70,g_se,x_16)
data下的两个文件是对数据的处理得到我们希望的训练集格式，utils.py帮助定义了卷积层，池化层，loss等模块，data_generator.py定义了如何获得数据，数据如何分配传到训练中，main.py是启动整个程序的入口。这里主要看一下maml.py。

```python
""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        ## 表明construct_model是这篇代码的核心部分
        #初始化维度，超参数
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        #论文有三个数据集所以这个地方分三种情况讨论
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        else:
            raise ValueError('Unrecognized data source.')
	
	# 构建训练图的整个过程
    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
        # 第一次需要初始化weight，后续共享
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            # 用于记录每一个任务每一次训练的各种指标
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates
                    
            # 元任务的训练过程
            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                # inputa 是训练集，inputb 是 task 的测试集
                inputa, inputb, labela, labelb = inp
                # task_outputbs 是每次梯度更新后参数在 inputb 数据上的输出，task_lossesb 是基于每个 task_outputb 计算出的 loss
                task_outputbs, task_lossesb = [], []

                # 对于分类任务增加准确率指标，是每次梯度更新后参数在 inputb 数据上的输出的准确率
                if self.classification:
                    task_accuraciesb = []

                # task_outputa 是第一次前向计算在 inputa 数据的输出，task_lossa 是基于 task_outputa 在参数 weight 上计算的 loss
                # 就是每一个元任务都用上一个任务得到的参数继续训练
                # 为什么记录一下第一次的loss，这个是想和pre-training进行一次对比
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                # 第一次的内部梯度更新
                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                # 元任务内部的梯度跟新，因为上面已经进行了一次更新，所以这里是num_updates - 1次
                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))   
                
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # 定义输出类型，不同任务输出格式也不一样
            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        # 元任务间的梯度跟新过程
        if 'train' in prefix:
            # lossesa 是 meta_batch_size 个具体任务在 inputa 数据上的第一次前向的 loss，
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            # lossesb[j] 是第 j 次更新时，meta_batch_size 个任务在 inputb 数据上的 loss
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # pretrain 使用 inputa 数据上的第一次 loss 和，pretrain 相当于迁移学习的预训练
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                # metatrain_op 最小化目标是每个 task 最后一次前向计算出的 loss 的平均值
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            # 验证过程
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        # 对预训练和MAML的数据记录
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    # 全连接层的构建
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    # 全连接层前向传播
    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    # 卷积层构建（5层）
    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    # 卷积层的前向传播
    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

```
以上是对整篇文章的阅读理解和代码理解，有一些解释不清楚，理解错误的地方垦请读者批评指正。