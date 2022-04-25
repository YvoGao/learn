查看python环境

```bash
import sys 
sys.executable 
```

下载anconada

````bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.07-Linux-x86_64.sh
````

安装

```bash
bash Anaconda3-2020.07-Linux-x86_64.sh
```

激活环境变量

```bash
source ~/.bashrc
```



在实验室一组服务器上使用，为了保证各个用户的环境不冲突，默认不激活conda，使用以下方式

```
export PATH="/$HOME/anaconda3/bin:$PATH"
export PATH="$PATH:$HOME/anaconda/bin"
```



创建环境

```
conda create -n tensorflow python=3.6
```

激活

```
conda info --envs
source activate tensorflow
python
```

退出环境

```
source deactivate
```



pycharm远程连接

[深度学习记录————服务器python虚拟环境配置+pycharm远程连接_yunlong_G的博客-CSDN博客](https://blog.csdn.net/yunlong_G/article/details/121532581)



测试代码

```
import tensorflow as tf
tf.compat.v1.disable_eager_execution()#保证sess.run()能够正常运行
hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session()#版本2.0的函数
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())

```

