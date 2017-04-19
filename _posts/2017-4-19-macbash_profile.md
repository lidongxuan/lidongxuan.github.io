---
layout: post
title: 环境配置：mac配置环境变量
comments: false
---

<!--more-->


这里说的是添加用户级环境变量。

（1）首先进入终端，然后用vim编辑.bash\_profile（注：Linux 里面是 .bashrc 而 Mac 是 .bash\_profile）

```
sudo vi ~/.bash_profile
```

（2）然后通过vim编辑器（当然也可以用其他编辑器如nano）添加或修改环境变量。关于vim操作的常见指令可以参考[这里]()。编辑完毕后在vim里输入
```
:wq
```
退出；

（3）最后再执行

```
source ~/.bash_profile
```

立即生效。

（4）通过```echo $PATH```指令可以查看之前的设置是否生效；通过```printenv```指令可以查看所有的环境变量。


### 参考资料 ###

[Mac OSX 添加环境变量的三种方法](http://yijiebuyi.com/blog/41ee3bab0c5bf1d43c7a8ccc7f0fe44e.html)