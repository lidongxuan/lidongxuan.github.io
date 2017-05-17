---
layout: post
title: 目标跟踪：Meanshift, Camshift
comments: false
---

<!--more-->

![1](/public/images/2017-4-8-meanshift/1.png)

Meanshift/Camshift是我最早接触的目标跟踪算法，现在稍作总结。Meanshift其实关键的部分就两个，一个是mean，也就是求质心（也可以叫加权平均值），二是shift，即向质心方向移动。在目标跟踪领域，meanshift其实是对单帧图片内迭代搜索目标位置的算法，而Camshift是对整个视频序列的连续处理，而且相比与meanshift对目标的尺度变化和旋转变化有鲁棒适应性。这里主要讲解Meanshift/Camshift在跟踪领域的内容。

### 一. HSV ###

RGB空间对于大家来说都比较熟悉，但是RGB对光照的变化比较敏感，这种敏感对目标跟踪而言是不利的。而HSV(Hue,Saturation,Value)是根据颜色的直观特性（色调H，饱和度S，亮度V）创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。

![2](/public/images/2017-4-8-meanshift/2.png)

### 二. 反向投影图 ###

Meanshift之所以能应用到目标跟踪领域，是因为根据初始**目标box**和初始**全图**能形成反向投影图，反向投影图的每个像素的值体现的是该像素是被跟踪目标的概率。那么反向投影图是如何形成的呢？

第一步是将原图片的所有像素从RGB空间映射到HSV空间。

然后，统计**目标box**区域内H（色调）分量的直方图，横坐标是色调变量（0～360°），纵坐标是该色调值下的像素个数，最后在对其进行归一化，使得该直方图成为概率直方图。

最后，将全图的HSV转换为反向投影图，比如（x=1，y=1）位置的H分量是40°，那么就将该值替换为上一步概率直方图中该值对应的概率，以此类推，于是反向投影图的每个像素的值体现了该像素是被跟踪目标的概率。

至此，将反向投影图作为meanshift的输入就很好理解了。

### 三. Meanshift ###

meanshift的流程如本文最上面的那个图所示，本质上就是求均值然后漂移到均值的位置。下面给出具体流程（注意这是meanshift在单帧图片内的迭代过程）：

（1）根据初始目标box的大小的位置确定meanshift搜索窗口的大小和位置（对应图中的那个圆），人们把搜索窗口的尺寸称为核窗口，那么核函数就类似一个脉冲函数，窗口内的值是1，窗口外的是0。好多其他算法中的核函数，差不多也是这么个意思，我感觉核函数和加权函数或加权映射的意义差不多

（2）计算搜索窗口内的质心，或叫加权平均位置。求法很简单，就是一阶距除以零阶距，具体见第一个参考资料

（3）求得的质心作为新的搜索窗口的中心

（4）重复第二步和第三步，直到跟踪窗口中心和质心“会聚”，即每次窗口移动的距离小于一定的阈值。

### 四. Camshift ###

由于Meanshift在跟踪中搜索框的大小一直不变，对目标的尺度变化不具有鲁棒性，Camshift的出现改进了这方面的不足。CamShift，即Continuously Adaptive Mean-Shift算法（连续自适应的Meanshift），利用**不变矩**对目标的尺寸进行估算，实现了连续自适应地调整跟踪窗口的大小和位置。

论文Computer Vision Face Tracking For Use in a Perceptual User Interface给出了估计目标大小和角度的公式，但对公式的具体详解没有给出。

关于尺度变化我们可以这样理解，假设连续两帧图片中的目标大小不同，位置不同，第一帧的目标跟踪box位置和大小都是已知的，那么我们可以根据meanshift迭代计算出第二帧的位置信息。接下来就是计算大小信息：由于**不变矩**的存在，这两者可以计算出相同的二阶矩，这样，我们可以就可以根据第一个图片中目标的位置与大小信息求得相关二阶矩，然后代入第二帧的位置信息，逆推出第二帧的大小信息。这样，就实现了对目标大小变化的适应性。具体的计算公式如下图（从原论文摘取）：

![3](/public/images/2017-4-8-meanshift/3.png)

### 参考资料 ###

[借助图像直方图来检测特定物(MeanShift、CamShift算法) 方向梯度直方图](http://www.jianshu.com/p/436743802642)

[Computer Vision Face Tracking For Use in a Perceptual User Interface](http://d1.amobbs.com/bbs_upload782111/files_37/ourdev_624458GGCHRU.pdf)

[meanshift与camshift跟踪研究，有代码](http://blog.csdn.net/jhh_move_on/article/details/36932239)

[遮挡情况下的目标跟踪技术研究，有不变矩的相关说明](http://d.wanfangdata.com.cn/Thesis/Y2659358)
