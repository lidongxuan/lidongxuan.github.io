---
layout: post
title: 编程相关：在mac上配合Dash使用sublime
comments: false
---

<!--more-->

### 关于Dash ###

[Dash](https://kapeli.com/dash)是一个mac端（ISO也有提供）上的API文档快速浏览器，Dash提供了很多常用的API，还有很多其他开发者提供的文档。有了它，开发者查阅文档的效率大大提升。
Dash的Integration里面提供了很多到第三方开发应用的文档查询链接。最吸引我的是Alfred和Sublime Text。

![path1](/public/images/2016-11-30-dash-sublime/1.png)

### Dash + Sublime Text3 ###

首先下载安装好Dash和Sublime。

将Dash和Sublime整合的方法如下：

1.安装[Sublime Package Control](https://packagecontrol.io/installation)

2.安装完成后重启Sublime，使用cmd+shift+p调出Sublime管理器搜索框，然后在输入install，找到install Package选项，回车。

3.回车完再输入Dashdoc，找到DashDoc选项，选中后回车

4.安装完毕，在Preferences->Package Settings里可以看到DashDoc已经安装

### 开始使用 ###

在代码选中你想查询的语句，再按下control+h，便可弹出文档

### 参考资料 ###

[http://www.tuicool.com/articles/MBNJRb](http://www.tuicool.com/articles/MBNJRb)