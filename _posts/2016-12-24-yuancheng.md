---
layout: post
title: 相关技能：ssh远程终端、VNC远程桌面、挂载远程目录
comments: false
---

<!--more-->

### 一.ssh远程终端 ###

（1）通过ssh进入远程服务器指令窗口

```
ssh username@ip地址
```

如:

```
ssh scs48xx@10.106.20.xxx
```

（2）然后再输入该用户的密码即可

（3）不用的时候输出输入```logout```指令登出即可

### 二.VNC远程桌面 ###

（1）首先保证服务器安装好了vncserver，ssh等服务组件

（2）首先通过ssh进入远程服务器指令窗口，参考上一部分的“**ssh远程终端**”

（3）建立大小为1440x900的远程桌面尺寸，编号为4（也可以是其他编号），如果被占用，那就再新建一个编号

```
vncserver -geometry 1440x900 :4
```

（4）接下来在个人笔记本的vnc viewer上登录这个远程桌面，

VNC Server那一栏填写：IP地址:编号

如下图：

![1](/public/images/2016-12-24-yuancheng/1.png)

（5）如果你想关掉这个VNC服务那就在服务器端执行下面的指令：

```
# 4是代表要kill的桌面的编号
vncserver -kill :4
```

注1：如果服务器关机，那么vnc服务会自动关闭，要使用再执行第（2）步

注2：如果想实现远程和本机公用剪切板，实现复制粘贴功能，可以输入以下指令：

```
vncconfig &
```

然后会弹出一个小窗口，全部打勾，然后就可以共享剪切板了，不想共享的时候把这个小窗口关闭即可。

### 三.挂载远程目录 ###

（1）首先在Mac上安装两个工具，可以通过homebrew安装：

```
brew install Caskroom/cask/osxfuse
brew install homebrew/fuse/sshfs
```

如果homebrew不成功，可以访问[http://osxfuse.github.io](http://osxfuse.github.io) 手动下载以下两个pkg

![1](/public/images/2016-12-24-yuancheng/2.png)

然后手动安装

（2）挂载相关目录至本地目录

```
sshfs username@服务器IP:要挂载的目录 本地目标目录
```
如：

```
sshfs scs4850@10.106.20.13:/home/scs4850/lidongxuan/ ~/linux/
```
 (3)使用结束后可以通过以下指令取消挂载，也可以直接在mac上右键推出
 
```
umount ~/linux/
```
