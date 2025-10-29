# Git提交更新步骤

## 提交过一次之后，可以从Step4开始操作，前三步已经保存了

## Step 1 下载git工具

**下载地址：[Git - Install for Windows**](https://git-scm.com/install/windows)

## Step 2 创建本地仓库

**在需要提交的文件夹中右键 --> 选择 git bash here --> 在弹出的命令框内输入 git init**

## Step 3 链接云端库

在github网站上复制库的地址，https://github.com/journeystart/MMC_DeepLearning.git

**在命令窗口输入 git remote add origin https://github.com/journeystart/MMC_DeepLearning.git**

若发现已经存在链接，通过指令 git remote rm origin 删除链接

可以使用指令 git remote -v 查看链接的云端主机

## Step 4 将需要推送的文件放入本地库

**在命令窗口输入 git add .**   表示将文件加入本地库暂存区

可以使用指令 git status 查看所有文件的状态

**在命令窗口输入 git commit -m "本次修改说明"** 表示将改动记录存在本地库中

’:wq'退出vim查看

## Step 5 将文件推送到云端

**在命令窗口输入 git pull** 表示从云端拉取最新的同步

‘:wq'退出vim查看，直接退出表示接受合并

**在命令窗口输入 git push -u origin master** 上传文档

