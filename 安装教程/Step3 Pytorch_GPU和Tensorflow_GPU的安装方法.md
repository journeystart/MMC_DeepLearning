详细安装方法参考文章：

Pytorch_CPU安装： [https://explinks.com/blog/ua-pytorch-cpu-version-installation-and-usage-guide/](https://explinks.com/blog/ua-pytorch-cpu-version-installation-and-usage-guide/)

Pytorch_GPU安装：[https://blog.csdn.net/qlkaicx/article/details/134577555](https://blog.csdn.net/qlkaicx/article/details/134577555)

Tensorflow_GPU安装：[https://blog.csdn.net/weixin_43412762/article/details/129824339](https://blog.csdn.net/weixin_43412762/article/details/129824339)

# 为防止链接失效，在此文字说明步骤
## 前提--查看是否有NVIDIV英伟达显卡
1. 控制面板 --> 设备管理器 --> 显示器配置 --> N开头的就是N卡

## 一、查看电脑的显卡驱动版本
1. 在**cmd命令窗口**中输入**nvidia-smi**，可以看到显卡版本号“CUDA Version：12.9”![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758848668579-5df3b0aa-3e4f-4268-b283-76154b426112.png)

## 二、安装CUDA
1. 在NVIDIA官网下载对应版本的CUDA版本（官网链接：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)），建议**安装的CUDA版本<=电脑显卡驱动版本。**
2. 选择“Windows --> x86_64 --> 当前电脑windows系统版本 --> exe(local)”
3. 可以安装在自定义目录下。
4. 安装过程一直点**继续**即可。
5. 检测是否安装成功：在命令窗口输入**nvcc --version**进行检查。

![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758848622371-551b0abc-198a-4c15-916f-1d1adb46031e.png)

## 三、安装cuDNN
1. 需要先注册一个CUDA的账号
2. 在官网上下载cuDNN（官网链接：[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)）
3. 选择“Windows --> x86_64 --> 10 --> exe(local)”
4. 下载好后的文件：![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758850223285-ba077057-1345-49f5-be9b-de6c0ba92a5b.png)
5. 将这些文件复制到CUDA的安装路径下，有重复的覆盖即可

![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758850307938-c45a9583-2628-48a8-8dcf-3e6d1c0cd296.png)

## 四、Pytorch_GPU安装
1. 进入Pytorch官网（官网网址：[https://pytorch.org/](https://pytorch.org/)）
2. 查看当前Pytorch版本支持的Python版本：![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758850826823-93c754de-0583-4b5d-a6d4-1513d29e59f1.png)
3. 创建虚拟环境（可以不创建，跳至第四步，推荐创建），使用Anaconda创建虚拟环境，在命令窗口输入**conda create -n pytorch-gpu python==3.11**。（格式为conda create -n 环境名称 python==版本号）  
	可以通过指令**conda env list**，查看当前计算机里创建的所有环境，其中带*标志的是当前激活的环境。  
	输入指令**conda activate 环境名**，即可切换到目标环境，同时命令行的前缀会显示环境名称。  
	如果要推出当前环境，输入**conda deactivate**命令。  
	删除创建的环境，输入**conda env remove -n 环境名称。**  
		 ![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758851628518-deae6543-69ec-48c3-aa5d-228e3c9ac312.png)
4. 下载Pytorch，在官网直接复制命令，在命令行窗口（选择的环境下）执行即可，若下载太慢可以加上镜像源下载![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758851853309-ffac6a16-7edd-4bec-a4f2-4b5990ddf36e.png)

## 五、测试Pytorch_GPU是否可用
```plain
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available()) #输出为True，则安装无误
```

![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758852221344-6b296d85-8ba3-4164-9daa-984d7f08850e.png)



# 总结
**PyTorch的GPU版本利用了NVIDIA的CUDA技术**，使得深度学习计算能够高效地在GPU上运行。使用GPU来执行深度学习计算可以显著加速计算，从而减少训练和推理时间。

**CUDA是NVIDIA推出的一种通用并行计算架构**，可以使GPU执行通用计算任务，而不仅仅是图形处理。在PyTorch中，可以使用CUDA来利用NVIDIA GPU的并行计算能力加速模型训练和推理。

**cuDNN是NVIDIA专门为深度学习模型设计的一个库**，它提供了高效的卷积操作和其他计算操作，可以进一步加速深度学习任务。**在PyTorch中使用cuDNN来优化深度学习模型的性能**。

总的来说，PyTorch的GPU版本通过与**NVIDIA的CUDA技术和cuDNN库的深度集成**，为深度学习研究和应用提供了强大、灵活且高效的计算能力。



## 因此Tensorflow的安装也需要安装CUDA和cuDNN，步骤相同，后面步骤详见CSDN
