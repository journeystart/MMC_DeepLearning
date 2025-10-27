# N83058：基于学习的图像编码现状报告
<font style="color:rgba(0, 0, 0, 0.85);">尽管该领域已取得一定进展，但这类编码方案仍面临诸多挑战，例如：</font>**<font style="color:rgba(0, 0, 0, 0.85);">哪种编码器 / 解码器架构更具前景（如循环神经网络 vs 变分自动编码器）、应采用哪种处理层对图像数据进行压缩、优化过程中应使用哪种质量评价指标。此外，该领域其他相关问题还包括：如何在有损图像编码中最小化量化带来的影响、应采用何种学习策略（如生成对抗网络中使用的双神经网络架构）、以及如何实现码率分配。</font>**

## <font style="color:rgba(0, 0, 0, 0.85);">一、基于学习的图像编码方案</font>
### 方案整理
+ Variable Rate Image Compression with Recurrent Neural Networks [1]: 
+ Full Resolution Image Compression with Recurrent Neural Networks [2]: 
+ Lossy Image Compression with Compressive Autoencoders [3]: 
+ End-to-end Optimized Image Compression [4]:
+ Variational Image Compression With a Scale Hyperprior [5]: 
+ Real-Time Adaptive Image Compression [6]:
+ Generative Adversarial Networks for Extreme Learned Image Compression [7]:

文献中部分研究并非提出完整的图像编码器，而是聚焦于改进现有图像压缩编码器的不同方面（主要是工具模块）。这类研究成果可根据图像压缩方案的不同组成部分分为以下几类：

+ 量化：

<font style="color:rgb(0, 0, 0);">在文献 [10] 中，研究者未针对每个量化步长（进而对应不同图像质量）单独学习一个模型，而是提出学习单一模型，随后采用不同的量化策略（如自适应（且可学习的）通道量化步长）。</font>

<font style="color:rgb(0, 0, 0);">文献 [11] 采用了一种不同的量化方法，通过对 latent 码进行激进剪枝以增强稀疏性。</font>

<font style="color:rgb(0, 0, 0);">文献 [12] 提出了一种非均匀量化方法，该方法根据特征分布进行优化；通过非均匀量化器对编码器 - 解码器网络进行迭代微调，以实现最佳性能。</font>

<font style="color:rgb(0, 0, 0);">文献 [13] 提出了一种统一的端到端学习框架，可联合优化模型参数、量化水平以及最终符号流的熵。该方法依赖于将数据 “软分配” 到量化水平的机制，且量化水平与模型参数会联合学习。</font>

+ 非线性变换：

<font style="color:rgb(0, 0, 0);">文献 [14] 提出了一种假设采用标量量化的非线性变换效率端到端优化框架，该框架可支持任意感知评价指标。通过该框架，能够为广义除法归一化变换及其对应的逆变换找到一组参数。</font>

<font style="color:rgb(0, 0, 0);">在文献 [14] 的基础上，文献 [15] 对这种优化后的变换进行了分析，结果表明：与其他方案相比，该变换能降低变换后分量之间的相关性。</font>

<font style="color:rgb(0, 0, 0);">文献 [16] 从寻找参数（训练算法）的角度分析了这些变换的可逆性，目的是最小化损失函数，进而确保最大的重建保真度。</font>

+ 熵编码：

<font style="color:rgb(0, 0, 0);">文献 [17] 的核心思想是利用条件概率模型捕捉 latent 表示的统计特性。在该方案中，采用了带有 3D 卷积神经网络（3D-CNN）的自动编码器架构进行上下文建模。在训练过程中，会迭代更新上下文模型，以学习 latent 表示各元素之间的依赖关系。</font>

<font style="color:rgb(0, 0, 0);">文献 [18] 提出了空间局部自适应熵编码模型，该模型在编码器与解码器之间共享，且会传输某种辅助信息。这使得熵编码模型能够适配特定图像的特征。</font>

<font style="color:rgb(0, 0, 0);">文献 [19] 提出了一种上下文自适应熵模型框架，该框架利用多种类型的上下文信息。通过选择并使用两种不同类型的上下文，可更准确地获取 latent 码元素分布的概率，从而高效利用空间相关性。</font>

### <font style="color:rgb(0, 0, 0);">分类体系</font>
确定一下分类维度为该分类体系的核心维度：

① 神经网络类型（Neural Network Type）

② 编码单元的大小（Coding Unit Size）

③ 空间相关性工具（Spatial Correlation Tools）

④ 码率控制策略（Bitrate Control Strategy）

⑤ 适用质量范围（Quality Range）

![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758701737797-f7ebb55f-f647-4c9c-b1b9-414b5f355b34.png)

## 二、公开可用的推昂编码器软件实现
![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758701772471-264d4f70-03d1-408e-a655-c2a45384be50.png)

## 三、可用于探索性研究的数据集
![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758701817577-0b59ec6d-84cf-4bed-a293-f8cee17b77cd.png)

![](https://cdn.nlark.com/yuque/0/2025/png/49665485/1758701849715-f4f947a8-6122-4845-bf8c-a387479ecb8e.png)

## 参考文献
[1] G. Toderici, S.M. O’Malley, S.J. Hwang, D. Vincent, D. Minnen, S. Baluja, M. Covell, R. Sukthankar, “Variable Rate Image Compression with Recurrent Neural Networks,” International Conference on Learning Representations, San Juan, Puerto Rico, May 2016.  
[2] G. Toderici, D. Vincent, N. Johnston, S.J. Hwang, D. Minnen, J. Shor, M. Covell, “Full Resolution Image Compression with Recurrent Neural Networks,” IEEE Conference on Computer Vision and Pattern Recognition, Aug. 2016.  
[3] L. Theis, W. Shi, A. Cunningham, F. Huszár, “Lossy Image Compression with Compressive Autoencoders,” International Conference on Learning Representations, Toulon, France, April 2017.  
[4] J. Ballé, V. Laparra, E. P. Simoncelli, “End-to-end Optimized Image Compression,” International Conference on Learning Representations, Toulon, France, April 2017.  
[5] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston, “Variational Image Compression With a Scale Hyperprior,” International Conference on Learning Representations, Vancouver, Canada, April 2018.  
[6] O. Rippel and L. Bourdev, “Real-Time Adaptive Image Compression,” International Conference on Machine Learning, Sydney, Australia, May 2017.  
[7] E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, L. Van Gool, “Generative Adversarial Networks for Extreme Learned Image Compression,” arXiv:1804.02958, October 2018.  
[8] A. Prakash, N. Moran, S. Garber, A. Dilillo, J. Storer, “Semantic Perceptual Image Compression Using Deep Convolution Networks,” Data Compression Conference, Salt Lake City, USA, March 2017.  
[9] M. Li, W. Zuo, S. Gu, D. Zhao, D. Zhang, “Learning Convolutional Networks for Content-weighted Image Compression,” IEEE International Conference on Computer Vision and Pattern Recognition, Salt Lake City, USA, June 2018.  
[10] T. Dumas, A. Roumy, C. Guillemot, “Autoencoder based Image Compression: can the Learning be Quantization Independent?,” IEEE International Conference on Acoustics, Speech and Signal Processing, Calgary, Canada, April 

[11] H. Zhao, P. Liao, “CAE-ADMM: Implicit Bitrate Optimization via ADMM-Based Pruning in Compressive Autoencoders”, [https://export.arxiv.org/abs/1901.07196](https://export.arxiv.org/abs/1901.07196)  
[12] J. Cai and L. Zhang, “Deep Image Compression With Iterative Non-Uniform Quantization,” IEEE InternatioonalConference on Image Processing, Athens, Greece, October 2018.  
[13] E. Agustsson, F. Mentzer, M. Tschannen, L. Cavigelli, R. Timofte, L. Benini, L. V. Gool, “Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations,” Neural Information Processing Systems, Long Beach, USA, December 2017.  
[14] J. Ballé, V. Laparra, E. P. Simoncelli, “End-to-end Optimization of Nonlinear Transform Codes for Perceptual Quality,” Picture Coding Symposium, Nuremberg, Germany, December 2016.  
[15] J. Ballé, V. Laparra, E. P. Simoncelli, “Density Modeling Of Images using a Generalized Normalization Transformation,” International Conference on Learning Representations, Toulon, France, April 2017.  
[16] J. Ballé, “Efficient Nonlinear Transforms for Lossy Image Compression,” Picture Coding Symposium, San Francisco, USA, June 2018.  
[17] F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, L. Van Gool, “Conditional Probability Models for Deep Image Compression,” IEEE International Conference on Computer Vision and Pattern Recognition, Salt Lake City, USA, June 2018.  
[18] D. Minnen, G. Toderici, S. Singh, S. J. Hwang, M. Covell, “Image-Dependent Local Entropy Models for Learned Image Compression,” International Conference on Image Processing, Athens, Greece, October 2018.  
[19] J. Lee, S. Cho, S.-K. Beack, “Context-adaptive Entropy Model for End-to-end Optimized Image Compression”, International Conference on Learning Representations, New Orleans, USA, May 2019.  
[20] M. Tschannen, E. Agustsson, M. Lucic, “Deep Generative Models for Distribution-Preserving Lossy Compression”, Advances in Neural Information Processing Systems 31, Montreal, Canada, September 2018.  
[21] T. Dumas, A. Roumy, C. Guillemot, “Image Compression With Stochastic Winner-Take-All Auto-Encoder,” IEEE International Conference on Acoustics, Speech and Signal Processing, New Orleans, USA, March 2017.  
[22] M. Akbari, J. Liang, J. Han, “DSSLIC: Deep Semantic Segmentation-based Layered Image Compression,” IEEE International Conference on Acoustics, Speech and Signal Processing, Brighton, UK, May 2019.  
[23] D. Minnen, J. Ballé, G. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” Advances in Neural Information Processing Systems, no. 31, 2018.  
[24] L. Zhou, C. Cai, Y. Gao, S. Su, J. Wu, “Variational Autoencoder for Low Bit-rate Image Compression,” CVPR Workshop and Challenge On Learned Image Compression, Salt Lake City, June 2018.  
[25] M. H. Baig, V. Koltun, L. Torresani, “Learning to Inpaint for Image Compression,” Advances in Neural Information Processing Systems, Long Beach, CA, USA.  
[26] S. Luo, Y. Yang, Y. Yin, C. Shen, Y. Zhao, M. Song, “DeepSIC: Deep Semantic Image Compression,” International Conference on Neural Information Processing, Siem Reap, Cambodia, December 2018.  
[27] A. Mousavi, G. Dasarathy, R. G. Baraniuk, “DeepCodec: Adaptive Sensing and Recovery via Deep Convolutional Neural Networks,” Annual Allerton Conference on Communication, Control, and Computing, Illinois, USA, October 2017.  
[28] N. Johnston, D. Vincent, D. Minnen, M. Covell, S. Singh, T. Chinen, S. J. Hwang, J. Shor, G. Toderici, “Improved Lossy Image Compression with Priming and Spatially Adaptive Bit Rates for Recurrent Networks,” International Conference on Computer Vision and Pattern Recognition, Salt Lake City, USA, June 2018.  
[29] M. Covell, N. Johnston, D. Minnen, S. J. Hwang, J. Shor, S. Singh, D. Vincent, G. Toderici, “Target-Quality Image Compression with Recurrent, Convolutional Neural Networks,” arXiv:1705.06687, May 2017.  
[30] D. Minnen, G. Toderici, M. Covell, T. Chinen, N. Johnston, J. Shor, S. J. Hwang, D. Vincent, S. Singh, “Spatially Adaptive Image Compression using a Tiled Deep Network,” International Conference on Image Processing, Beijng, China, September 2017.  
[31] F. Jiang, W. Tao, S. Liu, J. Ren, X. Guo, D. Zhao, “An End-To-End Compression Framework Based on Convolutional Neural Networks,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 28, no. 10, August 2017.  
[32] Z. Cheng, H. Sun, M. Takeuchi, J. Katto, “Deep Convolutional AutoEncoder-based Lossy Image Compression,”Picture Coding Symposium, San Francisco, USA, June 2018.  
[33] A. G. Ororbia, A. Mali, J. Wu, S. O’Connell, D. Miller, C. L. Giles, “Learned Iterative Decoding for Lossy Image Compression Systems,” arXiv:1803.05863, November 2018.  
[34] F. Hussain and J. Jeong, “Efficient Deep Neural Network for Digital Image Compression Employing Rectified Linear Neurons,” Journal of Sensors, vol. 2016, no. 3184840, 2016.  
[35] S. Santurkar, D. Budden, N. Shavit, “Generative Compression,” Picture Coding Symposium, San Francisco, USA, June 2018.  
[36] Z. Cheng, H. Sun, M. Takeuchi, J. Katto, “Performance Comparison of Convolutional AutoEncoders, Generative Adversarial Networks and Super-Resolution for Image Compression,”, CVPR Workshop and Challenge On Learned Image Compression, Salt Lake City, June 2018.  
[37] C. Aytekin, X. Ni, F. Cricri, J. Lainema, E. Aksu, M. Hannuksela, “Block-optimized Variable Bit Rate Neural Image Compression,” CVPR Workshop and Challenge On Learned Image Compression, Salt Lake City, June 2018.

