## cs231n_assignments_2021

参考：Lectures：https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk

Notes：https://cs231n.github.io/

Slides：http://cs231n.stanford.edu/slides/2021/

### Assignment 1

### Q1: k-Nearest Neighbor classifier

**knn.ipynb** 

### Q2: Training a SVM / Softmax classifier

**svm.ipynb** 、**softmax.ipynb** 实现SVM/Softmax Loss的线性分类器

### Q4: Two-Layer Neural Network

**two_layer_net.ipynb** 网络模型模块化、FC层、ReLU、SGD

### Q5: Higher Level Representations: Image Features

**features.ipynb** 图像特征工程、以HOG、方向梯度图为例

### Assignment 2

### Q1: Multi-Layer Fully Connected Neural Networks (16%)

**FullyConnectedNets.ipynb** 多隐层MLP、带动量的SGD、RMSProp/Adam

### Q2: Batch Normalization

**BatchNormalization.ipynb** 实现Batch Normalization层、Layer Normalization层

### Q3: Dropout 

**Dropout.ipynb**

### Q4: Convolutional Neural Networks

**ConvolutionalNetworks.ipynb** 实现2d-Convolution层、Max Pooling层、针对CNN的Group Normalization、Spatial BN

### Q5: PyTorch on CIFAR-10

torch.Tensor、torch.nn.Moudle、torch.nn.Sequential

### Assignment 3

### Q1: Image Captioning with Vanilla RNNs 

**RNN_Captioning.ipynb**

### Q2: Image Captioning with Transformers 

**Transformer_Captioning.ipynb**实现Transformer Decoder中的多头注意力机制与位置编码

### Q3: Network Visualization: Saliency Maps, Class Visualization, and Fooling Images 

**Network_Visualization.ipynb** 对输入图像使用梯度上升生成特定类别的可视化图像

### Q4: Generative Adversarial Networks 

**Generative_Adversarial_Networks.ipynb** MNIST数据集上的GAN（ Vanilla BCE Loss）、Least Squares GAN Loss、DCGAN

### Q5: Self-Supervised Learning for Image Classification

**Self_Supervised_Learning.ipynb**, 以自监督对比学习中的SimLR为例、自监督学习后得到的特征提取模型+分类器与纯监督学习对比

### Extra Credit: Image Captioning with LSTMs 

**LSTM_Captioning.ipynb** 修改Q1中RNN隐层单元结构，实现LSTM