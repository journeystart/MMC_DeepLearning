# CNN

## 一、CNN基础



## 二、通过MNIST数据库实现手写数字的识别

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.optim as optim

#数据的准备
batch_size = 64
#神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
#仅有灰度值->单通道   RGB -> 三通道 读入的图像张量一般为W*H*C (宽、高、通道数) 在pytorch中要转化为C*W*H
transform = transforms.Compose([
    #将数据转化为图像张量
    transforms.ToTensor(),
    #进行归一化处理，切换到0-1分布 （均值， 标准差）
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                          )
test_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform
                               )
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size
                          )


#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, padding=2)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5, padding=2)
        #self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=3, padding=1)
        #池化层
        self.pooling = torch.nn.MaxPool2d(2)  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(980, 10)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        #x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
model = Net()
#在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#训练
#将一次迭代封装入函数中
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        #在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        for data in test_loader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)#dim=1沿着索引为1的维度(行)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

### 1. transforms.Compose

表示数据的预处理操作集合，通过Compose函数将预处理函数打包，从而实现批处理。

```python
transform = transforms.Compose([
    #将数据转化为图像张量
    transforms.ToTensor(),
    #进行归一化处理，切换到0-1分布 （均值， 标准差）
    transforms.Normalize((0.1307, ), (0.3081, ))
])
```

这里实现两个预处理操作，之后只要传入transform，就相当于传入这个预处理组。

预处理组可以提前将图像进行旋转、分割等一系列操作。

### 2. datasets

数据集导入

```python
train_dataset = datasets.MNIST(root='./dataset/mnist/', # 数据集的位置
                               train=True,              # MNIST数据集分好了训练集和测试集
                               download=True,			# 如果找不到数据集则进行下载
                               transform=transform		# 预处理调用函数组
                               )
```

### 3. DataLoader

通过分组，将数据分组成多个训练单位，实现批处理操作。通过GPU对组内的数据进行批量处理，提速。

```python
batch_size = 64
train_loader = DataLoader(train_dataset,        # 进行分组的数据
                          shuffle=True,			# 是否进行打乱
                          batch_size=batch_size	# 分组的数据量大小
                          )
```

DataLoader库不仅能提供数据分组，还能够将数据打乱，重排等操作。

### 4. CNN模型的建立和Class类的使用

将CNN网络写到一个类中，方便调用和管理。

```python
#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, padding=2)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5, padding=2)
        #self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=3, padding=1)
        #池化层
        self.pooling = torch.nn.MaxPool2d(2)  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(980, 10)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        #x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
```

对于class类，Net(torch.nn.Module)括号内的表示父类，python作为面对对象的编程，其特点之一就是继承，当前类可以继承父类的函数和变量集合。

```python
    def __init__(self): # 类初始化函数，self是传入自身，以实现自我调用
        super(Net, self).__init__() # super函数可以创建一个父级的代理，从而调用父级的函数
 		# 其中super(使用哪个类的父级，传入参数).调用父级的函数名称()
        # super函数调用的时候时进行按照MRO（方法解析顺序）传递进行的
        
# 例子
class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        print("B")
        super().__init__()

class C(A):
    def __init__(self):
        print("C")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D")
        super().__init__()

d = D()
# d的MRO为“D->B->C->A”，按照解析顺序进行可以避免A的多次初始化
# super相当于找了一个代理，帮忙执行函数
# 代码输出‘D B C A’

# 而如果直接调用函数，没有通过super进行传递，那么就是找到最近的函数进行执行
class A:
    def hello(self): print("A")

class B(A):
    def hello(self): print("B")

class C(A):
    def hello(self): print("C")

class D(B, C):
    pass

d = D()
d.hello()
# 代码输出‘B’
        
```

\_\_call\_\_魔术方法，在代码中，Net类的\_\_call\_\_被写为forward函数

```python
model = Net()
outputs = model(inputs)
```

代码中这样的写法相当于，先定义一个变量modle属于Net类，同时也执行了\_\_init\_\_魔法函数；再通过\_\_call\_\_魔法函数实现了，将inputs传入forward函数。

### 5. GPU加速、损失与优化器

```python
#在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

对于分类问题使用的默认损失函数——交叉熵损失函数(CrocessEntropyLoss)，$f(x)=-log{(p_{true})}$，正确类别的预测概率越大，损失越小，能够很好的“惩罚”错误的预测，推动模型更快地学习区分类别。

SGD（随机梯度下降法），设置学习率为0.01，冲量为0.5。$v_{t+1}=μv_{t}−η∇L(w_t)$，$μ$为冲量系数，表示惯性的大小，保持上一状态的程度，可以加快收敛速度，减少震荡，避免陷入局部最小值（小球有惯性，会顺势滑过小坑，直达更低的地方。

### 6.训练

```python
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, start=0): # enumerate穷举函数，将train_loader中的元组全部枚举，并自动为每个元组添加索引，索引序号从start开始,这里是每次取出一个batch
        inputs, target = data # 这里将data数据拆分成inputs和targrt
        # 在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device) 
        # 将获取到的inputs和target传入GPU进行处理
        # 那为什么这里需要重新赋值，这是由于对于Tensor类，他的.to()函数不会改变原有像素的，而前面的model.to(device)却没有重新赋值，这是由于对于Module.to()函数不会修改原始值，是直接替换的。

        # 清除旧梯度 + 前向预测 + 计算损失 + 反向计算梯度 + 更新参数
        optimizer.zero_grad()  # 这里需要先清除旧的梯度，由于pytorch的backward是梯度的累加，因此每次需要先清除梯度信息
        outputs = model(inputs)# 利用__call__魔术方法，调用前向预测
        loss = criterion(outputs, target) # 使用前面已经设置的优化器计算损失
        loss.backward() # 自动计算反向梯度，把结果存在 param.grad 里
        optimizer.step() #自动更新参数

        running_loss += loss.item()
        
        # batch数量每到300，输出一次loss
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
```

`model.parameters()`是 `nn.Parameter` 对象，是继承自 `Tensor` 的特殊类型。

每个参数有两个重要属性：

1. **`param.data`** → 真实存储的数值（权重本身）。
2. **`param.grad`** → 梯度，只有在调用 `loss.backward()` 后才会生成。

### 7. 测试

```python
def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        for data in test_loader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)#dim=1沿着索引为1的维度(行)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))
```

`with`是python的上下文管理器，用于确保资源被正确管理和释放，即使发生异常也不例外。

```python
with 上下文管理器表达式 [as 变量]:
    # 代码块
```

在**文件操作**、**数据库链接**和**线程锁**的结构中使用，能有非常好的效果。

```python
#######文件操作#######
# 传统方式 - 需要手动关闭文件
file = open('example.txt', 'r')
content = file.read()
file.close()  # 必须记得关闭！

# 使用 with - 自动关闭文件
with open('example.txt', 'r') as file:
    content = file.read()
# 这里文件会自动关闭，即使发生异常也会关闭

#######数据库连接#######
import sqlite3

with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    results = cursor.fetchall()
# 连接会自动关闭

#######线程锁#######
import threading

lock = threading.Lock()

with lock:
    # 临界区代码
    # 锁会自动获取和释放
    shared_variable += 1
```

在pytorch中的这个操作中

```python
# torch.no_grad() 返回一个上下文管理器对象
with torch.no_grad():  # 进入上下文
    # 在这里的操作不会计算梯度
    output = model(input)
# 退出上下文，恢复原来的梯度计算状态
```

#### 上下文管理器的工作原理

上下文管理器需要实现两个特殊方法：

- `__enter__()`: 进入上下文时调用
- `__exit__()`: 退出上下文时调用

```python
### 自定义上下文管理器示例：
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.time()
        print(f"执行时间: {self.end - self.start:.2f} 秒")

# 使用自定义上下文管理器
with Timer():
    # 测量这段代码的执行时间
    result = sum(i**2 for i in range(1000000))
```