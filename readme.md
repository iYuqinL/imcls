### 代码结构

```
.
├── DataSets  # 训练数据集和测试数据集
├── imcls     # 检测算法模块。里面包括算法模型核心代码以及模型优化代码。不过由于里面有一些模块并未完成(engine模块相关等)。所以模型的训练由train.py显式重写的
├── results   # 测试集检测结果
├── tools     # 一些与算法模型无关的工具
│   ├── diff_res.py
│   └── train_valid_split.py
├── train-outs  # 训练模型权重结果
├── train.py  # 模型训练代码
├── test.py   # 模型推理测试代码
```

算法的实现具有非常清晰的结构，对于不同的模型，不同的约束等仅需要简单的修改对应的配置就可以。

`imcls`模块里面包含了多种基本的分类模型算法。

```
imcls
├── __init__.py
├── checkpoint      # 模型checkpoint实现，目前尚未完善
├── config          # 网络yaml配置文件读取相关代码
├── data            # 数据集及数据增强处理代码
│   ├── __init__.py
│   ├── build.py
│   ├── datasets    # 数据集读取代码，custom 数据集需要再此处添加对应的数据读取方式
│   ├── samplers    # 数据采样器
│   └── transforms  # 数据增强变换
├── engine          # 训练器，目前尚未完善。完善后可以实现更加简便的训练以及训练日志的记录
├── modeling        # 网络模块化
│   ├── __init__.py
│   ├── backbone    # 骨干网络，目前收录resnet_cbam，resnest，efficientnet
│   │                # 以及torchvisiosn 里面的网络结构。
│   ├── heads       # 网络头部，一般都是全连接层
│   ├── meta_arch   # 基本网络结构，即backbone和heads的组合，会根据配置进行组合。
│   ├── modeling.py # 临时实现的分类模型，未来完善后将弃用
│   └── network.py  # 临时实现的分类模型，未来完善后将弃用(已弃用)
├── nn_module       # 一些小模块
│   ├── attention.py
│   ├── __init__.py
│   └── loss_trick.py
├── solver          # 优化器相关
│   ├── build.py
│   ├── __init__.py
│   └── lr_scheduler.py
└── utils
```

### 代码运行

#### 依赖安装

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.6
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- fvcore

#### 训练

参照 configs 里面的配置，配置好网络，优化器等之后，将 train.py 的

```
cfg_file = "configs/call_smoke_cls/resnest200-S2.yaml"
```

改为你的配置文件路径。

执行 

```
python3 train.py
```

即可。

#### 测试

与训练部分类似。