# DeepCompression-PyTorch
PyTorch implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding' by Song Han, Huizi Mao, William J. Dally 

### Dependence

- python3.7+
- pytorch, torchvision
- numpy
- [huffman](https://pypi.org/project/huffman/)

### Structure

```
.
├── LICENSE
├── README.md
├── encode  // Huffman decode
│   ├── __init__.py
│   ├── function
│   │   ├── __init__.py
│   │   ├── encode.py
│   │   └── helper.py
│   └── main.py
├── encode_main.py
├── prune_main.py
├── pruning  // Prune Layer
│   ├── function
│   │   ├── helper.py
│   │   └── prune.py
│   ├── main.py
│   └── net
│       ├── PruneAlexNet.py
│       ├── PruneLeNet5.py
│       └── PruneVGG16.py
├── quantization // Quantizate Layer
│   ├── __init__.py
│   ├── function
│   │   ├── __init__.py
│   │   ├── helper.py
│   │   ├── netcodebook.py
│   │   └── weight_share.py
│   ├── main.py
│   └── net
│       ├── AlexNet.py
│       ├── LeNet5.py
│       └── VGG16.py
├── quantization_main.py
└── util
    └── log.py
```

### Usage

#### Pruning

```
python prune_main.py -net LeNet -data MNIST
```

- Network name:  LeNet, AlexNet VGG16
- Dataset name: MNIST CIFAR10 CIFAR100

#### Quantization

```
python quantization_main.py -net LeNet -data MNIST
```

- Network name:  LeNet, AlexNet VGG16
- Dataset name: MNIST CIFAR10 CIFAR100