import os
from quantization.net.LeNet5 import LeNet5
import encode.function.helper as helper
from pruning.function.helper import test
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import encode.function.encode as encode

use_cuda = False  # torch.cuda.is_available()
quantization_result_path = './quantization/result/LeNet_codebook'
encode_huffman_root = './encode/result/'
if not os.path.exists(encode_huffman_root):
    os.mkdir(encode_huffman_root)
encode_huffman_name = 'LeNet_encode'
encode_codebook_path = encode_huffman_root + encode_huffman_name
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
data_dir = './data'
quantization_conv_bits = 8
quantization_fc_bits = 4
max_conv_bits = 2 ** quantization_conv_bits
max_fc_bits = 2 ** quantization_fc_bits
test_batch_size = 4
testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         **kwargs)

net = LeNet5()
# Load codebook from file
conv_layer_num, nz_num, conv_diff, fc_diff, \
conv_codebook_index, fc_codebook_index, codebook_value \
    = helper.load_codebook(net, quantization_result_path, max_conv_bits, max_fc_bits)

# Init net using codebook
helper.codebook_to_init(net, conv_layer_num, nz_num, conv_diff, fc_diff,
                        conv_codebook_index, fc_codebook_index, codebook_value, max_conv_bits, max_fc_bits)
test(use_cuda, testloader, net)

symbol_list = [conv_diff, conv_codebook_index, fc_diff, fc_codebook_index]
layer_nz_num = nz_num[0::2] + nz_num[1::2]
half_conv_layer_num = int(conv_layer_num / 2)

huffman_map_list = []
outputs_list = []
for i in range(len(symbol_list)):
    index = 0
    layer_nz = layer_nz_num[:half_conv_layer_num] if i < 2 else layer_nz_num[half_conv_layer_num:]
    for j in range(len(layer_nz)):
        inputs = symbol_list[i][index:index + layer_nz[j]]
        index += layer_nz[j]
        symbol_probability = encode.compute_symbol_probability(inputs)
        huffman_map = encode.encode_huffman(symbol_probability.items())
        huffman_map_list.append(huffman_map)
        outputs_list.append(encode.encode_data(inputs, huffman_map))

# TODO 保存Huffman编码后的数据outputs_list

nz_num.dtype = np.uint8
codebook_value.dtype = np.uint8
save_obj = np.concatenate((nz_num, codebook_value))
save_obj.tofile(encode_codebook_path)
