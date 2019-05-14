import os
from quantization.net.LeNet5 import LeNet5
import encode.function.helper as helper
from pruning.function.helper import test
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import encode.function.encode as encode
import json

use_cuda = False  # torch.cuda.is_available()
quantization_result_path = './quantization/result/LeNet_MNIST_codebook.pth'
encode_huffman_root = './encode/result/'
if not os.path.exists(encode_huffman_root):
    os.mkdir(encode_huffman_root)
encode_huffman_name = 'LeNet_encode'
encode_huffman_map_name = 'LeNet_huffman_map.json'
encode_codebook_path = encode_huffman_root + encode_huffman_name
encode_huffman_map_path = encode_huffman_root + encode_huffman_map_name
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

huffman_map_dict = {}
outputs_str = 0
for i in range(len(symbol_list)):
    index = 0
    layer_nz = layer_nz_num[:half_conv_layer_num] if i < 2 else layer_nz_num[half_conv_layer_num:]
    for j in range(len(layer_nz)):
        print(i, j)
        inputs = symbol_list[i][index:index + layer_nz[j]]
        index += layer_nz[j]
        print(len(inputs))
        symbol_probability = encode.compute_symbol_probability(inputs)
        print(sorted(symbol_probability.items()))
        huffman_map = encode.encode_huffman(symbol_probability.items())
        print(huffman_map)
#         outputs = encode.encode_data(inputs, huffman_map)
#         keys = list(huffman_map.keys()).copy()
#         for key in keys:
#             huffman_map[str(key)] = huffman_map.pop(key)
#         huffman_map_dict[str(i)+'_'+str(j)] = huffman_map
#         for output in outputs:
#             outputs_str += output
#
# outputs_str += '0' * (8 - (len(outputs_str) & 0x7))
# outputs_arr = np.zeros(len(outputs_str) >> 3,dtype=np.uint8)
# for i in range(len(outputs_str) >> 3):
#     outputs_arr[i] = int(outputs_str[i<<3: (i<<3) + 8],2)
#
# nz_num.dtype = np.uint8
# codebook_value.dtype = np.uint8
# save_obj = np.concatenate((nz_num, codebook_value,outputs_arr))
# save_obj.tofile(encode_codebook_path)
# #print(huffman_map_dict)
# with open(encode_huffman_map_path,'w') as file:
#     huffman_map_str = json.dumps(huffman_map_dict)
#     file.write(huffman_map_str)
