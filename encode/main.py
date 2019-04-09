import os
from quantization.net.LeNet5 import LeNet5
import encode.function.helper as helper

quantization_result_path = './quantization/result/LeNet_codebook'
encode_huffman_root = './encode/result/'
if not os.path.exists(encode_huffman_root):
    os.mkdir(encode_huffman_root)
encode_huffman_name = 'LeNet_encode'
encode_codebook_path = encode_huffman_root + encode_huffman_name
net = LeNet5()
conv_bits = 8
fc_bits = 4
nz_num, conv_diff, fc_diff, conv_codebook_index, fc_codebook_index, codebook_value\
    = helper.load_codebook(net, quantization_result_path, conv_bits, fc_bits)
