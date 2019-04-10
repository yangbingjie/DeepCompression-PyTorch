import os
from quantization.net.LeNet5 import LeNet5
import encode.function.helper as helper
from pruning.function.helper import test
import torch
import torchvision
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()
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
use_cuda = torch.cuda.is_available()
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
