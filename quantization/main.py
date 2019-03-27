from pruning.net.LeNet5 import LeNet5
from quantization.function.weight_share import share_weight
from quantization.net.SparseLeNet5 import SparseLeNet5
from quantization.function.helper import retrain_codebook

before_path = './pruning/result/LeNet_retrain'
dense_net = LeNet5()
codebook, nz_num, conv_diff, fc_diff = share_weight(dense_net, before_path, 8, 5)

net = SparseLeNet5()
retrain_codebook(net, codebook, nz_num, conv_diff, fc_diff)
