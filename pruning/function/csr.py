from scipy.sparse import csr_matrix
import numpy as np


class WeightCSR(csr_matrix):
    def __init__(self, tensor, index_bits=8):
        self.tensor = tensor.numpy().reshape(-1)
        # non zero number
        self.nz_num = 0
        self.index_bits = index_bits

    def tensor_to_csr(self):
        # index diff list
        max_index = 2 ** self.index_bits
        diff_list = []
        last_index = -1
        value_list = []
        for i, value in enumerate(self.tensor):
            diff = i - last_index - 1
            if diff >= max_index - 1:
                self.nz_num += 1
                # diff_list.append(bin(max_index - 1)[2:].zfill(self.index_bits))
                diff_list.append(max_index - 1)
                value_list.append(0)
                last_index = i
            elif abs(value) < 1e-6:
                continue
            else:
                self.nz_num += 1
                # diff_list.append(bin(diff)[2:].zfill(self.index_bits))
                diff_list.append(diff)
                value_list.append(value)
                last_index = i
        return (diff_list, value_list)

     # def csr_to_tensor(self, shape):
     #
