from scipy.sparse import csr_matrix
import numpy as np


class WeightCSR(csr_matrix):
    def __init__(self, tensor, index_bits=8):
        self.tensor = tensor.numpy().reshape(-1)
        # non zero number
        self.nz_num = 0
        self.index_max = 2 ** index_bits

    def tensor_to_csr(self):
        # index diff list
        diff_list = []
        last_index = 0
        value_list = []
        for i, value in enumerate(self.tensor):
            diff = i - last_index
            if diff >= self.index_max:
                self.nz_num += 1
                diff_list.append(self.index_max)
                value_list.append(0)
                last_index = i
            elif abs(value) < 1e-6:
                continue
            else:
                self.nz_num += 1
                diff_list.append(diff)
                value_list.append(value)
                last_index = i
        print(diff_list)
        print(value_list)
        print(self.nz_num)
        return (diff_list, value_list)

     # def csr_to_tensor(self, shape):
     #
