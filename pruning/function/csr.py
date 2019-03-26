from scipy.sparse import csr_matrix
import torch
from collections import deque


class WeightCSR(csr_matrix):
    def __init__(self, tensor, index_bits=8):
        self.tensor = tensor.reshape(-1)
        # non zero number
        self.nz_num = 0
        self.max_index = 2 ** index_bits

    def tensor_to_csr(self):
        # index diff list
        diff_list = deque([])
        last_index = -1
        value_list = deque([])
        for i, value in enumerate(self.tensor):
            diff = i - last_index - 1
            if diff >= self.max_index - 1:
                self.nz_num += 1
                diff_list.append(self.max_index - 1)
                value_list.append(0)
                last_index = i
            elif value < 1e-5:
                continue
            else:
                self.nz_num += 1
                diff_list.append(diff)
                value_list.append(value)
                last_index = i
        return diff_list, value_list
