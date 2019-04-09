import math
import numpy as np


def load_codebook(net, path, conv_bits, fc_bits):
    conv_layer_num = 0
    fc_layer_num = 0
    fin = open(path, 'rb')
    for name, x in net.named_parameters():
        if name.endswith('mask'):
            continue
        if name.startswith('conv'):
            conv_layer_num += 1
        elif name.startswith('fc'):
            fc_layer_num += 1
    nz_num = np.fromfile(fin, dtype=np.uint32, count=conv_layer_num + fc_layer_num)
    conv_diff_num = sum(nz_num[:conv_layer_num])
    conv_diff = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)

    fc_merge_num = math.floor((sum(nz_num[conv_layer_num:]) + 1) / 2)
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)

    print(nz_num)
    print(len(conv_diff), conv_diff[-10:])
    print(len(fc_merge_diff), fc_merge_diff[-10:])
    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 39316 [ 17 242  34  50 164  44  26   3   6 128]

    # Split 8 bits index to 4 bits index
    fc_diff = []
    max_bits = 2 ** fc_bits
    for i in range(len(fc_merge_diff)):
        fc_diff.append(int(fc_merge_diff[i] / max_bits))  # first 4 bits
        fc_diff.append(fc_merge_diff[i] % max_bits)  # last 4 bits
    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0:
        fc_diff = fc_diff[:fc_num_sum]

    conv_codebook_index = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)
    fc_codebook_index_merge = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)
    codebook_value_num = int((2 ** conv_bits) * (conv_layer_num / 2) + (2 ** fc_bits) * (fc_layer_num / 2))
    codebook_value = np.fromfile(fin, dtype=np.float32, count=codebook_value_num)

    print(len(conv_codebook_index), conv_codebook_index[-10:])
    print(len(fc_codebook_index_merge), fc_codebook_index_merge[-10:])
    print(len(codebook_value), codebook_value[-10:])
    # 8537 [136 122 119 132 236  73 126  75  16  74]
    # 39316 [170 134 100 198 167  99 150  37   5 176]
    # 544 [ 0.15731171  0.11615839 -0.00030401 -0.12842683  0.12538931  0.12218501
    #   0.18652469  0.19523832  0.25232622  0.296758  ]

    # Split 8 bits index to 4 bits index
    fc_codebook_index = []
    max_bits = 2 ** fc_bits
    for i in range(len(fc_codebook_index_merge)):
        fc_codebook_index.append(int(fc_codebook_index_merge[i] / max_bits))  # first 4 bits
        fc_codebook_index.append(fc_codebook_index_merge[i] % max_bits)  # last 4 bits
    if fc_num_sum % 2 != 0:
        fc_codebook_index = fc_codebook_index[:fc_num_sum]
    fc_codebook_index = np.asarray(fc_codebook_index, dtype=np.uint8)

    print(nz_num)
    print(len(conv_diff), conv_diff[-10:])
    print(len(fc_diff), fc_diff[-10:])
    print(len(conv_codebook_index), conv_codebook_index[-10:])
    print(len(fc_codebook_index), fc_codebook_index[-10:])
    print(len(codebook_value), codebook_value[-10:])
    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 78631 [4, 2, 12, 1, 10, 0, 3, 0, 6, 8]
    # 8537 [136 122 119 132 236  73 126  75  16  74]
    # 78631 [ 7  6  3  9  6  2  5  0  5 11]
    # 544 [ 0.15731171  0.11615839 -0.00030401 -0.12842683  0.12538931  0.12218501
    #   0.18652469  0.19523832  0.25232622  0.296758  ]

    return nz_num, conv_diff, fc_diff, conv_codebook_index, fc_codebook_index, codebook_value
