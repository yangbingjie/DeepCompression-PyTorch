import math
import numpy as np


def load_codebook(net, path, max_conv_bits, max_fc_bits):
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

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 202755 [0 0 0 0 0 0 0 0 0 0]

    # Split 8 bits index to 4 bits index

    fc_diff = []
    for i in range(len(fc_merge_diff)):
        fc_diff.append(int(fc_merge_diff[i] / max_fc_bits))  # first 4 bits
        fc_diff.append(fc_merge_diff[i] % max_fc_bits)  # last 4 bits
    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0:
        fc_diff = fc_diff[:fc_num_sum]
    fc_diff = np.asarray(fc_diff, dtype=np.uint8)

    conv_codebook_index = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)
    fc_codebook_index_merge = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)
    codebook_value_num = int(max_conv_bits * (conv_layer_num / 2) + (2 ** max_fc_bits) * (fc_layer_num / 2))
    codebook_value = np.fromfile(fin, dtype=np.float32, count=codebook_value_num)

    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index_merge), fc_codebook_index_merge[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # 5669 [  2 228 211 229  76 152  23 116 111  25]
    # 202755 [200  66  71 152 140 171  86 151  87 197]
    # 544 [-0.11808116 -0.06328904  0.1446653   0.05191407 -0.03960273 -0.0174285
    #  -0.0174285   0.00504891  0.22879101  0.05191407]

    # Split 8 bits index to 4 bits index

    fc_codebook_index = []
    for i in range(len(fc_codebook_index_merge)):
        fc_codebook_index.append(int(fc_codebook_index_merge[i] / max_fc_bits))  # first 4 bits
        fc_codebook_index.append(fc_codebook_index_merge[i] % max_fc_bits)  # last 4 bits
    if fc_num_sum % 2 != 0:
        fc_codebook_index = fc_codebook_index[:fc_num_sum]
    fc_codebook_index = np.asarray(fc_codebook_index, dtype=np.uint8)
    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index), fc_codebook_index[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 405510 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 5669 [  2 228 211 229  76 152  23 116 111  25]
    # 405510 [10 11  5  6  9  7  5  7 12  5]
    # 544 [-0.11808116 -0.06328904  0.1446653   0.05191407 -0.03960273 -0.0174285
    #  -0.0174285   0.00504891  0.22879101  0.05191407]

    return conv_layer_num, nz_num, conv_diff, fc_diff, conv_codebook_index, fc_codebook_index, codebook_value


def codebook_to_init(net, conv_layer_length, nz_num, conv_diff, fc_diff, conv_codebook_index, fc_codebook_index,
                     codebook_value, max_conv_bits, max_fc_bits):
    state_dict = net.state_dict()
    conv_layer_index = 0
    fc_layer_index = 0
    codebook_value_index = 0

    layer_codebook_value = []
    for i, (key, value) in enumerate(state_dict.items()):
        shape = value.shape
        value = value.view(-1)
        value.zero_()
        if i < conv_layer_length:
            layer_diff = conv_diff[conv_layer_index:conv_layer_index + nz_num[i]]
            layer_codebook_index = conv_codebook_index[conv_layer_index:conv_layer_index + nz_num[i]]
            if not key.endswith('bias'):
                layer_codebook_value = codebook_value[codebook_value_index:codebook_value_index + max_conv_bits]
                codebook_value_index += max_conv_bits
            conv_layer_index += nz_num[i]
        else:
            layer_diff = fc_diff[fc_layer_index:fc_layer_index + nz_num[i]]
            layer_codebook_index = fc_codebook_index[fc_layer_index:fc_layer_index + nz_num[i]]
            if not key.endswith('bias'):
                layer_codebook_value = codebook_value[codebook_value_index:codebook_value_index + max_fc_bits]
                codebook_value_index += max_fc_bits
            fc_layer_index += nz_num[i]

        dense_index = 0
        sparse_index = 0
        while sparse_index < len(layer_diff):
            dense_index += layer_diff[sparse_index]
            value[dense_index] = float(layer_codebook_value[layer_codebook_index[sparse_index]])
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)

