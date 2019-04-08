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

    fc_merge_num = int((sum(nz_num[conv_layer_num:]) + 1) / 2)
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)

    # Split 8 bits index to 4 bits index
    fc_diff = []
    max_bits = 2 ** fc_bits
    for i in range(len(fc_merge_diff)):
        fc_diff.append(int(fc_merge_diff[i] / max_bits))  # first 4 bits
        fc_diff.append(fc_merge_diff[i] % max_bits)  # last 4 bits
    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0:
        fc_diff = fc_diff[:fc_num_sum]
    fc_diff = np.asarray(fc_diff, dtype=np.uint8)

    codebook_index_num = conv_diff_num + len(fc_diff)
    conv_codebook_index = np.fromfile(fin, dtype=np.uint8, count=codebook_index_num)
    fc_codebook_index_merge = np.fromfile(fin, dtype=np.uint8, count=codebook_index_num)
    codebook_value_num = (2 ** conv_bits) * conv_layer_num + (2 ** fc_bits) * fc_layer_num
    codebook_value = np.fromfile(fin, dtype=np.float32, count=codebook_value_num)

    # Split 8 bits index to 4 bits index
    fc_codebook_index = []
    max_bits = 2 ** fc_bits
    for i in range(len(fc_codebook_index_merge)):
        fc_codebook_index.append(int(fc_codebook_index_merge[i] / max_bits))  # first 4 bits
        fc_codebook_index.append(fc_codebook_index_merge[i] % max_bits)  # last 4 bits
    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0:
        fc_codebook_index = fc_codebook_index[:fc_num_sum]
    fc_codebook_index = np.asarray(fc_codebook_index, dtype=np.uint8)

    return nz_num, conv_diff, fc_merge_diff, conv_codebook_index, fc_codebook_index, codebook_value
