import numpy as np
import quantization.function.helper as helper
from sklearn.cluster import KMeans
from quantization.function.netcodebook import NetCodebook


def share_weight(net, before_path, conv_bits, fc_bits, prune_fc_bits):
    '''quantization

    Args:
        net:            the network object
        before_path:    the path of the pruned model
        conv_bits:      the bits of each value in convolutional layer
        fc_bits:        the bits of each value in full-connect layer        
        prune_fc_bits:  the bits of each index in full-connect layer

    Returns:
        conv_layer_num:     the Number of convolutional layers
        codebook:           he sparse value of each convolutional layers
        nz_num:             the Number of non-zero value in each layers
        conv_diff:          the sparse index of each convolutional layers
        fc_diff:            the sparse index of each full-connect layers
    '''
    conv_layer_num, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array \
        = helper.load_sparse_model(net, before_path, prune_fc_bits)
    conv_index = 0
    fc_index = 0
    codebook = NetCodebook(conv_bits, fc_bits)
    have_bias = True
    stride = 2 if have_bias else 1
    layer_nz_num = nz_num[0::stride] + nz_num[1::stride]
    for i in range(len(layer_nz_num)):
        layer_type = 'conv' if i < conv_layer_num / stride else 'fc'
        if layer_type == 'fc':
            bits = fc_bits
            layer_weight = fc_value_array[fc_index:fc_index + layer_nz_num[i]]
            fc_index += layer_nz_num[i]
        else:
            bits = conv_bits
            layer_weight = conv_value_array[conv_index:conv_index + layer_nz_num[i]]
            conv_index += layer_nz_num[i]

        min_weight = min(layer_weight)
        max_weight = max(layer_weight)
        # Use linear initialization for kmeans
        n_clusters = 2 ** bits
        space = np.linspace(min_weight, max_weight, num=n_clusters, dtype=np.float32)
        # kmeans = KMeans(n_clusters=n_clusters, init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
        #                 algorithm="full")
        kmeans = KMeans(n_clusters=n_clusters,  init='k-means++', n_init=1, precompute_distances=True,
                        algorithm="full")
        kmeans.fit(layer_weight.reshape(-1, 1))
        codebook_index = np.array(kmeans.labels_, dtype=np.uint8)
        # print(new_layer_weight[:5])
        codebook_value = kmeans.cluster_centers_[:n_clusters]

        codebook.add_layer_codebook(codebook_index.reshape(-1), codebook_value.reshape(-1))

    return conv_layer_num, codebook, nz_num, conv_diff, fc_diff
