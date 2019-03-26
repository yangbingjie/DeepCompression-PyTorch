import numpy as np
import pruning.function.helper as helper
from sklearn.cluster import KMeans
from quantization.codebook import Codebook


def share_weight(net, before_path, conv_bits, fc_bits):
    conv_layer_length, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array \
        = helper.load_sparse_model(net, before_path)
    index = 0
    codebook = Codebook(conv_bits, fc_bits)
    bits = conv_bits
    for i in range(int(len(nz_num) / 2)):
        layer_type = 'conv' if i < conv_layer_length else 'fc'
        if layer_type == 'fc':
            bits = fc_bits
            layer_weight = fc_value_array[index:index + nz_num[i * 2]]
        else:
            layer_weight = conv_value_array[index:index + nz_num[i * 2]]
        print(np.any(np.isnan(layer_weight)))
        print(np.all(np.isfinite(layer_weight)))
        print(layer_weight.dtype)

        index += nz_num[i * 2]
        min_weight = min(layer_weight)
        max_weight = max(layer_weight)
        # Use linear initialization for kmeans
        space = np.linspace(min_weight, max_weight, num=2 ** bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
                        algorithm="full")
        kmeans.fit(layer_weight.reshape(-1, 1))
        new_layer_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        # print(new_layer_weight)
        codebook.add_layer_codebook(layer_type, new_layer_weight)
