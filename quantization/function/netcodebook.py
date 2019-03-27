class NetCodebook():
    def __init__(self, conv_bits, fc_bits):
        self.conv_bits = conv_bits
        self.fc_bits = fc_bits
        self.conv_codebook_index = []
        self.fc_codebook_index = []
        self.conv_codebook_value = []
        self.fc_codebook_value = []
        self.conv_kmeans_model = []
        self.fc_kmeans_model = []

    def add_layer_codebook(self, layer_type, layer_codebook_index, layer_codebook_value, kmeans_model):
        if layer_type == 'conv':
            self.conv_codebook_index.append(layer_codebook_index)
            self.conv_codebook_value.append(layer_codebook_value)
            self.conv_kmeans_model.append(kmeans_model)
        else:
            self.fc_codebook_index.append(layer_codebook_index)
            self.fc_codebook_value.append(layer_codebook_value)
            self.fc_kmeans_model.append(kmeans_model)
