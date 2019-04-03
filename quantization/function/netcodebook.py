class NetCodebook():
    def __init__(self, conv_bits, fc_bits):
        self.conv_bits = conv_bits
        self.fc_bits = fc_bits
        self.codebook_index = []
        self.codebook_value = []

    def add_layer_codebook(self, layer_codebook_index, layer_codebook_value):
        self.codebook_index.append(layer_codebook_index)
        self.codebook_value.append(layer_codebook_value)
