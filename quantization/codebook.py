class Codebook():
    def __init__(self, conv_bits, fc_bits):
        self.conv_bits = conv_bits
        self.fc_bits = fc_bits
        self.conv_layer_list = []
        self.fc_layer_list = []

    def add_layer_codebook(self, layer_type, layer_codebook_list):
        if layer_type == 'conv':
            self.conv_layer_list.append(layer_codebook_list)
        else:
            self.fc_layer_list.append(layer_codebook_list)
