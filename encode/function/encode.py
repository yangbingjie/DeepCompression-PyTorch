import huffman


def compute_symbol_probability(input):
    symbol_probability = {}
    for item in input:
        if item in symbol_probability:
            symbol_probability[item] += 1
        else:
            symbol_probability[item] = 1
    return symbol_probability


def encode_huffman(symbol_probability):
    return huffman.codebook(symbol_probability)


def encode_data(input, map):
    output = [0]*len(input)
    for i in range(len(input)):
        output[i] = map[input[i]]
    return output
