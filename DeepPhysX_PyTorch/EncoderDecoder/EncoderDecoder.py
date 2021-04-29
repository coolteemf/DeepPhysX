import torch.nn as nn
from itertools import chain


class EncoderDecoder:

    def __init__(self, layers=None, nb_encoding_layers=0, nb_decoding_layers=0):

        self.layers = [] if layers is None else layers
        self.nbEncodingLayers = nb_encoding_layers
        self.nbDecodingLayers = nb_decoding_layers if nb_decoding_layers is not 0 else nb_encoding_layers
        self.encoder = []
        self.decoder = []

        if len(self.layers) == 1:
            self.encoder = [self.layers[0] for _ in range(self.nbEncodingLayers)]
            self.decoder = [self.layers[0] for _ in range(self.nbDecodingLayers)]
        elif len(self.layers) == self.nbEncodingLayers == self.nbDecodingLayers:
            self.encoder = [self.layers[i] for i in range(self.nbEncodingLayers)]
            self.decoder = [self.layers[self.nbEncodingLayers - i] for i in range(self.nbDecodingLayers)]
        elif len(self.layers) == self.nbEncodingLayers + self.nbDecodingLayers:
            self.encoder = [self.layers[i] for i in range(self.nbEncodingLayers)]
            self.decoder = [self.layers[self.nbEncodingLayers + i] for i in range(self.nbDecodingLayers)]
        elif len(self.layers) > self.nbEncodingLayers:
            self.encoder = [self.layers[i] for i in range(self.nbEncodingLayers)]
            self.decoder = [self.layers[self.nbEncodingLayers + i] for i in range(len(self.layers))]
        else:
            print("[ENCODER-DECODER] Layers count is {} which does not match any known pattern.".format(len(self.layers)))
            print("                  Hence, the network is empty.")

    def setupEncoder(self):
        return nn.Sequential(*self.encoder)

    def setupDecoder(self):
        return nn.Sequential(*chain(*self.decoder[:-1]))

    def executeSequential(self):
        return