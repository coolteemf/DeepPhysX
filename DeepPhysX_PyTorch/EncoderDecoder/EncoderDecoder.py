from torch.nn import Sequential


class EncoderDecoder:

    def __init__(self,
                 layers=[],
                 nb_encoding_layers=0,
                 nb_decoding_layers=0):

        self.layers = layers
        self.nbEncodingLayers = nb_encoding_layers
        self.nbDecodingLayers = nb_decoding_layers if nb_decoding_layers > 0 else nb_encoding_layers
        self.encoder = []
        self.decoder = []

        # Rectangle network (Each layer of the encoder and decoder has the same amount of neurones)
        if len(self.layers) == 1:
            self.encoder = self.layers
            self.decoder = self.layers

        # Symetric Network {layers: [50, 10], nbEncodingLayers : 2, nbDecodingLayers: 2}
        # Network will have [50 - 10 - 10 - 50] neurones on each layers
        elif len(self.layers) == self.nbEncodingLayers == self.nbDecodingLayers:
            self.encoder = self.layers
            self.decoder = self.layers[::-1]

        # Asymetric Network {layers: [50, 10, 11, 15, 73], nbEncodingLayers : 2, nbDecodingLayers: 3}
        # Network will have [50 - 10 - 11 - 15 - 73] neurones on each layers
        elif len(self.layers) > self.nbEncodingLayers:
            self.encoder = self.layers[:self.nbEncodingLayers]
            self.decoder = self.layers[self.nbEncodingLayers:]
        else:
            print("[ENCODER-DECODER] Layers count is {} which does not match any known pattern.".format(len(self.layers)))
            print("                  Hence, the network is empty.")

    def setupEncoder(self):
        return Sequential(*self.encoder)

    def setupDecoder(self):
        return Sequential(*self.decoder)

    def executeSequential(self):
        return