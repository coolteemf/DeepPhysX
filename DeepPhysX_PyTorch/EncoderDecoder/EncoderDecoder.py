from torch.nn import Sequential


class EncoderDecoder:

    def __init__(self,
                 layers=[],
                 nb_encoding_layers=0,
                 nb_decoding_layers=0):

        self.layers = layers
        self.nb_encoding_layers = nb_encoding_layers
        self.nb_decoding_layers = nb_decoding_layers if nb_decoding_layers > 0 else nb_encoding_layers

        # Rectangle network (Each layer of the encoder and decoder has the same amount of neurones)
        if len(self.layers) == 1:
            self.encoder = layers * self.nb_encoding_layers
            self.decoder = layers * self.nb_decoding_layers

        # Symmetric Network {layers: [50, 10], nbEncodingLayers : 2, nbDecodingLayers: 2}
        # Network will have [50 - 10 - 10 - 50] neurones on each layers
        elif len(self.layers) == self.nb_encoding_layers == self.nb_decoding_layers:
            self.encoder = self.layers
            self.decoder = self.layers[::-1]

        # Asymmetric Network {layers: [50, 10, 11, 15, 73], nbEncodingLayers : 2, nbDecodingLayers: 3}
        # Network will have [50 - 10 - 11 - 15 - 73] neurones on each layers
        elif len(self.layers) > self.nb_encoding_layers:
            self.encoder = self.layers[:self.nb_encoding_layers]
            self.decoder = self.layers[self.nb_encoding_layers:]

        else:
            raise ValueError(f"[EncoderDecoder] Layers count {len(self.layers)} does not match any known pattern.")

    def setupEncoder(self):
        return Sequential(*self.encoder)

    def setupDecoder(self):
        return Sequential(*self.decoder)

    def executeSequential(self):
        return
