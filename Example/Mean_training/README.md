This folder serves as an example on "How to use DeepPhysX to train a network".

DeepPhysX is based on the idea of data PRODUCERS and data CONSUMERS.
![alt text](https://imgur.com/Fb3pu1E)


The Producer.py file represents the behavior of a single client (PRODUCER).
A client must define:
    How to generate/transform data (input and output)
    When to send data to the CONSUMERS using the predefined functions

The Consumers.py represent the behavior of the training process.
A training process must define:
    How long will the training run (#batch, #epoch)
    The amount of data to create (#batch_size)
    How many producers to create
    Where to store the data
    The artificial neural network.

