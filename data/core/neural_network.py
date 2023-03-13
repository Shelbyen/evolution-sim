import numpy as np


def rec_feedforward(inputs, layers):
    new_input = np.array([], float)
    layer = layers[0]
    for neuron in layer:
        new_input = np.append(new_input, neuron.feedforward(inputs))
    if len(layers) == 1:
        return new_input
    else:
        return rec_feedforward(new_input, layers[1:])


def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, back_layer):
        self.weights = np.random.random_sample(back_layer)
        self.bias = np.random.uniform(0, 0.3)

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuralNetwork:
    def __init__(self, descendant=None, size=None):
        if size is None:
            size = [6, 5]

        if not descendant:
            self.layers = np.array([], object)
            for i in range(size[0]):
                self.layer = np.array([], object)
                for j in range(size[1]):
                    if i != 0:
                        self.layer = np.append(self.layer, Neuron(5))
                    else:
                        self.layer = np.append(self.layer, Neuron(1))
                self.layers = np.append(self.layers, self.layer)
            self.layers = self.layers.reshape(size[0], size[1])
        else:
            self.layers = descendant

    def feedforward(self, inputs):
        if len(inputs) != len(self.layers[0]):
            return -1
        new_inputs = np.array([], float)
        first_layer = self.layers[0]
        i = 0
        for neuron in first_layer:
            new_inputs = np.append(new_inputs, neuron.feedforward(inputs[i]))
            i += 1
        return rec_feedforward(new_inputs, self.layers[1:])


nn = NeuralNetwork()
print(nn.feedforward([5000, 0.89410000, 0.15678, 0.5372745, 0.968568]))
