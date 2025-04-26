from mathfunctions import Activations, LossFunctions
import numpy as np
import csv
from objects import Game, DataFormater
from progressbar import ProgressBar
import concurrent.futures
import threading
from typing import Tuple
from math import sqrt
class RylansNeuralNetwork:
    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(1, len(layer_sizes)):  # Skip input layer
            self.layers.append(RylansNeuralNetwork.Layer(layer_sizes[i], layer_sizes[i - 1]))
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.size != self.layers[0].input_size: return []
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backward(self, inputs: np.ndarray, expected: np.ndarray, learning_rate: float = 0.01, loss_funtcion=LossFunctions.MSE, loss_derivative=LossFunctions.Derivatives.MSE, optimizer_sgd:bool=False, optimizer_momentum:bool=False, beta: float = 0.9):
        activations = [inputs] # an array that stores the outputs to each layer
        for layer in self.layers:
            inputs = layer.forward(inputs)
            activations.append(inputs)
        output = activations[-1]
        loss_gradient = loss_derivative(activations[-1], expected) # loss with output layer and expected
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            prev_activations = activations[index]

            new_loss_gradient = np.zeros(layer.input_size)

            for j, neuron in enumerate(layer.neurons):
                d_activation = neuron.derivative(neuron.output)  # Activation function derivative
                neuron.delta = loss_gradient[j] * d_activation  # Store delta for weight update
                # Standard SGD update
                if optimizer_sgd:
                    neuron.weights -= learning_rate * neuron.delta * prev_activations
                    neuron.bias -= learning_rate * neuron.delta
                # Momentum SGD update
                elif optimizer_momentum:
                    neuron.momentum = beta * neuron.momentum + (1 - beta) * (neuron.delta * prev_activations)
                    neuron.bias_momentum = beta * neuron.bias_momentum + (1 - beta) * neuron.delta
                    neuron.weights -= learning_rate * neuron.momentum
                    neuron.bias -= learning_rate * neuron.bias_momentum
                else:
                    # Update weights and biases using gradient descent
                    neuron.weights -= learning_rate * neuron.delta * prev_activations
                    neuron.bias -= learning_rate * neuron.delta
                new_loss_gradient += neuron.delta * neuron.weights

            loss_gradient = new_loss_gradient
        return loss_funtcion(output, expected)
    def predict(self, input_data) -> Tuple[float, ...]:
        output = np.array(self.forward(input_data))
        return tuple(output)
    def predict_loss(self, input_data, loss_func=LossFunctions.MSE) -> Tuple[float, ...]:
        '''Returns (*output, *expected, loss) by forwarding through network'''
        expected = np.array(DataFormater.gameToOutputs(game))
        output = np.array(self.forward(input_data))
        loss = loss_func(output, expected)
        
        return (*output, *expected, loss)
    def __str__(self) -> str:
        # Visual representation of the neural network
        network_info = f"Neural Network Layers: {self.layer_sizes}\n"

        for layer_index, layer in enumerate(self.layers):
            network_info += f"Layer {layer_index + 1} (Size: {layer.size}):\n"
            
            # Visual representation of neurons in the layer
            for neuron_index, neuron in enumerate(layer.neurons):
                weights_str = ', '.join([f"{w:.2f}" for w in neuron.weights[:5]])  # First 5 weights for brevity
                network_info += f"  Neuron {neuron_index + 1}: Weights = [{weights_str}], Bias = {neuron.bias:.2f}\n"
                # Optionally, display more weights if needed, or truncate them

        return network_info
    def save(self, file: str = 'weights-biases.csv'):
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Save the network size (layer sizes)
            writer.writerow(self.layer_sizes)
            # Save the weights and biases for each layer and neuron
            for layer in self.layers:
                for neuron in layer.neurons:
                    writer.writerow(np.concatenate([neuron.weights, [neuron.bias]]))
    def load(file: str = 'weights-biases.csv') -> 'RylansNeuralNetwork':
        with open(file, 'r') as f:
            reader = csv.reader(f)
            weights_and_biases = list(reader)
        
        # The first row contains the layer sizes
        layer_sizes = list(map(int, weights_and_biases[0]))
        
        # Create a new network with the loaded layer sizes
        network = RylansNeuralNetwork(layer_sizes)
        
        # Load the weights and biases into the newly created network
        index = 1  # Start from the second row (first row is the layer sizes)
        for layer_index, layer in enumerate(network.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                # Load the weights and bias for each neuron
                row = np.array(weights_and_biases[index], dtype=float)
                neuron.weights = row[:-1]  # All except the last value are weights
                neuron.bias = row[-1]  # Last value is the bias
                index += 1
        return network
    def train(self, df_games, df_players, total:int=1000, learning_rate=1e-6):
        games = Game.GamesGenerator(df_games, df_players, total)
        pb = ProgressBar(total=total, desc="Training...", unit="", color=3, length=50, other="Loss")
        losses = []
        for _ in range(total):
            game = next(games)
            input_data = np.array(DataFormater.gameToInputs(game))
            expected = np.array(DataFormater.gameToOutputs(game))
            losses.append(self.backward(inputs=input_data, expected=expected, learning_rate=learning_rate))
            pb.update(other=(sum(losses) / len(losses)))
            if len(losses) > 500:
                losses.pop(0)
        return
    class Layer:
        def __init__(self, size: int, input_size: int):
            self.size = size
            self.neurons = [RylansNeuralNetwork.Neuron(input_size) for _ in range(size)]
            self.input_size = input_size
        def forward(self, inputs: np.ndarray) -> np.ndarray:
            return np.array([neuron.forward(inputs) for neuron in self.neurons])
    class Neuron:
        def __init__(self, input_size: int, activation=Activations.LeakyReLU, derivative=Activations.Derivates.LeakyReLU, weights=None, bias=None):
            self.activation = activation
            self.derivative = derivative
            # If weights and bias are provided (during load), use them; otherwise, initialize randomly
            if weights is None:
                self.weights = np.random.randn(input_size) * np.sqrt(1 / input_size) # 1 for Sgmoid, 2 for ReLU
            else:
                self.weights = weights
            if bias is None:
                self.bias = np.random.randn() / 10.0
            else:
                self.bias = bias
            self.output = 0

            # Store past gradients for optimizers
            self.delta = 0  # Used for standard SGD but also in momentum
            self.momentum = np.zeros(input_size)  # Momentum term
            self.bias_momentum = 0
        def forward(self, inputs: np.ndarray) -> float:
            z = np.dot(self.weights, inputs) + self.bias
            self.output = self.activation(z)
            return self.output

if __name__ == "__main__":
    from fetchdata import DataCollector
    collector = DataCollector("data.csv")
    df_games = collector.read_csv(0)
    df_players = collector.read_csv(1)
    
    # network = RylansNeuralNetwork([19, 64, 32, 32, 32, 2])
    network = RylansNeuralNetwork.load()
    # network.train(df_games, df_players, 50, 1e-7)
    # network.save()