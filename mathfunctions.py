from math import exp, sqrt, log
import numpy as np
class Activations:
    def Sigmoid(x) -> float:
        return 1.0 / (1.0 + exp(-x))
    def ReLU(x) -> float:
        return x if x > 0.0 else 0.0
    def Step(x) -> float:
        return 1 if x > 0.0 else 0.0
    def Linear(x) -> float:
        return x
    def Tanh(x) -> float:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    def LeakyReLU(x, a=0.1) -> float:
        return a*x if x < 0 else x
    def ELU(x, a=1) -> float:
        return x if x >= 0 else a*(exp(x) - 1)
    def Swish(x) ->  float:
        return x * Activations.Sigmoid(x)
    def GELU(x) -> float:
        a = 0.450158
        b = 0.044715
        return 0.5 * x * (1 + Activations.Tanh( a*( x + b*x*x*x ) ))
    class Derivates:
        def Sigmoid(x) -> float:
            return 1.0 / (1.0 + exp(-x))
        def ReLU(x) -> float:
            return 1 if x > 0 else 0.0
        def Step(x) -> float:
            return 0.0
        def Linear(x) -> float:
            return 1
        def Tanh(x) -> float:
            return 1.0 - Activations.Tanh(x) * Activations.Tanh(x)
        def LeakyReLU(x, a=0.1) -> float:
            return a if x < 0 else 1
        def ELU(x, a=1) -> float:
            return 1 if x >= 0 else a*exp(x)
        def Swish(x) ->  float:
            return Activations.Sigmoid(x) * (1 + x * Activations.Sigmoid(x) * exp(-x))
        def GELU(x) -> float:
            a = 0.450158
            b = 0.044715
            numerator = exp(2*a*(b*x*x*x + x)) * (exp(2*a*(b*x*x*x+x)) + 6*a*b*x*x*x + 2*a*x + 1)
            denominator = exp(2 * a * (b*x*x*x+x)) + 1
            return numerator / (denominator * denominator)

class CostFunctions:
    pass
    class Derivatives:
        pass

class LossFunctions:
    _SMALL_CONSTANT=1e-6
    def MAE(actual:np.ndarray, expected:np.ndarray):
        return np.sum(np.abs(actual - expected)) / len(actual)
    def MSE(actual:np.ndarray, expected:np.ndarray):
        return np.sum((actual - expected) ** 2) / len(actual)
    def RMSE(actual:np.ndarray, expected:np.ndarray):
        return np.sqrt(LossFunctions.MSE(actual, expected))
    def BinaryCrossentropy(actual:np.ndarray, expected:np.ndarray):
        epsilon = 1e-12
        expected = np.clip(expected, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(expected) + (1 - actual) * np.log(1 - expected))
    def CategoricalCrossentropy(actual:np.ndarray, expected:np.ndarray):
        epsilon = 1e-12
        expected = np.clip(expected, epsilon, 1) 
        return -np.sum(actual * np.log(expected)) / actual.shape[0]
    class Derivatives:
        def MAE(actual:np.ndarray, expected:np.ndarray):
            return np.where(actual > expected, 1, -1) / len(actual)
        def MSE(actual:np.ndarray, expected:np.ndarray):
            return 2 * (actual - expected) / len(actual)
        def RMSE(actual:np.ndarray, expected:np.ndarray):
            mse_grad = 2 * (actual - expected) / len(actual)
            return mse_grad / (2 * np.sqrt(np.sum((actual - expected) ** 2) / len(actual) + LossFunctions._SMALL_CONSTANT))
        def BinaryCrossentropy(actual:np.ndarray, expected:np.ndarray):
            epsilon = 1e-12
            expected = np.clip(expected, epsilon, 1 - epsilon)
            return -(actual / expected - (1 - actual) / (1 - expected)) / len(actual)
        def CategoricalCrossentropy(actual:np.ndarray, expected:np.ndarray):
            epsilon = 1e-12
            expected = np.clip(expected, epsilon, 1)
            return -actual / expected / actual.shape[0]




