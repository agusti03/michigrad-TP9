import random
from michigrad.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # realiza la forward pass de la neurona 
        # x es un arreglo
        # pasar nonlin = False para usar función de activación lineal
        # nonlin = True usa ReLU
        # puede modificarse para implementar cualquier función de activ.
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) 
            # zip combina wi con xi
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    # Define una capa de neuronas

    def __init__(self, nin, nout, **kwargs):
        # recibe cantidad de entradas (nin) y salidas (nout)
        # p.e : Layer(2, 1) es una neurona que recibe dos entradas y tiene una salida
        # puede recibir nonlin
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] 
            # lista de neuronas
            # todas con la misma cantidad de entradas y salidas

    def __call__(self, x):
        # aplica la función de activación a cada neurona de la capa
        # x es el valor de la entrada
        out = [n(x) for n in self.neurons] 
            # se aplica la función de activación a cada neurona
            # se llama a Neuron.__call__(x)
            
        return out[0] if len(out) == 1 else out
            # out es un arreglo de tamaño nout

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # la falsedad de la expresividad de python

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    # perceptron multicapa
    # p.e : MLP(2, [3, 3, 1]) es un ppn con 
    #   capa de entrada dos entradas
    #   una hidden con 3 entradas
    #   segunda hidden con 3 entradas
    #   capa de salida con una entrada
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # lista que se recorre para crear las capas
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
            # cada capa se crea con sus entradas y las entradas de la capa siguiente
            # capa de entrada: 2, 3
            # hidden 1 : 3, 3
            # hidden 2 : 3, 1
            # salida : 1 (no tiene salida)

    def __call__(self, x):
        # llama a la función que define cada Layer
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"