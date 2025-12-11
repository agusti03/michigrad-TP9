import random
from michigrad.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        # realiza la forward pass de la neurona 
        # x es un arreglo
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) 
            # zip combina wi con xi

        return act
        #return act.relu() if self.nonlin else act

    def parameters(self):
        # retorna una lista tal que [w0, w1, ..., wn, b]
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer(Module):
    # Define una capa de neuronas

    def __init__(self, nin, nout, nonlin="relu", **kwargs, ):
        # recibe cantidad de entradas (nin) y salidas (nout)
        # p.e : Layer(2, 1) es una neurona que recibe dos entradas y tiene una salida
        # puede recibir nonlin
        self.neurons = [Neuron(nin) for _ in range(nout)] 
            # lista de neuronas
            # todas con la misma cantidad de entradas y salidas
        self.nonlin = nonlin # se agrega como atributo el tipo de función de activación

    def __call__(self, x):
        # aplica la función de activación a cada capa
        # x es el valor de la entrada
        match self.nonlin:
            case "relu": out = [n(x).relu() for n in self.neurons] 
            case "sigmoid": out = [n(x).sigmoid() for n in self.neurons]
            case "tanh": out = [n(x).tanh() for n in self.neurons]
            case _: out = [n(x) for n in self.neurons] 
            
        return out[0] if len(out) == 1 else out
            # out es un arreglo de tamaño nout

    def parameters(self):
        # obtener la lista de parametros de todas las neuronas
        return [p for n in self.neurons for p in n.parameters()] # recorrido de lista de listas
    
    def __repr__(self):
        match self.nonlin:
            case "relu": return f"ReLU Layer of [{', '.join(str(n) for n in self.neurons)}]"
            case "sigmoid": return f"Sigmoid Layer of [{', '.join(str(n) for n in self.neurons)}]"
            case "tanh": return f"Tanh Layer of [{', '.join(str(n) for n in self.neurons)}]"
            case _: return f"Linear Layer of [{', '.join(str(n) for n in self.neurons)}]"  
class MLP(Module):
    # perceptron multicapa
    # p.e : MLP(2, [3, 3, 1]) es un ppn con 
    #   capa de entrada dos entradas
    #   una hidden con 3 entradas
    #   segunda hidden con 3 entradas
    #   capa de salida con una entrada


    """
    def __init__(self, nin, nouts, nonlin=True):
        sz = [nin] + nouts # lista que se recorre para crear las capas
        self.layers = []
        for i in range(len(nouts)):
            # aca la idea es que si nonlin es False se creen las capas con nonlin=False
            # y que si es True, la última capa no tenga función de activación
            layer_nonlin = nonlin
            if(nonlin):
                layer_nonlin = i != len(nouts) - 1 # la última capa no tiene función de activación
        
            layer = Layer(sz[i], sz[i+1], nonlin=layer_nonlin)
            self.layers.append(layer)
            # cada capa se crea con sus entradas y las entradas de la capa siguiente
            # capa de entrada: 2, 3
            # hidden 1 : 3, 3
            # hidden 2 : 3, 1
            # salida : 1 (no tiene salida)

    """
    def __init__(self, nin, nouts, nonlin=True):
        sz = [nin] + nouts 
        self.layers = []
        for i in range(len(nouts)-1):
            layer = Layer(sz[i], sz[i+1], nonlin=nonlin)
            self.layers.append(layer)
        layer = Layer(sz[-2], sz[-1], nonlin="linear") # última capa sin función de activación
        self.layers.append(layer)
    

    def __call__(self, x):
        # llama a la función que define cada Layer
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # obtiene la lista de parámetros de todas las capas (de todas las neuronas)
        return [p for layer in self.layers for p in layer.parameters()] # recorrido de lista de listas

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"