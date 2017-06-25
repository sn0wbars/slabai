import numpy as np

class network:
    # struct - list of sizes of
    def __init__(self, struct, learn_rate = None ,func = None, func_der = None): 
        self.func = func if func is not None else sigmoid 
        self.func_der = func_der if func_der is not None else sigmoid_der
        self.learn_rate = learn_rate
        self.struct = struct
        self.num_layers = len(struct)
        self.biases = [np.random.randn(x,1) for x in struct[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(struct[1:],struct[:-1])]
        
    def cost_der(self, output, y):
        return output - y
    
    def backpropagation(self, x, y):
        a = np.asarray(x).reshape(-1,1)
        y = np.asarray(y).reshape(-1,1)
        a_list = [x]
        z_list = []
        for b, w in zip(self.biases, self.weights):
            z = w.dot(a) + b
            z_list.append(z)
            a = self.func(z)
            a_list.append(a)
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        nabla_b[-1] = self.cost_der(a, y) * self.func_der(z) #first derivative
        nabla_w[-1] = np.dot(nabla_b[-1], a_list[-2].transpose())
        
        for i in range(self.num_layers - 3, 0, -1):
            print(1)
            z = z_list[i]
            nabla_b[i] = np.dot(self.weights[i+1].transpose(), nabla_b[i+1]) * self.func_der(z)
            nabla_w[i] = np.dot(nabla_b[i+1], a_list[i-1].transpose())
        return (nabla_b, nabla_w)
        
    def feedforward(self, res):
        res = np.asarray(res).reshape(-1,1)
        for w, b in zip(self.weights, self.biases):
            res = self.func(w.dot(res) + b)
        return res
    
    def train(self, train_pair, learn_rate = 0.1):
        if self.learn_rate is None:
            self.learn_rate = learn_rate 
        
        nabla_b, nabla_w = self.backpropagation(*train_pair)
        self.weights = [w - learn_rate * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learn_rate * nb for b, nb in zip(self.biases, nabla_b)]    


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))