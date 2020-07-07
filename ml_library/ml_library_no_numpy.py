""" This is the ANN implementation without
numpy due to pypy7.3/numpy incompatibility """

import random as r
import math
import yaml


# activation functions
logsig = lambda val: 1 / (1 + exp(-val))
tanh = lambda val: 2 / (1 + exp(-2 * val)) - 1
relu = lambda val: val if val > 0 else 0
null = lambda val: val


def exp(val):
    try:
        result = math.exp(val)
    except OverflowError:
        result = 0
    return result


def _zero2(x, y):
    return [[0 for _ in range(y)] for _ in range(x)]


def _zero3(x, y, z):
    return [[[0 for _ in range(z)] for _ in range(y)] for _ in range(x)]


class ANN(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def init_network(self, arch):
        self.arch = arch
        self.num_layers = len(arch)

        # initialize empty arrays
        self._init_tensors()

        for i in range(self.num_layers-1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i+1]):
                    self.w[i][j][k] = r.uniform(-self.weight_lim, self.weight_lim)

        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                self.wb[i][j] = r.uniform(-self.weight_lim, self.weight_lim)
                self.wr[i][j] = r.uniform(-self.weight_lim, self.weight_lim)

    def _init_tensors(self):
        self.num_layers = len(self.arch)
        self.H = _zero2(self.num_layers, max(self.arch))
        self.last_H = _zero2(self.num_layers, max(self.arch))
        self.act_funcs = [None] * self.num_layers
        self.w = _zero3(self.num_layers - 1, max(self.arch), max(self.arch))
        self.wr = _zero2(self.num_layers - 1, max(self.arch))
        self.wb = _zero2(self.num_layers - 1, max(self.arch))

        # calculate weight limit for initializing random weights
        self.weight_lim = (4*math.sqrt(6))/math.sqrt(self.arch[0]+self.arch[1])

    def forward_prop(self, layer, input, act_func):
        output = []
        for j in range(self.arch[layer+1]):
            value = 0
            for i in range(self.arch[layer]):
                value += input[i] * self.w[layer][i][j]
            # output.append(act_func(value + self.wr[layer][j] * self.H[layer + 1][j] + self.wb[layer][j]))
            output.append(act_func(value + self.wb[layer][j]))

        if layer == 0:
            self.H[layer] = input
        self.H[layer + 1] = output

        self.act_funcs[layer + 1] = act_func
        return output

    def back_prop(self, output, target):
        last_layer_neuron = self.arch[-1]
        p = _zero2(self.num_layers, max(self.arch))

        # todo incorporate recurrent weight back prop
        for i in range(last_layer_neuron):
            p[self.num_layers-1][i] = (output[i] - target[i]) * self.d_act_func(output[i], self.act_funcs[-1])

        for i in range(self.num_layers-2, 0, -1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i+1]):
                    p[i][j] += p[i+1][k]*self.w[i][j][k]
                p[i][j] *= self.d_act_func(self.H[i][j], self.act_funcs[i])

        # update weights
        for i in range(self.num_layers-1):
            # if i == 0:
            #     self.H[i] = self.image_data  # this is the magic line
            for j in range(self.arch[i]):
                for k in range(self.arch[i+1]):
                    self.w[i][j][k] -= self.learning_rate * p[i+1][k] * self.H[i][j]

        # update bias weights
        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                # todo, update recurrent weights
                self.wb[i][j] -= self.learning_rate * p[i+1][j]

    def load(self, filepath):
        assert '.yaml' in filepath
        print('Loading Network from {}'.format(filepath))
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)

        self.arch = data['metadata']['arch']
        self._init_tensors()

        weight_list = data['weights'].split(',')
        # recurrent_weight_list = data['recurrent_weights'].split(',')  # todo uncomment
        bias_weight_list = data['bias_weights'].split(',')

        for i in range(self.num_layers - 1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i + 1]):
                    self.w[i][j][k] = float(weight_list.pop(0))

        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                # self.wr[i][j] = float(recurrent_weight_list.pop(0))  # todo uncomment
                self.wb[i][j] = float(bias_weight_list.pop(0))

    def save(self, filepath):
        assert '.yaml' in filepath
        print('Saving Network to {}'.format(filepath))
        data = {}
        data['metadata'] = {}
        data['metadata']['arch'] = self.arch

        wt_string = ''
        for i in range(self.num_layers - 1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i + 1]):
                    wt_string += str(self.w[i][j][k]) + ','

        wr_string = ''
        bwt_string = ''
        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                wr_string += str(self.wr[i][j]) + ','
                bwt_string += str(self.wb[i][j]) + ','
        data['weights'] = wt_string
        data['recurrent_weights'] = wr_string
        data['bias_weights'] = bwt_string

        with open(filepath, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

    @staticmethod
    def d_act_func(val, function):
        if function == logsig:
            return val * (1-val)
        else:
            return 1


# # TESTING
# inp = [10.0, 11.0, 12.0]
# tar = [0.0, 1.0, 0.0]
# mlp = ANN(0.1)
# mlp.init_network([3, 2, 3])
# while True:
#     x = mlp.forward_prop(0, inp, logsig)
#     y = mlp.forward_prop(1, x, logsig)
#     print(y)
#
#     mlp.back_prop(y, tar)
#     # exit()
