import ast
import math
import numpy as np
import random as r
import yaml

logsig = lambda val: 1 / (1 + math.exp(-val))
vlogsig = np.vectorize(logsig)
tanh = lambda val: 2 / (1 + math.exp(-2 * val)) - 1


class ANN(object):
    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

    def init_network(self, arch):
        self.arch = arch
        self.num_layers = len(arch)

        # initialize empty arrays
        self._init_tensors()

        weight_lim = (4*math.sqrt(6))/math.sqrt(self.arch[0]+self.arch[1])

        for i in range(self.num_layers-1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i+1]):
                    self.w[i][j][k] = r.uniform(-weight_lim, weight_lim)

        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                self.wb[i][j] = r.uniform(-weight_lim, weight_lim)

    @staticmethod
    def calc_act_fun(val, act_fun):
        if act_fun == 'logsig':
            return vlogsig(val)
        elif act_fun == 'tanh':
            return tanh(val)
        # elif act_fun == 'ReLU':
        #     output[i] == self.ReLU(output[i])

    def forward_prop(self, layer, input, act_fun):
        if layer == 0:
            self.H[layer] = input

        # output = [None] * self.arch[layer+1]
        # for i in range(self.arch[layer+1]):
        #     output[i] = self.calc_act_fun(np.dot(input, self.w[layer, :self.arch[layer], i]) + self.wb[layer][i], act_fun)

        _output = self.calc_act_fun(np.dot(input, self.w[layer, :self.arch[layer], :]) + self.wb[layer], act_fun)

        truncated_output = _output[:self.arch[layer+1]]

        self.H[layer + 1][:self.arch[layer+1]] = truncated_output
        self.act_funs[layer+1] = act_fun

        return truncated_output


    def back_prop(self, output, target):

        last_layer_neuron = self.arch[-1]
        p = np.zeros((self.num_layers, max(self.arch)))
        p[self.num_layers-1][:last_layer_neuron] = list(map(lambda i: (output[i] - target[i]) * self.d_act_fun(output[i], self.act_funs[self.num_layers-1]), range(last_layer_neuron)))

        for i in range(self.num_layers-2, 0, -1):
            p[i][:self.arch[i]] = np.dot(self.w[i][:self.arch[i]][:], p[i + 1]) * \
                                  self.d_act_fun(self.H[i][:self.arch[i]], self.act_funs[i])

            # for j in range(self.arch[i]):
            #     p[i][j] = np.dot(p[i+1], self.w[i][j])*self.d_act_fun(self.H[i][j], self.act_funs[i])
            #


            # p[i] = list(map(lambda j: np.dot(p[i+1], self.w[i][j])*self.d_act_fun(self.H[i][j], self.act_funs[i]), range(self.arch[i])))

            # for j in range(self.arch[i]):
            #     for k in range(self.arch[i+1]):
            #         p[i][j] += p[i+1][k]*self.w[i][j][k]
            #     p[i][j] *= self.d_act_fun(self.H[i][j], self.act_funs[i])


        for i in range(self.num_layers-1):
            # if i == 0:
            #     self.H[i] = self.image_data  # this is the magic line
            for j in range(self.arch[i]):
                self.w[i][j] -= np.dot(self.learning_rate * self.H[i][j], p[i+1])

            self.wb[i] -= np.dot(self.learning_rate, p[i+1])

    def load(self, filepath):
        assert '.yaml' in filepath
        print('Loading Network from {}'.format(filepath))
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)

        self.arch = data['metadata']['arch']
        self._init_tensors()

        weight_list = data['weights'].split(',')
        bias_weight_list = data['bias_weights'].split(',')

        for i in range(self.num_layers - 1):
            for j in range(self.arch[i]):
                for k in range(self.arch[i + 1]):
                    self.w[i][j][k] = float(weight_list.pop(0))

        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
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

        bwt_string = ''
        for i in range(self.num_layers-1):
            for j in range(self.arch[i+1]):
                bwt_string += str(self.wb[i][j]) + ','
        data['weights'] = wt_string
        data['bias_weights'] = bwt_string

        with open(filepath, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)


    @staticmethod
    def d_act_fun(val, function):
        if function == 'logsig':
            return val * (1-val)
        else:
            return 1

    def _init_tensors(self):
        self.num_layers = len(self.arch)
        self.H = np.zeros((self.num_layers, max(self.arch)))
        self.act_funs = [''] * self.num_layers

        self.w = np.zeros((self.num_layers-1,max(self.arch),max(self.arch)))
        self.wb = np.zeros((self.num_layers-1,max(self.arch)))

# TESTING
inp = [10.0, 11.0, 12.0]
tar = [0.0, 1.0, 0.0]
mlp = ANN(0.01)
mlp.init_network([3, 2, 3])
while True:
    x = mlp.forward_prop(0, inp, 'logsig')
    y = mlp.forward_prop(1, x, 'logsig')
    print(y)

    mlp.back_prop(y, tar)
    # exit()
