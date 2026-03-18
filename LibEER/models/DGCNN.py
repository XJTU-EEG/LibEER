import torch
import torch.nn as nn
import torch.utils.data
import yaml

param_path = 'config/model_param/DGCNN.yaml'


# EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks
# DGCNN paper link : https://ieeexplore.ieee.org/document/8320798
# T. Song, W. Zheng, P. Song and Z. Cui, "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks," in IEEE Transactions on Affective Computing, vol. 11, no. 3, pp. 532-541, 1 July-Sept. 2020, doi: 10.1109/TAFFC.2018.2817622.


class DGCNN(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, k=2, relu_is=1, layers=None, dropout_rate=0.5):
        # num_electrodes(int): The number of electrodes.
        # in_channels(int): The feature dimension of each electrode.
        # num_classes(int): The number of classes to predict.
        # k_(int): The number of graph convolutional layers.
        # relu_is(int): The function we use
        # out_channel(int): The feature dimension of  the graph after GCN.
        super(DGCNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.relu_is = relu_is
        self.get_param()
        if num_electrodes == 62:
            self.layers = [64]
        elif num_electrodes == 32:
            self.layers = [128]

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(GraphConv(self.k, self.in_channels, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.graphConvs.append(GraphConv(self.k, self.layers[i], self.layers[i + 1]))

        self.fc = nn.Linear(self.num_electrodes * self.layers[-1], 256, bias=True)
        self.fc2 = nn.Linear(256, self.num_classes, bias=True)
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList()
        if self.relu_is == 1:
            for i in range(len(self.layers)):
                self.b_relus.append(B1ReLU(self.layers[i]))
        elif self.relu_is == 2:
            for i in range(len(self.layers)):
                self.b_relus.append(B2ReLU(self.adj.shape[0], self.layers[i]))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weight()

    def get_param(self):
        try:
            fd = open(param_path, 'r', encoding='utf-8')
            model_param = yaml.load(fd, Loader=yaml.FullLoader)
            fd.close()
            self.k = model_param['params']['k']
            self.relu_is = model_param['params']['relu_is']
            self.layers = model_param['params']['layers']
            self.dropout_rate = model_param['params']['dropout']
            print("\nUsing setting from {}\n".format(param_path))
        except IOError:
            print("\n{} may not exist or not available".format(param_path))

        print("DGCNN Model, Parameters:\n")
        print("{:45}{:20}".format("k (The order of Chebyshev polynomials):", self.k))
        print("{:45}{:20}".format("relu_is (The type of B_Relu func):", self.relu_is))
        print("{:45}{:20}".format("layers (The channels of each layers):", str(self.layers)))
        print("{:45}{:20}\n".format("dropout rate:", self.dropout_rate))
        if self.k != 2 or self.relu_is != 1 or self.layers != [128] or self.dropout_rate != 0.5:
            print("Not Using Default Setting, the performance may be not the best")
        print("Starting......")

    def init_weight(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




def laplacian(w):
    """
    calculate the laplacian of the adjacency matrix
    :param w: the adjacency matrix
    :return: l: the normalized Laplacian matrix
    """
    # d is the sum of each row of a matrix.
    d = torch.sum(w, dim=1)
    # reciprocal square root of a vector
    d_re = 1 / torch.sqrt(d + 1e-5)
    # create a matrix with the d_re vector as its diagonal elements
    d_matrix = torch.diag_embed(d_re)
    # calculate the laplacian matrix
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap


class GraphConv(nn.Module):
    """
    Graph convolution based on Chebyshev polynomials
    """

    def __init__(self, k, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        # self.truncated_normal_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        """
        calculate the chebyshev polynomial
        :param x : input x
        :param lap: the input laplacian matrix
        :return: the chebyshev polynomial components
        """
        t = torch.ones(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
        if self.k == 1:
            return t.unsqueeze(1)
        if self.k == 2:
            return torch.cat((x.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        elif self.k > 2:
            # T_0 of chebyshev polynomials, just x (identity matrix multiply x), shape: (batch, ele_channel, in_channel)
            tk_minus_one = x
            # T_1 of chebyshev polynomials, shape: (batch, ele_channel, in_channel)
            tk = torch.matmul(lap, x)
            # add the T_0, T_1, T_2 items to the Chebyshev components, t shape: (batch, 3, ele_channel, in_channel)
            t = torch.cat((x.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)
            for i in range(3, self.k):
                # T_(k-1) and T_(k-2)
                tk_minus_two, tk_minus_one = tk_minus_one, tk
                # calculate the T_(k), shape: (batch, ele_channel, in_channel)
                tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
                # add the T_k items to the Chebyshev components, shape: (batch, i+1, ele_channel, in_channel)
                t = torch.cat((t, tk.unsqueeze(1)), dim=1)
            return t

    def forward(self, x, lap):
        """
        :param x: (batch_size, ele_channel, in_channel)
        :param lap: the laplacian matrix
        :return: the result of Graph conv
        """
        # obtain the chebyshev polynomial, t shape: (batch, k, ele_channel, in_channel)
        cp = self.chebyshev_polynomial(x, lap)
        # transpose cp to: (batch, ele_channel, in_channel, k)
        cp = cp.permute(0, 2, 3, 1)
        # reshape cp to: (batch, ele_channel, in_channel * k)
        cp = cp.flatten(start_dim=2)
        # perform filter operation of order K
        out = torch.matmul(cp, self.weight)
        return out

class NewSparseL2Regularization(nn.Module):
    def __init__(self, l2_lambda):
        super(NewSparseL2Regularization, self).__init__()
        self.l2_lambda = l2_lambda
    def forward(self, x):
        l2_reg = torch.tensor(0.).to(next(x.parameters()).device)
        for param in x.parameters():
            l2_reg += torch.norm(param)
        return l2_reg * self.l2_lambda

class SparseL2Regularization(nn.Module):
    def __init__(self, l2_lambda):
        super(SparseL2Regularization, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, x):
        l2_norm = torch.norm(x, p=2)
        return self.l2_lambda * l2_norm


class B1ReLU(nn.Module):
    def __init__(self, bias_shape):
        super(B1ReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)


class B2ReLU(nn.Module):
    def __init__(self, bias_shape1, bias_shape2):
        super(B2ReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, bias_shape1, bias_shape2))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)

