import math
import tensorlayerx as tlx
from gammagl.layers.conv import GCNIIConv


class GCNIIModel(tlx.nn.Module):
    """GCN with Initial residual and Identity mapping"""

    def __init__(self, feature_dim, hidden_dim, num_class, num_layers,
                 alpha, beta, lambd, variant, keep_rate, name=None):
        super().__init__(name=name)

        self.linear_head = tlx.layers.Dense(n_units=hidden_dim,
                                            in_channels=feature_dim,
                                            b_init=None)
        self.linear_tail = tlx.layers.Dense(n_units=num_class,
                                            in_channels=hidden_dim,
                                            b_init=None)

        self.convs = tlx.nn.SequentialLayer()
        for i in range(1, num_layers + 1):
            # beta = lambd / i if variant else beta
            beta = math.log(lambd / i + 1) if variant else beta
            self.convs.append(GCNIIConv(hidden_dim, hidden_dim, alpha, beta, variant))

        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(keep_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x0 = x = self.relu(self.linear_head(self.dropout(x)))
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x0, x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)
        x = self.relu(self.linear_tail(self.dropout(x)))
        return x