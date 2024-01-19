import tensorlayerx as tlx
from ..layers.conv.rgat_conv import RGATConv
# from rgat_conv import RGATConv

class RGATModel(tlx.nn.Module):
    """relational graph convoluation nerworks"""

    def __init__(self, feature_dim, hidden_dim, num_class, num_relations, heads, num_entity, num_bases,
                 num_blocks, attention_mechanism, attention_mode, dim, concat, negative_slope,
                 edge_dim, add_bias, aggr, drop_rate, name=None):
        super().__init__(name=name)
        self.conv1 = RGATConv(in_channels=feature_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases = num_bases,
                              num_blocks = num_blocks,  attention_mechanism = attention_mechanism, attention_mode = attention_mode,
                              heads = heads, dim = dim, concat = concat, negative_slope = negative_slope,
                              edge_dim = edge_dim, add_bias = add_bias, aggr = aggr, drop_rate = drop_rate)
        self.conv2 = RGATConv(in_channels=hidden_dim * heads, out_channels=num_class, num_relations=num_relations, num_bases=num_bases,
                              num_blocks=num_blocks, attention_mechanism=attention_mechanism, attention_mode=attention_mode,
                              heads=heads, dim=dim, concat=concat, negative_slope=negative_slope,
                              edge_dim=edge_dim, add_bias=add_bias, aggr = aggr, drop_rate = drop_rate)
        self.relu = tlx.ReLU()
        self.init_input = tlx.random_normal(shape=(num_entity, feature_dim), dtype=tlx.float32)

    def forward(self, edge_index, edge_type, edge_attr, num_nodes, aggr):
        x = self.conv1(self.init_input, edge_index, edge_type, edge_attr, num_nodes, aggr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_type, edge_attr, num_nodes, aggr)
        return x
