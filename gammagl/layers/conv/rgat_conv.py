import tensorlayerx as tlx
from ..conv import MessagePassing
from gammagl.utils import *
from gammagl.mpops import *


class RGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases=None,  # basis decomposition正则化
                 num_blocks=None,  # block-diagonal-decomposition正则化
                 attention_mechanism="across-relation",  # within or across relation
                 attention_mode="additive-self-attention",  # add or multiple
                 heads=1,
                 dim=1,  # q&k kernel的维数
                 concat=True, # multi-head attention输出时，选择concat还是mean
                 negative_slope=0.2, # LeakyReLUYongde
                 edge_dim=None, # 边特征的维数
                 add_bias=True,
                 drop_rate = 0.,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.attention_mechanism = attention_mechanism
        self.attention_mode = attention_mode
        self.heads = heads
        self.dim = dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        self.add_bias = add_bias
        self.dropout_rate = drop_rate
        self.aggr = kwargs.get('aggr', 'sum')

        if (self.attention_mechanism != "within-relation"
                and self.attention_mechanism != "across-relation"):
            raise ValueError('attention mechanism must either be '
                             '"within-relation" or "across-relation"')

        if (self.attention_mode != "additive-self-attention"
                and self.attention_mode != "multiplicative-self-attention"):
            raise ValueError('attention mode must either be '
                             '"additive-self-attention" or '
                             '"multiplicative-self-attention"')

        if self.attention_mode == "additive-self-attention" and self.dim > 1:
            raise ValueError('"additive-self-attention" mode cannot be '
                             'applied when value of d is greater than 1. '
                             'Use "multiplicative-self-attention" instead.')

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        initor = tlx.initializers.TruncatedNormal()
        # 初始化注意力核q和k
        self.q = self._get_weights("q", shape=(self.heads * self.out_channels , self.heads * self.dim), init=initor)
        self.k = self._get_weights("k", shape=(self.heads * self.out_channels , self.heads * self.dim), init=initor)
        # 初始化bias
        if self.add_bias and self.concat:
            self.bias = self._get_weights("bias",shape=(self.heads * self.dim * self.out_channels,), init=initor)
        elif self.add_bias and not self.concat:
            self.bias = self._get_weights("bias",shape=(self.dim * self.out_channels,), init=initor)
        else:
            self.bias = None

        # 如果egde_dim是多维的
        if self.edge_dim is not None:
            self.lin_edge = tlx.layers.Linear(out_features=self.out_channels * self.heads,
                                        in_features=self.edge_dim,
                                        b_init=None)
            self.e = self._get_weights("e", shape=(self.heads * self.out_channels, self.heads * self.dim), init=initor)
        else:
            self.lin_edge = None
            self.e = None

        # 如果选择basis-decomposition
        # 初始化att和basis
        if num_bases is not None:
            self.att = self._get_weights("att", shape=(self.num_relations, self.num_bases),init=initor,order=True)
            self.basis = self._get_weights("basis", shape=(self.num_bases, self.in_channels,self.heads*self.out_channels),init=initor,order=True)
        # 如果选择block-diag-decomposition
        elif num_blocks is not None:
            assert (
                    self.in_channels % self.num_blocks == 0
                    and (self.heads * self.out_channels) % self.num_blocks == 0), (
                "both 'in_channels' and 'heads * out_channels' must be "
                "multiple of 'num_blocks' used")
            self.weight = self._get_weights("weight", shape=(self.num_relations, self.num_blocks,
                            self.in_channels // self.num_blocks,
                            (self.heads * self.out_channels) //
                            self.num_blocks),init=initor,order=True)
        else:
            self.weight = self._get_weights("weight",shape=(self.num_relations, self.in_channels, self.heads * self.out_channels),init=initor,order=True)

        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)
        self._alpha = None

    def message(self, x, edge_index, edge_attr = None, edge_type = None, num_nodes=None ):
        x_i = tlx.gather(x, edge_index[0, :])
        x_j = tlx.gather(x, edge_index[1, :])
        # x_i = tlx.reshape(x_i, shape=(-1, self.heads * self.dim, self.out_channels))
        # x_j = tlx.reshape(x_j, shape=(-1, self.heads * self.dim, self.out_channels))

        # x_i = tlx.reshape(x_i, [-1, self.in_channels])
        # x_j = tlx.reshape(x_j, [-1, self.in_channels])

        if self.num_bases is not None:  # Basis-decomposition =================
            w = tlx.matmul(self.att, tlx.reshape(self.basis, [self.num_bases, -1]))
            w = tlx.reshape(w,[self.num_relations, self.in_channels, self.heads * self.out_channels])
        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if (x_i.dtype == tlx.int64 and x_j.dtype == tlx.int64
                and self.num_blocks is not None):
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')
            w = self.weight
            x_i = tlx.reshape(x_i,[-1, 1, w.shape[1], w.shape[2]])
            x_j = tlx.reshape(x_j,[-1, 1, w.shape[1], w.shape[2]])
            w = tlx.gather(w, edge_type)
            outi = tlx.einsum('abcd,acde->ace', x_i, w)
            outi = tlx.reshape(outi,[-1, self.heads * self.out_channels])
            outj = tlx.einsum('abcd,acde->ace', x_j, w)
            outj = tlx.reshape(outj,[-1, self.heads * self.out_channels])
        else:  # No regularization/Basis-decomposition ========================
            if self.num_bases is None:
                w = self.weight
            w = tlx.gather(w, edge_type)
            # outi = w*hi
            # outj = w*hj
            outi = tlx.squeeze(tlx.matmul(tlx.expand_dims(x_i, axis=1), w), axis=-2)
            outj = tlx.squeeze(tlx.matmul(tlx.expand_dims(x_j, axis=1), w), axis=-2)
            # outi = tlx.squeeze(tlx.matmul(x_i, w), axis=-2)
            # outj = tlx.squeeze(tlx.matmul(x_j, w), axis=-2)
        # qi = w*hi*Q
        # kj = w*hj*K
        qi = tlx.matmul(outi, self.q)
        kj = tlx.matmul(outj, self.k)

        alpha_edge, alpha = 0, tlx.zeros([1])
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None, (
                "Please set 'edge_dim = edge_attr.size(-1)' while calling the "
                "RGATConv layer")
            edge_attributes = tlx.reshape(self.lin_edge(edge_attr),[-1, self.heads * self.out_channels])
            if edge_attributes.size(0) != edge_attr.size(0):
                edge_attributes = tlx.gather(edge_attributes, edge_type)
            alpha_edge = tlx.matmul(edge_attributes, self.e)

        # additive or multiplicative
        if self.attention_mode == "additive-self-attention":
            if edge_attr is not None:
                alpha = tlx.add(qi, kj) + alpha_edge
            else:
                alpha = tlx.add(qi, kj)
            alpha = self.leaky_relu(alpha)
        elif self.attention_mode == "multiplicative-self-attention":
            if edge_attr is not None:
                alpha = (qi * kj) * alpha_edge
            else:
                alpha = qi * kj

        # within or across relation
        if self.attention_mechanism == "within-relation":
            across_out = tlx.zeros_like(alpha)
            for r in range(self.num_relations):
                mask = tlx.equal(edge_type, r)
                dst_index = edge_index[1, :]
                segment_ids = tf.boolean_mask(dst_index, mask)
                soft_data = tf.boolean_mask(alpha, mask)
                print(alpha[mask])
                print(dst_index[mask])
                # assert tf.reduce_max(segment_ids) < num_nodes
                # across_out[mask] = segment_softmax(alpha[mask], segment_ids, num_nodes)
                across_out[mask] = segment_softmax(soft_data, segment_ids, num_nodes)
            #     mask = edge_type == r
            #     # segment_ids = edge_index[1, mask]
            #     # tlx.reshape(segment_ids, (-1,))
            #     across_out[mask] = segment_softmax(alpha[mask],edge_index[1, :] , num_nodes)
            alpha = across_out
        elif self.attention_mechanism == "across-relation":
            alpha = segment_softmax(alpha, edge_index[1, :], num_nodes)
            # alpha = tlx.softmax(alpha,axis=1)

        # self._alpha = alpha
        if self.dropout_rate > 0:
            alpha = self.dropout(alpha)
        else:
            alpha = alpha  # original

        if self.attention_mode == "additive-self-attention":
            return (tlx.reshape(alpha,[-1, self.heads, 1]) * tlx.reshape(outj,[-1, self.heads, self.out_channels]))
        else:
            return (tlx.reshape(alpha,[-1, self.heads, self.dim, 1]) * tlx.reshape(outj,[-1, self.heads, 1, self.out_channels]))

    def forward(self, x, edge_index, edge_type=None, edge_attr=None, num_nodes = None, aggr = 'sum'):
        # propagate_type: (x: Tensor, edge_type: OptTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,num_nodes = num_nodes, aggr = aggr)
        return out

    def propagate(self, x, edge_index, edge_type, edge_attr, num_nodes, aggr='sum', fuse_kernel=False,  **kwargs):
        """
        Function that perform message passing.
        Args:
            x: input node feature
            edge_index: edges from src to dst
            aggr: aggregation type, default='sum', optional=['sum', 'mean', 'max']
            fuse_kernel: use fused kernel function to speed up, default = False
            kwargs: other parameters dict

        """

        # if 'num_nodes' not in kwargs.keys() or kwargs['num_nodes'] is None:
        #     kwargs['num_nodes'] = x.shape[0]

        coll_dict = self.__collect__(x, edge_index, edge_type, edge_attr, num_nodes, aggr, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        msg_kwargs['edge_attr'] = edge_attr
        msg_kwargs['num_nodes'] = num_nodes
        msg_kwargs['edge_type'] = edge_type
        msg = self.message(**msg_kwargs)
        x = self.aggregate(msg, edge_index, num_nodes=num_nodes, aggr=aggr)
        x = self.update(x)
        return x

    def update(self,x):
        if self.attention_mode == "additive-self-attention":
            if self.concat is True:
                x = tlx.reshape(x,[-1, self.heads * self.out_channels])
            else:
                x = tlx.reduce_mean(x, axis=1)

            if self.bias is not None:
                x = x + self.bias

            return x
        else:
            if self.concat is True:
                x = tlx.reshape(x,[-1, self.heads * self.dim * self.out_channels])
            else:
                x = tlx.reduce_mean(x, axis=1)
                x = tlx.reshape(x,[-1, self.dim * self.out_channels])

            if self.bias is not None:
                x = x + self.bias

            return x

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum'):
        """
        Function that aggregates message from edges to destination nodes.

        Parameters
        ----------
        msg: tensor
            message construct by message function.
        edge_index: tensor
            edges from src to dst.
        num_nodes: int
            number of nodes of the graph.
        aggr: str
            aggregation type, default = 'sum', optional=['sum', 'mean', 'max'].

        Returns
        -------
        tensor
            output representation.

        """
        dst_index = edge_index[1, :]
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_nodes)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_nodes)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_nodes)
        elif aggr == 'min':
            return unsorted_segment_min(msg, dst_index, num_nodes)
        else:
            raise NotImplementedError('Not support for this opearator')

    def __collect__(self, x, edge_index, edge_type, edge_attr, num_nodes, aggr, kwargs):
        out = {}

        for k, v in kwargs.items():
            out[k] = v
        out['x'] = x
        out['edge_index'] = edge_index
        out['aggr'] = aggr
        out['edge_type'] = edge_type
        out['edge_attr'] = edge_attr
        out['num_nodes'] = num_nodes

        return out