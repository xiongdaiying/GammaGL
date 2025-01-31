import argparse
import numpy
import tensorlayerx as tlx
from tensorlayerx.dataflow import ConcatDataset
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets import Entities
import os.path as osp
from gammagl.datasets import TUDataset
from gammagl.loader import DataLoader
import numpy as np
from ...gammagl.models.rgat import RGATModel
# from rgat import RGATModel


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self._backbone(data['edge_index'], data['edge_type'],data['edge_attr'],data['num_nodes'], args.aggregation)
        train_logits = tlx.gather(logits, data['train_idx'])
        loss = self._loss_fn(train_logits, data['train_y'])
        return loss

def evaluate(net, data, y, metrics):
    net.set_eval()
    logits = net(data['edge_index'], data['edge_type'])
    _logits = tlx.gather(logits, data['test_idx'])
    _y = y  # tlx.gather(y, data['test_idx'])
    metrics.update(_logits, _y)
    acc = metrics.result()
    metrics.reset()
    return acc


# not use , perform on full graph
def k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes, relabel_nodes=False, flow='source_to_target'):
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = np.zeros(num_nodes, dtype=np.bool)
    edge_mask = np.zeros(row.shape[0], dtype=np.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = node_idx.flatten()
    else:
        node_idx = node_idx
    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill(False)
        node_mask[subsets[-1]] = True
        edge_mask = node_mask[row]
        subsets.append(col[edge_mask])

    subset, inv = np.unique(np.concatenate(subsets), return_inverse=True)
    numel = 1
    for n in node_idx.shape:
        numel *= n
    inv = inv[:numel]

    node_mask.fill(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = -np.ones((num_nodes,))
        node_idx[subset] = np.arange(subset.shape[0])
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

def main(args):
    # load dataset
    if str.lower(args.dataset) not in ['aifb', 'mutag', 'tox21']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if str.lower(args.dataset) == 'tox21':
        # Assays NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NRPPAR-gamma, SR-ARE, SR-ATAD5 were used for training.
        # Assays SR-HSE, SR-MMP, and SR-p53 were used for model evaluation.
        #
        TOX21_TRAIN_TASKS = [
            'Tox21_AhR_training', 'Tox21_AR_training', 'Tox21_AR-LBD_training', 'Tox21_ARE_training', 'Tox21_aromatase_training', 'Tox21_ATAD5_training',
            'Tox21_ER_training', 'Tox21_ER-LBD_training', 'Tox21_HSE_training', 'Tox21_MMP_training', 'Tox21_p53_training', 'Tox21_PPAR-gamma_training'
        ]
        TOX21_TEST_TASKS = [
            'Tox21_AhR_testing', 'Tox21_AR_testing', 'Tox21_AR-LBD_testing', 'Tox21_ARE_testing', 'Tox21_aromatase_testing', 'Tox21_ATAD5_testing',
            'Tox21_ER_testing', 'Tox21_ER-LBD_testing', 'Tox21_HSE_testing', 'Tox21_MMP_testing', 'Tox21_p53_testing', 'Tox21_PPAR-gamma_testing'
        ]
        TOX21_EVALUATION_TASKS = [
            'Tox21_AhR_evaluation', 'Tox21_AR_evaluation', 'Tox21_AR-LBD_evaluation', 'Tox21_ARE_evaluation', 'Tox21_aromatase_evaluation', 'Tox21_ATAD5_evaluation',
            'Tox21_ER_evaluation', 'Tox21_ER-LBD_evaluation', 'Tox21_HSE_evaluation', 'Tox21_MMP_evaluation', 'Tox21_p53_evaluation', 'Tox21_PPAR-gamma_evaluation'
        ]
        path = args.dataset_path

        datasets = [TUDataset(path, name=name) for name in TOX21_TRAIN_TASKS]
        train_dataset = ConcatDataset(datasets)

        datasets = [TUDataset(path, name=name) for name in TOX21_TEST_TASKS]
        test_dataset = ConcatDataset(datasets)

        datasets = [TUDataset(path, name=name) for name in TOX21_EVALUATION_TASKS]
        val_dataset = ConcatDataset(datasets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        net = RGATModel(feature_dim=data['num_nodes'],
                   hidden_dim=args.hidden_dim,
                   num_class=data['num_class'],
                   num_relations=data['num_relations'],
                   name="RGAT")
        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
        train_weights = net.trainable_weights
        loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
        best_val_acc = 0
        for epoch in range(args.n_epoch):
            net.set_train()
            for data in train_loader:
                train_loss = train_one_step(data, data.y)
            net.set_eval()
            total_correct = 0
            for data in val_loader:
                val_logits = net(data.x, data.edge_index, data.batch)
                # val_logits = net(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
                pred = tlx.argmax(val_logits, axis=-1)
                total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data.y).astype(int))))
            val_acc = total_correct / len(val_dataset)
            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val acc: {:.4f}".format(val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

        net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
        if tlx.BACKEND == 'torch':
            net.to(data['edge_index'].device)
        net.set_eval()
        total_correct = 0
        for data in test_loader:
            test_logits = net(data.x, data.edge_index, data.batch)
            # test_logits = net(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
            pred = tlx.argmax(test_logits, axis=-1)
            total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))
        test_acc = total_correct / len(test_dataset)
        print("Test acc:  {:.4f}".format(test_acc))

    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../Entities')
        dataset = Entities(path, args.dataset)
        graph = dataset[0]
        graph.numpy()

        node_idx = np.concatenate([graph.train_idx, graph.test_idx])
        node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, 2, graph.edge_index, graph.num_nodes, relabel_nodes=True)

        graph.num_nodes = node_idx.shape[0]
        graph.edge_index = edge_index
        graph.edge_type = graph.edge_type[edge_mask]
        graph.train_idx = mapping[:graph.train_idx.shape[0]]
        graph.test_idx = mapping[graph.train_idx.shape[0]:]
        graph.tensor()


        data = {
            'edge_index': graph.edge_index,
            'edge_type': graph.edge_type,
            'train_idx': graph.train_idx,
            'test_idx': graph.test_idx,
            'train_y': graph.train_y, 
            'test_y': graph.test_y, 
            'num_class': int(dataset.num_classes),
            'num_relations': dataset.num_relations,
            'num_nodes': graph.num_nodes,
            'edge_attr': None
        }

        net = RGATModel(feature_dim=32,
                        hidden_dim=args.hidden_dim,
                        num_class=data['num_class'],
                        num_relations=data['num_relations'],
                        heads = args.heads,
                        num_entity = graph.num_nodes,
                        num_bases=args.num_bases,
                        num_blocks=args.num_blocks,
                        attention_mechanism=args.attention_mechanism,
                        attention_mode=args.attention_mode,
                        dim=args.dim,
                        concat=args.concat,
                        negative_slope=args.negative_slope,
                        edge_dim=args.edge_dim,
                        add_bias=args.add_bias,
                        aggr = args.aggregation,
                        drop_rate=args.drop_rate,
                        name="RGAT")
        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
        metrics = tlx.metrics.Accuracy()
        train_weights = net.trainable_weights

        loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

        best_val_acc = 0
        for epoch in range(args.n_epoch):
            net.set_train()
            train_loss = train_one_step(data, data['train_y'])
            net.set_eval() 
            logits = net(data['edge_index'], data['edge_type'], data['edge_attr'],data['num_nodes'],args.aggregation) 
            val_logits = tlx.gather(logits, data['test_idx'])
            val_acc = calculate_acc(val_logits, data['test_y'], metrics)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val acc: {:.4f}".format(val_acc))

            # save best model on evaluation set
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

        net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict') 
        if tlx.BACKEND == 'torch':
            net.to(data['edge_index'].device)
        net.set_eval()
        # logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
        logits = net(data['edge_index'], data['edge_type'], data['edge_attr'], data['num_nodes'],args.aggregation)  # ����Ԥ����
        test_logits = tlx.gather(logits, data['test_idx'])
        test_acc = calculate_acc(test_logits, data['test_y'], metrics)
        print("Test acc:  {:.4f}".format(test_acc))

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument("--aggregation", type=str, default='sum', help='aggregate type')
    parser.add_argument('--dataset', type=str, default='mutag', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch')
    parser.add_argument("--heads", type=int, default=2, help="number of heads for stablization")
    parser.add_argument('--num_bases', type=int, default=None, help='number of bases')
    parser.add_argument('--num_blocks', type=int, default=None, help='number of blocks')
    parser.add_argument('--attention_mechanism', type=str, default='within-relation',
                        help='The attention mechanism to use within-relation or across-relation')
    parser.add_argument('--attention_mode', type=str, default='multiplicative-self-attention',
                        help='The mode to calculate attention logits additive-self-attention or multiplicative-self-attention')
    parser.add_argument('--dim', type=int, default=1, help='Number of dimensions for query and key kernels.')
    parser.add_argument('--concat', type=bool, default=True, help='If set to False, the multi-head attentions are averaged instead of concatenated.')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='LeakyReLU angle of the negative slope.')
    parser.add_argument('--edge_dim', type=int, default=None, help='Edge feature dimensionality.')
    parser.add_argument('--add_bias', type=bool, default=True, help='If set to False, the layer will notlearn an additive bias.')
    args = parser.parse_args()

    main(args)

