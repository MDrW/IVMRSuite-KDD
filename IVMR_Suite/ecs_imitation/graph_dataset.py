import os
import torch
import torch_geometric
import pickle
import numpy as np
import pdb


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        result_path, edge_path = filepath
        with open(result_path, "rb") as f:
            results = pickle.load(f)
            item_state, box_state, cur_placement, action, reward, done, cost_matrix, action_available, pre_actions = results
            item_one_hot = np.zeros(len(item_state), dtype=int)
            item_one_hot[action[0]] = 1
            box_one_hot = np.zeros(len(box_state), dtype=int)
            box_one_hot[action[1]] = 1
        
        with open(edge_path, 'rb') as f:
            edge_feats = pickle.load(f)
            item_assign_cost, item_mutex_box = edge_feats
        
        return item_state, box_state, cur_placement, item_assign_cost, item_mutex_box, item_one_hot, box_one_hot, cost_matrix, action_available, action, pre_actions
    
    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        item_state, box_state, cur_placement, item_assign_cost, item_mutex_box, item_one_hot, box_one_hot, cost_matrix, action_available, action, pre_actions = self.process_sample(self.sample_files[index])
        state = [item_state, box_state, cur_placement, item_assign_cost, item_mutex_box, cost_matrix, action_available, pre_actions]
        graph = self.process_and_get_graph(state)
        graph.file_name = self.sample_files[index][0]
        graph.item_sols = torch.FloatTensor(item_one_hot).reshape(-1)
        graph.box_sols = torch.FloatTensor(box_one_hot).reshape(-1)
        graph.actions = torch.LongTensor(action).reshape(1, 2)
        graph.box_masks = torch.LongTensor(action_available[action[0]].todense()).reshape(-1)
        return graph 
        
    def process_and_get_graph(self, state):
        item_state, box_state, cur_placement, item_assign_cost, item_mutex_box, cost_matrix, action_available, pre_actions = state
        item_features, box_features = item_state, box_state
        edge_features = np.concatenate([
            np.asarray(cur_placement.todense())[:, :, None],
            np.asarray(item_assign_cost.todense()+cost_matrix.todense())[:, :, None],
            # np.asarray(cost_matrix.todense())[:, :, None],
            # np.asarray(item_mutex_box.todense())[:, :, None],
            # np.asarray(action_available.todense())[:, :, None],
        ], axis=-1)
        
        nodes_connected = action_available.tocoo()
        
        nodes_connected = torch.sparse.FloatTensor(torch.LongTensor([
            nodes_connected.row.tolist(), nodes_connected.col.tolist()]),
            torch.FloatTensor(nodes_connected.data.astype(np.float32)))
        
        edge_indices = nodes_connected._indices()
        edge_features = edge_features[edge_indices[0], edge_indices[1]].reshape(-1, edge_features.shape[-1])
    
        graph = BipartiteNodeData(
            item_features=torch.FloatTensor(item_features),
            edge_indices=torch.LongTensor(edge_indices),
            edge_features=torch.FloatTensor(edge_features),
            box_features=torch.FloatTensor(box_features),
        )
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = item_features.shape[0] + box_features.shape[0]
        
        graph.nitems = item_features.shape[0]
        graph.nboxes = box_features.shape[0]
        graph.n_edge_feats = edge_features.shape[-1]
        graph.n_item_feats = item_features.shape[-1]
        graph.n_box_feats = box_features.shape[-1]
        graph.item_masks = torch.LongTensor(action_available.sum(axis=-1)).reshape(-1)
        graph.pre_actions = torch.LongTensor(pre_actions)
        graph.num_pre_acts = len(pre_actions)
        
        return graph


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(
            self,
            item_features,
            edge_indices,
            edge_features,
            box_features,

    ):
        super().__init__()
        self.item_features = item_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.box_features = box_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.item_features.size(0)], [self.box_features.size(0)]]
            )
        elif key == "candidates":
            return self.box_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
