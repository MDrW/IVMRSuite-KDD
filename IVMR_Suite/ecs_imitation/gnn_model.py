import os
import torch
import torch_geometric
import pickle
import numpy as np
import pdb


class GNNPolicy(torch.nn.Module):
    def __init__(self, 
        n_item_feats, 
        n_box_feats, 
        n_edge_feats,
        emb_size_list=[64, 32],
        out_emb_size_list=[32],
        n_gnn_layers=2,
        is_use_full_infos=False,
        is_gnn_resnet=False, 
        graph_type='gcn',
        normalization='batch',
        activate_func='LeakyReLU',
        is_box_outfeats_use_item=True,
        is_item_use_pre_acts=False,
    ):
        super().__init__()
        self.is_use_full_infos = is_use_full_infos
        self.is_gnn_resnet = is_gnn_resnet
        self.n_gnn_layers = n_gnn_layers
        self.is_box_outfeats_use_item = is_box_outfeats_use_item
        self.is_item_use_pre_acts = is_item_use_pre_acts
        
        # if graph_type == 'gat':
        #     Graph = GATConvolution
        # elif graph_type == 'gin':
        #     Graph = GINConvolution
        # else:
        #     Graph = BipartiteGraphConvolution
        
        self.activate_func = getattr(torch.nn, activate_func)
        
        def _construct_fcn(embsize_list, name, norm):
            fully_connected_net = torch.nn.Sequential()
            if norm == 'batch':
                fully_connected_net.add_module(name=f'{name}_batchnorm', module=torch.nn.BatchNorm1d(embsize_list[0]))
            elif norm == 'layer':
                fully_connected_net.add_module(name=f'{name}_layernorm', module=torch.nn.LayerNorm(embsize_list[0]))
            for i in range(len(embsize_list)-1):
                fully_connected_net.add_module(name=f'{name}_linear{i}', module=torch.nn.Linear(embsize_list[i], embsize_list[i+1]))
                fully_connected_net.add_module(name=f'{name}_activate{i}', module=self.activate_func())
            return fully_connected_net

        # Item Embedding
        item_embsize_list = [n_item_feats] + emb_size_list
        self.item_embedding = _construct_fcn(item_embsize_list, name='item_emb', norm=normalization)
        
        # Box Embedding
        box_embsize_list = [n_box_feats] + emb_size_list
        self.box_embedding = _construct_fcn(box_embsize_list, name='box_emb', norm=normalization)

        # EDGE EMBEDDING
        edge_embsize_list = [n_edge_feats] + emb_size_list
        self.edge_embedding = _construct_fcn(edge_embsize_list, name='edge_emb', norm=normalization)
        
        self.emb_size = emb_size_list[-1]
        self.gnn_layers = torch.nn.ModuleList()
        
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(GraphLayer(self.emb_size, graph_type))
        #     gnn_item_to_box = Graph(emb_size=self.emb_size)
        #     gnn_box_to_item = Graph(emb_size=self.emb_size)
        #     self.gnn_layers.append([gnn_item_to_box, gnn_box_to_item])
        
        if self.is_gnn_resnet:
            gnn_out_size = self.emb_size * (n_gnn_layers+1)
        else:
            gnn_out_size = self.emb_size
        
        if self.is_use_full_infos:
            output_embed_size = gnn_out_size * 3
        else:
            output_embed_size = gnn_out_size
        
        if self.is_item_use_pre_acts:
            item_outemb_size = output_embed_size + gnn_out_size * 3
        else:
            item_outemb_size = output_embed_size
        item_outsize_list = [item_outemb_size] + out_emb_size_list
        self.item_output = _construct_fcn(item_outsize_list, name='item_out', norm='batch')
        self.item_output.add_module(name=f'item_out_final', module=torch.nn.Linear(item_outsize_list[-1], 1))
        
        if is_box_outfeats_use_item:
            box_outemb_size = output_embed_size + gnn_out_size
        else:
            box_outemb_size = output_embed_size
        box_outsize_list = [box_outemb_size] + out_emb_size_list
        self.box_output = _construct_fcn(box_outsize_list, name='box_out', norm='batch')
        self.box_output.add_module(name=f'box_out_final', module=torch.nn.Linear(box_outsize_list[-1], 1)) 

    def forward(self, item_features, edge_indices, edge_features, box_features, 
                selected_item_idxes, n_items_list, n_boxes_list, n_pre_actions=None, pre_actions=None):
        # reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension
        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        box_features = self.box_embedding(box_features)

        # Two half convolutions
        items_feats, boxes_feats = [item_features], [box_features]
        
        for gnn_layer in self.gnn_layers:
            this_item_feats, this_box_feats = gnn_layer(items_feats[-1], edge_indices, edge_features, boxes_feats[-1])
            # this_box_feats = gnn_layer[0](
            #     items_feats[-1], edge_indices, edge_features, boxes_feats[-1]
            # )
            boxes_feats.append(this_box_feats)
            
            # this_item_feats = gnn_layer[1](
            #     boxes_feats[-1], reversed_edge_indices, edge_features, items_feats[-1]
            # )
            items_feats.append(this_item_feats)
        
        if self.is_gnn_resnet:
            item_features = torch.cat(items_feats, dim=-1)
            box_features = torch.cat(boxes_feats, dim=-1)
        else:
            item_features = items_feats[-1]
            box_features = boxes_feats[-1]
        
        i_index_arrow, b_index_arrow = 0, 0
        ind, p_idx = 0, 0
        items_output_feats, boxes_output_feats = [], []
        for n_items, n_boxes, item_idx in zip(n_items_list, n_boxes_list, selected_item_idxes):
            this_items = item_features[i_index_arrow:i_index_arrow+n_items, :]
            this_boxes = box_features[b_index_arrow:b_index_arrow+n_boxes, :]
            
            if self.is_use_full_infos:
                this_items_full = this_items.mean(dim=0).unsqueeze(0)
                this_boxes_full = this_boxes.mean(dim=0).unsqueeze(0)
                this_items_output = torch.cat([this_items, this_items_full.repeat(n_items, 1), this_boxes_full.repeat(n_items, 1)], dim=-1)
                this_boxes_output = torch.cat([this_boxes, this_boxes_full.repeat(n_boxes, 1), this_items_full.repeat(n_boxes, 1)], dim=-1)
            else:
                this_items_output = this_items
                this_boxes_output = this_boxes
            
            if self.is_item_use_pre_acts:
                n_acts = n_pre_actions[ind]
                if n_acts <= 0:
                    actions_state = torch.zeros((len(this_items), this_items.shape[1]*3), dtype=this_items.dtype).to(device=this_items.device)
                else:
                    pacts = pre_actions[p_idx:p_idx+n_acts]
                    actions_state = torch.cat([
                        this_items[pacts[:, 0]].clone().detach().mean(dim=0),
                        this_boxes[pacts[:, 1]].clone().detach().mean(dim=0),
                        this_boxes[pacts[:, 2]].clone().detach().mean(dim=0),
                    ], dim=0).reshape(1, -1).repeat(len(this_items), 1)
                this_items_output = torch.cat([this_items_output, actions_state], dim=-1)
                ind += 1
                p_idx += n_acts
            
            if self.is_box_outfeats_use_item:
                this_boxes_output = torch.cat([this_boxes_output, this_items[item_idx, :].clone().detach().reshape(1, -1).repeat(n_boxes, 1)], dim=-1)
            
            items_output_feats.append(this_items_output)
            boxes_output_feats.append(this_boxes_output)
            i_index_arrow += n_items
            b_index_arrow += n_boxes
        
        items_output_feats = torch.cat(items_output_feats, dim=0)
        boxes_output_feats = torch.cat(boxes_output_feats, dim=0)

        # A final MLP on the variable features
        items_out = self.item_output(items_output_feats)
        boxes_out = self.box_output(boxes_output_feats)
        
        return items_out, boxes_out, items_output_feats, boxes_output_feats
    
    def preforward(self, item_features, edge_indices, edge_features, box_features, 
                   n_items_list=None, n_boxes_list=None, n_pre_actions=None, pre_actions=None):
        #reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension
        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        box_features = self.box_embedding(box_features)

        # Two half convolutions
        items_feats, boxes_feats = [item_features], [box_features]
        for gnn_layer in self.gnn_layers:
            this_item_feats, this_box_feats = gnn_layer(items_feats[-1], edge_indices, edge_features, boxes_feats[-1])
            # this_box_feats = gnn_layer[0](
            #     items_feats[-1], edge_indices, edge_features, boxes_feats[-1]
            # )
            boxes_feats.append(this_box_feats)
            
            # this_item_feats = gnn_layer[1](
            #     boxes_feats[-1], reversed_edge_indices, edge_features, items_feats[-1]
            # )
            items_feats.append(this_item_feats)
        
        if self.is_gnn_resnet:
            item_features = torch.cat(items_feats, dim=-1)
            box_features = torch.cat(boxes_feats, dim=-1)
        else:
            item_features = items_feats[-1]
            box_features = boxes_feats[-1]
        
        if (self.is_use_full_infos or self.is_item_use_pre_acts) and (n_items_list is not None) and (n_boxes_list is not None):
            i_index_arrow, b_index_arrow = 0, 0
            ind, p_idx = 0, 0
            items_output_feats, boxes_output_feats = [], []
            for n_items, n_boxes in zip(n_items_list, n_boxes_list):
                this_items = item_features[i_index_arrow:i_index_arrow+n_items, :]
                this_boxes = box_features[b_index_arrow:b_index_arrow+n_boxes, :]
                
                if self.is_use_full_infos:
                    this_items_full = this_items.mean(dim=0).unsqueeze(0)
                    this_boxes_full = this_boxes.mean(dim=0).unsqueeze(0)
                
                    this_items_output = torch.cat([this_items, this_items_full.repeat(n_items, 1), this_boxes_full.repeat(n_items, 1)], dim=-1)
                    this_boxes_output = torch.cat([this_boxes, this_boxes_full.repeat(n_boxes, 1), this_items_full.repeat(n_boxes, 1)], dim=-1)
                else:
                    this_items_output, this_boxes_output = this_items, this_boxes
                
                if self.is_item_use_pre_acts:
                    n_acts = n_pre_actions[ind]
                    if n_acts <= 0:
                        actions_state = torch.zeros((len(this_items), this_items.shape[1]*3), dtype=this_items.dtype).to(device=this_items.device)
                    else:
                        pacts = pre_actions[p_idx:p_idx+n_acts]
                        actions_state = torch.cat([
                            this_items[pacts[:, 0]].clone().detach().mean(dim=0),
                            this_boxes[pacts[:, 1]].clone().detach().mean(dim=0),
                            this_boxes[pacts[:, 2]].clone().detach().mean(dim=0),
                        ], dim=0).reshape(1, -1).repeat(len(this_items), 1)
                    this_items_output = torch.cat([this_items_output, actions_state], dim=-1)
                    ind += 1
                    p_idx += n_acts
                
                items_output_feats.append(this_items_output)
                boxes_output_feats.append(this_boxes_output)
                i_index_arrow += n_items
                b_index_arrow += n_boxes
            items_output_feats = torch.cat(items_output_feats, dim=0)
            boxes_output_feats = torch.cat(boxes_output_feats, dim=0)
        else:
            items_output_feats, boxes_output_feats = item_features, box_features
        
        return items_output_feats, boxes_output_feats, item_features

    def forward_items(self, items_output_feats):
        items_out = self.item_output(items_output_feats)
        return items_out
    
    def forward_boxes(self, boxes_output_feats, items_feats, n_items_list, n_boxes_list, selected_item_idxes):
        if self.is_box_outfeats_use_item:
            i_index_arrow, b_index_arrow = 0, 0
            boxes_feats = []
            for n_items, n_boxes, item_idx in zip(n_items_list, n_boxes_list, selected_item_idxes):
                this_items = items_feats[i_index_arrow:i_index_arrow+n_items, :]
                this_boxes = boxes_output_feats[b_index_arrow:b_index_arrow+n_boxes, :]
                this_boxes_output = torch.cat([
                    this_boxes, 
                    this_items[item_idx, :].clone().detach().reshape(1, -1).repeat(n_boxes, 1)
                ], dim=-1)
                boxes_feats.append(this_boxes_output)
                i_index_arrow += n_items
                b_index_arrow += n_boxes
            boxes_output_feats = torch.cat(boxes_feats, dim=0)
        
        boxes_out = self.box_output(boxes_output_feats)
        return boxes_out
    
    def parallel_to_device(self, device_ids):
        self.item_embedding = torch.nn.DataParallel(self.item_embedding.to(device_ids[0]), device_ids)
        self.edge_embedding = torch.nn.DataParallel(self.edge_embedding.to(device_ids[0]), device_ids)
        self.box_embedding = torch.nn.DataParallel(self.box_embedding.to(device_ids[0]), device_ids)
        
        self.item_output = torch.nn.DataParallel(self.item_output.to(device_ids[0]), device_ids)
        self.box_output = torch.nn.DataParallel(self.box_output.to(device_ids[0]), device_ids)
        
        for gnn_layer in self.gnn_layers:
            gnn_layer.parallel_to_device(device_ids)
            # gnn_layer[0].parallel_to_device(device_ids)
            # gnn_layer[1].parallel_to_device(device_ids)
        
        self.device_ids = device_ids
    
    def to_device(self, device):
        self.item_embedding = self.item_embedding.to(device=device)
        self.edge_embedding = self.edge_embedding.to(device=device)
        self.box_embedding = self.box_embedding.to(device=device)
        
        self.item_output = self.item_output.to(device=device)
        self.box_output = self.box_output.to(device=device)
        
        for gnn_layer in self.gnn_layers:
            gnn_layer = gnn_layer.to(device=device)
            # gnn_layer[0] = gnn_layer[0].to(device=device)
            # gnn_layer[1] = gnn_layer[1].to(device=device)


class GraphLayer(torch.nn.Module):
    def __init__(self, emb_size, graph_type):
        super().__init__()
        if graph_type == 'gat':
            Graph = GATConvolution
        elif graph_type == 'gin':
            Graph = GINConvolution
        else:
            Graph = BipartiteGraphConvolution
        
        self.gnn_item_to_box = Graph(emb_size=emb_size)
        self.gnn_box_to_item = Graph(emb_size=emb_size)
        
    def forward(self, items_feats, edge_indices, edge_features, boxes_feats):
        proc_box_feats = self.gnn_item_to_box(
            items_feats, edge_indices, edge_features, boxes_feats,
        )
        
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        proc_item_feats = self.gnn_box_to_item(
            proc_box_feats, reversed_edge_indices, edge_features, items_feats,
        )
        return proc_item_feats, proc_box_feats
    
    def parallel_to_device(self, device_ids):
        self.gnn_item_to_box.parallel_to_device(device_ids)
        self.gnn_box_to_item.parallel_to_device(device_ids)
      

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, emb_size):
        super().__init__("add")
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
    
    def parallel_to_device(self, device_ids):
        self.feature_module_left = torch.nn.DataParallel(self.feature_module_left.to(device_ids[0]), device_ids)
        self.feature_module_right = torch.nn.DataParallel(self.feature_module_right.to(device_ids[0]), device_ids)
        self.feature_module_edge = torch.nn.DataParallel(self.feature_module_edge.to(device_ids[0]), device_ids)
        self.feature_module_final = torch.nn.DataParallel(self.feature_module_final.to(device_ids[0]), device_ids)
        self.post_conv_module = torch.nn.DataParallel(self.post_conv_module.to(device_ids[0]), device_ids)
        self.output_module = torch.nn.DataParallel(self.output_module.to(device_ids[0]), device_ids)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


class GATConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 **kwargs):
        super().__init__('add')

        self.heads = 4
        self.in_channels = emb_size
        self.out_channels = emb_size // self.heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)
        self.lin_r = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)

        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, self.out_channels * 3))

        # output_layer
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.lin_l.weight)
        torch.nn.init.orthogonal_(self.lin_r.weight)
        torch.nn.init.orthogonal_(self.att)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(left_features)
        x_r = self.lin_r(right_features)
        # print(x_l.shape, x_r.shape, left_features.shape, right_features.shape, edge_features.shape)

        out = self.propagate(edge_indices, x=(x_l, x_r), size=(left_features.shape[0], right_features.shape[0]), edge_features=edge_features)
        return self.output_module(torch.cat([out, right_features], dim=-1))

    def message(self, x_j, x_i,
                index,
                edge_features):
        # print(x_i.shape, x_j.shape, edge_features.shape)
        x = torch.cat([x_i, x_j, edge_features], dim=-1)
        x = torch.nn.LeakyReLU(self.negative_slope)(x)
        x = x.view(-1, self.heads, self.out_channels * 3)
        # print(x.shape, self.att.shape)
        alpha_tmp = (x * self.att).sum(dim=-1)
        # print(alpha_tmp.shape)
        alpha = torch_geometric.utils.softmax(alpha_tmp, index)
        x = x_j.view(-1, self.heads, self.out_channels) * alpha.unsqueeze(-1)
        return x.view(-1, self.heads * self.out_channels)
    
    def parallel_to_device(self, device_ids):
        self.lin_l = torch.nn.DataParallel(self.lin_l.to(device_ids[0]), device_ids)
        self.lin_r = torch.nn.DataParallel(self.lin_r.to(device_ids[0]), device_ids)
        # self.att = torch.nn.DataParallel(self.att.to(device_ids[0]), device_ids)
        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, self.out_channels * 3).to(device_ids[0]))
        self.output_module = torch.nn.DataParallel(self.output_module.to(device_ids[0]), device_ids)


class GINConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size, eps: float = 0.5, train_eps: bool = True):
        super().__init__('add')

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size * 2, emb_size * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size * 4, emb_size * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size * 2, emb_size),
            torch.nn.LeakyReLU(),
        )

        #self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.train_eps = train_eps

        self.reset_parameters()

    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                torch.nn.init.orthogonal_(t)
            else:
                torch.nn.init.normal_(t)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)

        output += (1 + self.eps) * right_features
        return self.output_module(output)

    def message(self, node_features_j, edge_features):
        output = torch.nn.functional.relu(node_features_j + edge_features)
        return output
    
    def parallel_to_device(self, device_ids):
        self.output_module = torch.nn.DataParallel(self.output_module.to(device_ids[0]), device_ids)
        if self.train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([self.initial_eps]).to(device_ids[0]))
        else:
            self.eps = self.eps.to(device_ids[0])
