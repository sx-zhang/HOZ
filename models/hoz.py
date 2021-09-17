from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput

import scipy.sparse as sp
import numpy as np
import scipy.io as scio
import os
import json
import copy

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_scene_graph(path):
    graph = {}
    scenes = ['Kitchens', 'Living_Rooms', 'Bedrooms', 'Bathrooms']
    for s in scenes:
        data = scio.loadmat(os.path.join(path, s+'.mat'))
        graph[s] = data

    return graph

def dijkstra(graph, src):
    length = len(graph)
    type_ = type(graph)
    if type_ == list:
        nodes = [i for i in range(length)]
    elif type_ == dict:
        nodes = graph.keys()

    visited = [src]
    path = {src:{src:[]}}
    nodes.remove(src)
    distance_graph = {src:0}
    pre = next = src

    while nodes:
        distance = float('inf')
        for v in visited:
             for d in nodes:
                new_dist = graph[src][v] + graph[v][d]
                if new_dist <= distance:
                    distance = new_dist
                    next = d
                    pre = v
                    graph[src][d] = new_dist


        path[src][next] = [i for i in path[src][pre]]
        path[src][next].append(next)

        distance_graph[next] = distance

        visited.append(next)
        nodes.remove(next)

    return distance_graph, path


def normalize(x):
    x = -np.log(x)
    nozero_x = x[np.nonzero(x)]
    new_array = np.zeros(x.shape)
    x_max = np.max(nozero_x)
    x_min = np.min(nozero_x)
    x_ = x_max-x_min
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] == 0:
                continue
            else:
                new_array[i][j] = (x[i][j]-x_min)/x_
    return new_array.tolist()


class HOZ(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(HOZ, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.action_at_a = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), requires_grad=False)
        self.action_at_b = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), requires_grad=False)
        self.action_at_scale = nn.Parameter(torch.tensor(0.58), requires_grad=False)

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(518, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.graph_detection_other_info_linear_1 = nn.Linear(6, self.num_cate)
        self.graph_detection_other_info_linear_2 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_3 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_4 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_5 = nn.Linear(self.num_cate, self.num_cate)

        self.embed_action = nn.Linear(action_space, 10)

        pointwise_in_channels = 64 + self.num_cate + 10 + self.num_cate + 4

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.info_embedding = nn.Linear(5,49)
        self.scene_embedding = nn.Conv2d(86,64,1,1)
        self.scene_classifier = nn.Linear(64*7*7,4)

        self.graph_data = load_scene_graph('./scence_graph')

        self.zone_number, self.feature_length = self.graph_data['Kitchens']['node_features'].shape
        self.gcn_input = nn.Parameter(torch.zeros((self.zone_number, self.feature_length)), requires_grad=False)
        self.zones_feature = nn.Parameter(torch.zeros((self.zone_number, self.feature_length)), requires_grad=False)
        self.graph_buffer = [None for i in range(4)]
        self.scene_num = None
        self.fuse_scale = nn.Parameter(torch.tensor(0.005), requires_grad=True)
        self.state_index = 0
        self.target_index = 0
        self.sub_goal_index = 0
        # get and normalize adjacency matrix.
        self.adj_list = {}
        for k, v in self.graph_data.items():
            A_raw = v['edges']
            # rand e
            # A_raw = 0.2*np.ones((8,8))+0.002*np.random.rand(8,8)
            # Used to search shortest path
            self.adj_list[k] = normalize(A_raw)
            # Used to gcn compute
            A = normalize_adj(A_raw).tocsr().toarray()
            self.graph_data[k]['edges'] = A

        # last layer of resnet18.
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[-2:]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

        self.W0 = nn.Linear(22, 22, bias=False)
    def reset(self):
        self.graph_buffer = [None for i in range(4)]
    def gcn_embed(self, scene_name):
        x = self.gcn_input
        # print(x.max(),x.min())
        # rand n
        # x = 0.2*torch.rand(8,22).float().to(x.device)
        A = torch.from_numpy(self.graph_data[scene_name]['edges']).float().to(x.device)
        x = torch.mm(A, x)
        x = F.relu(self.W0(x))
        # x = torch.mm(A, x)
        # x = F.relu(self.W1(x))
        state_embedding = x[self.state_index]
        subgoal_embedding = x[self.sub_goal_index]
        target_embedding = x[self.target_index]
        # out = torch.cat((state_embedding.view(1, 512), target_embedding.view(1, 512)), dim=1)
        return state_embedding, subgoal_embedding, target_embedding

    def sub_goal(self, scene_vec, target_object, state):
        # scene_name, scene_vec = self.find_scene(scene)
        # scene_vec = scene_vec.to(state.device)
        scenes = ['Kitchens', 'Living_Rooms', 'Bedrooms', 'Bathrooms']
        scene_name = scenes[torch.argmax(scene_vec).item()]
        scene_graph = self.graph_data[scene_name]
        state_zone_feature = self.state_zone(scene_graph, state)
        target_zone_feature = self.target_zone(scene_graph, target_object)
        self.get_subgoal_index(scene_name)
        return state_zone_feature, target_zone_feature, scene_vec, scene_name

    def state_zone(self, scene_graph, feature):
        if self.graph_buffer[self.scene_num] is None:
            self.zones_feature.data = torch.from_numpy(scene_graph['node_features']).float().to(self.zones_feature.device)
        else:
            self.zones_feature.data = self.graph_buffer[self.scene_num]
        self.gcn_input.data = self.zones_feature.data
        state_feature = feature.view(1, 22).repeat(self.zone_number, 1)
        distance = F.pairwise_distance(state_feature, self.zones_feature, p=2)
        index = distance.argmin()
        self.gcn_input.data[index] = self.fuse_scale*feature.squeeze() + (1-self.fuse_scale)*self.zones_feature[index]
        self.graph_buffer[self.scene_num] = copy.deepcopy(self.gcn_input.data)
        self.state_index = index

        return self.zones_feature[index]

    def target_zone(self, scene_graph, target_object):
        index = torch.nonzero(target_object.squeeze())[0]
        # contain_objects = torch.from_numpy(scene_graph['node_features']).to(self.zones_feature.device)
        contain_objects = self.zones_feature
        max_index = contain_objects[:, index].argmax()
        self.target_index = max_index
        return self.zones_feature[max_index]

    def get_subgoal_index(self, scene_name):
        state_index = int(self.state_index)
        target_index = int(self.target_index)
        distance, path = dijkstra(self.adj_list[scene_name], state_index)
        trajectory = path[state_index][target_index]
        if len(trajectory) == 0:
            self.sub_goal_index = self.target_index
        else:
            self.sub_goal_index = trajectory[0]

    def embedding(self, state, target, action_embedding_input, scene, target_object):
        at_v = torch.mul(target['info'][:, -1].view(target['info'].shape[0], 1), target['indicator'])
        at = torch.mul(torch.max(at_v), self.action_at_scale)
        action_at = torch.mul(at, self.action_at_a) + self.action_at_b

        info_embedding = F.relu(self.info_embedding(target['info']))

        stat_onehot_vec = torch.sign(target['info'][:, -1])
        target_object = target['indicator']
        target_info = torch.cat((target['info'], target['indicator']), dim=1)
        target_info = F.relu(self.graph_detection_other_info_linear_1(target_info))
        target_info = target_info.t()
        target_info = F.relu(self.graph_detection_other_info_linear_2(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_3(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_4(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_5(target_info))
        target_appear = torch.mm(target['appear'].t(), target_info).t()
        target = torch.cat((target_appear, target['info'], target['indicator']), dim=1)

        target = F.relu(self.graph_detection_feature(target))
        target_embedding = target.reshape(1, self.num_cate, 7, 7)

        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

        image_embedding = F.relu(self.conv1(state))

        scene_embedding = F.relu(self.scene_embedding(torch.cat((info_embedding.view(1,22,7,7),image_embedding),dim=1))).view(1,-1)
        scene_vec = F.softmax(self.scene_classifier(scene_embedding),dim=1).squeeze()
        self.scene_num = torch.argmax(scene_vec)

        x = self.dropout(image_embedding)

        state_zone, next_zone, scene_vec, scene_name = self.sub_goal(scene_vec, target_object, stat_onehot_vec)
        state_zone_embedding, subgoal_zone_embedding, target_zone_embedding = self.gcn_embed(scene_name)

        x = torch.cat((x, target_embedding, action_reshaped, scene_vec.view(1, 4, 1, 1).repeat(1, 1, 7, 7),
                       subgoal_zone_embedding.view(1, 22, 1, 1).repeat(1, 1, 7, 7)), dim=1)


        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)


        return out, image_embedding,action_at

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        scene = model_input.scene
        target_object = model_input.target_object

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs

        x, image_embedding , action_at= self.embedding(state, target, action_probs, scene, target_object)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        actor_out = torch.mul(actor_out, action_at)
        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
