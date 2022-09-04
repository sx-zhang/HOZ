import h5py
import numpy as np
import my_cluster as mc
import networkx as nx
import scipy.io as scio
from km_match import KMMatcher


def get_location(index, location_list):
    return_list = []
    for i in index:
        return_list.append(location_list[i])

    return return_list

def get_distance(location_A, location_B):
    count = 0
    for A_point in location_A:
        for B_point in location_B:
            d = abs(float(A_point.split('|')[0])-float(B_point.split('|')[0])) \
                + abs(float(A_point.split('|')[1])-float(B_point.split('|')[1]))
            if d <= 0.5:
                count += 1
    return float(count/(len(location_A)*len(location_B)))

def add_node(G, center_feature, location_list, index):
    num = center_feature.shape[0]
    for i in range(num):
        coordinate_list = get_location(index[i], location_list)
        G.add_node(i, feature=center_feature[i], coordinate=coordinate_list)
    return G


def add_edge(G, center_feature):
    num = center_feature.shape[0]
    for i in range(num):
        for j in range(i+1, num):
            distance = get_distance(G._node[i]['coordinate'], G._node[j]['coordinate'])
            G.add_edge(i, j, weight=distance)

    return G


def cluster_feature(feature_list, zone_number=16):

    record,centers=mc.k_means(feature_list,zone_number,300)

    return record, centers

def get_det(id):
    my_list = []
    location_list = []
    # f = h5py.File("./data/FloorPlan"+str(id)+"/det_feature_22_cates.hdf5", "r")
    f = h5py.File("YOUR_DATA_PATH"+str(id)+"/det_feature_22_cates.hdf5", "r")
    my_key = list(f.keys())
    for item in my_key:
        feature = f[item][:]
        feature_1d = np.mean(feature, axis=1)
        feature_1d=np.squeeze(feature_1d)
        feature_1d[feature_1d>0]=1
        my_list.append(np.squeeze(feature_1d))
        location_list.append(item)
    f.close()
    return my_list, location_list
def get_room_graph(id, zone_number):
    feature_list, location_list = get_det(id)
    cluster_record, center_feature = cluster_feature(feature_list, zone_number=zone_number)

    g = nx.Graph()
    g = add_node(g, center_feature, location_list, cluster_record)
    g = add_edge(g, center_feature)

    return g
def get_weights(vec1, vec2):
    weights = np.zeros((vec1.shape[0], vec2.shape[0]))
    for i in range(vec1.shape[0]):
        for j in range(vec2.shape[0]):
            d = 1/(np.sum(vec1[i]*vec2[j])+0.1)+np.linalg.norm(vec1[i] - vec2[j])
            weights[i][j] = 1.0/d

    return weights

def get_edge_weight(node_list1, node_list2, data):
    edge_weight = []
    for i in range(len(data)):
        edge_weight.append(data[i]['edges'][node_list1[i]][node_list2[i]])
    new_edge_weight = np.mean(np.array(edge_weight))
    return new_edge_weight

def get_scene_graph(data, link):
    new_node_features = []
    for node_link in link:
        node = []
        for index, id in enumerate(node_link):
            node.append(data[index]['node_features'][id])
        new_node_features.append(np.mean(np.array(node), axis=0))
    new_edges = np.ones(data[0]['edges'].shape)
    for i in range(data[0]['edges'].shape[0]):
        for j in range(data[0]['edges'].shape[1]):
            if i == j:
                continue
            else:
                edge_weight = get_edge_weight(link[i], link[j], data)
                new_edges[i][j] = edge_weight

    save_dict = dict(node_features=np.array(new_node_features),
                     edges=np.array(new_edges))
    return save_dict

def change(G):
    features = []

    for k, v in G._node.items():
        features.append(v['feature'])
    edges = np.ones((len(features), len(features)))
    for k_i, v in G._adj.items():
        for k_j, distance in v.items():
            edges[k_i][k_j] = distance['weight']

    save_dict = dict(node_features=np.array(features),
                     edges=np.array(edges))
    return save_dict

if __name__ == '__main__':
    all_objects=[
        'AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
        'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
        'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster',
    ]
    room_id = dict(Kitchens=[k for k in range(1, 31)],
                   Living_Rooms=[k for k in range(201, 231)],
                   Bedrooms=[k for k in range(301, 331)],
                   Bathrooms=[k for k in range(401, 431)])
    for k, v in room_id.items():
        data=[]
        for id in v[0:20]:
            G = get_room_graph(id, zone_number=8)
            D = change(G)
            data.append(D)
        for i in range(0, len(data)-1):
            weights = get_weights(data[i]['node_features'], data[i+1]['node_features'])
            matcher = KMMatcher(weights)
            link = matcher.solve(verbose=True)
            if i == 0:
                link_all = link
            else:
                for point in link:
                    for index, existing_point in enumerate(link_all):
                        if existing_point[-1] == point[0] and len(existing_point)<i+2:
                            link_all[index].append(point[1])
        scene_graph = get_scene_graph(data, link_all)
        scio.savemat('./outputs/{}.mat'.format(k), scene_graph)