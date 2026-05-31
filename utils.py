import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_clustering import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from opt import args

def read_pickle_file(path):
    with open(path, "rb") as file_obj:
        loader = pkl._Unpickler(file_obj)
        loader.encoding = "latin1"
        return loader.load()


def parse_index_file(filename):
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f]


def load_data(dataset):
    file_keys = ["x", "y", "tx", "ty", "allx", "ally", "graph"]

    loaded_items = [
        read_pickle_file(f"data/ind.{dataset}.{key}")
        for key in file_keys
    ]

    x, y, tx, ty, allx, ally, graph = loaded_items

    test_indices = parse_index_file(f"data/ind.{dataset}.test.index")
    sorted_test_indices = np.sort(test_indices)

    if dataset == "citeseer":
        full_test_range = range(min(test_indices), max(test_indices) + 1)

        fixed_tx = sp.lil_matrix((len(full_test_range), x.shape[1]))
        fixed_tx[sorted_test_indices - min(test_indices), :] = tx
        tx = fixed_tx

        fixed_ty = np.zeros((len(full_test_range), y.shape[1]))
        fixed_ty[sorted_test_indices - min(test_indices), :] = ty
        ty = fixed_ty

    feature_matrix = sp.vstack([allx, tx]).tolil()
    feature_matrix[test_indices, :] = feature_matrix[sorted_test_indices, :]
    feature_tensor = torch.FloatTensor(np.asarray(feature_matrix.todense()))

    label_matrix = np.vstack([ally, ty])
    label_matrix[test_indices, :] = label_matrix[sorted_test_indices, :]

    adjacency = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    label_vector = np.argmax(label_matrix, axis=1)

    return adjacency, feature_tensor, label_vector


def normalize_adjacency(adj, norm_type="sym", add_self_loop=True):
    adj = sp.coo_matrix(adj)
    identity = sp.eye(adj.shape[0])

    graph_matrix = adj + identity if add_self_loop else adj
    degree = np.asarray(graph_matrix.sum(axis=1)).flatten()

    if norm_type == "sym":
        degree_inv_sqrt = sp.diags(np.power(degree, -0.5))
        normalized_adj = graph_matrix.dot(degree_inv_sqrt).T.dot(degree_inv_sqrt).tocoo()

    elif norm_type == "left":
        degree_inv = sp.diags(np.power(degree, -1.0))
        normalized_adj = degree_inv.dot(graph_matrix).tocoo()

    else:
        raise ValueError("norm_type must be either 'sym' or 'left'.")

    return identity, identity - normalized_adj


def apply_filter_series(features, operators):
    feature_array = sp.csr_matrix(features).toarray()
    accumulated = torch.zeros_like(torch.FloatTensor(feature_array))

    for operator in operators:
        feature_array = operator.dot(feature_array)
        accumulated += torch.FloatTensor(feature_array)

    return accumulated / len(operators)


def preprocess_graph(features, adj, layer_l, layer_h, norm="sym", renorm=True):
    identity, graph_laplacian = normalize_adjacency(
        adj,
        norm_type=norm,
        add_self_loop=renorm
    )

    low_pass_ops = [
        identity - graph_laplacian
        for _ in range(layer_l)
    ]

    high_pass_ops = [
        graph_laplacian
        for _ in range(layer_h)
    ]

    low_frequency_features = apply_filter_series(features, low_pass_ops)
    high_frequency_features = apply_filter_series(features, high_pass_ops)

    return [
        torch.FloatTensor(low_frequency_features),
        torch.FloatTensor(high_frequency_features)
    ]


def calculate_weighted_features_z(alpha, beta, sm_fea_s_list, sm_fea_s_noise):
    device = args.device

    alpha = alpha.to(device)
    beta = beta.to(device)

    low_frequency = sm_fea_s_list[0].to(device)
    high_frequency = sm_fea_s_list[1].to(device)
    noisy_features = sm_fea_s_noise.to(device)

    high_noise_mix = alpha * high_frequency + (1.0 - alpha) * noisy_features
    final_features = beta * low_frequency + (1.0 - beta) * high_noise_mix

    return final_features, alpha.item(), beta.item()


def add_gaussian_noise(X, sigma_X, A_hat, alpha=0.5, sigma=0.1, eps=1e-8):
    degree = A_hat.sum(axis=1, keepdims=True)
    transition_matrix = A_hat / (degree + eps)

    neighbor_mean = transition_matrix.dot(X)
    local_variation = transition_matrix.dot((X - neighbor_mean) ** 2).mean(axis=1)

    min_var = local_variation.min()
    max_var = local_variation.max()

    normalized_variation = (local_variation - min_var) / (max_var - min_var + eps)
    node_noise_scale = 1.0 + alpha * normalized_variation

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    gaussian_noise = torch.normal(
        mean=0,
        std=sigma_X,
        size=X.size()
    )

    adaptive_noise = gaussian_noise * node_noise_scale[:, None]

    return X + adaptive_noise


def laplacian(adj):
    degree_values = np.asarray(adj.sum(axis=1)).flatten()
    degree_matrix = sp.diags(degree_values)
    laplacian_matrix = degree_matrix - adj

    return torch.FloatTensor(laplacian_matrix.toarray())


def align_cluster_labels(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    true_classes = list(set(y_true))
    pred_classes = list(set(y_pred))

    if len(true_classes) != len(pred_classes):
        missing_index = 0
        for cls in true_classes:
            if cls not in pred_classes:
                y_pred[missing_index] = cls
                missing_index += 1

    pred_classes = list(set(y_pred))

    if len(true_classes) != len(pred_classes):
        print("error")
        return None

    assignment_matrix = np.zeros(
        (len(true_classes), len(pred_classes)),
        dtype=int
    )

    for i, true_cls in enumerate(true_classes):
        true_positions = np.where(y_true == true_cls)[0]

        for j, pred_cls in enumerate(pred_classes):
            matched_positions = true_positions[y_pred[true_positions] == pred_cls]
            assignment_matrix[i, j] = len(matched_positions)

    matcher = Munkres()
    optimal_pairs = matcher.compute((-assignment_matrix).tolist())

    aligned_predictions = np.zeros(len(y_pred))

    for i, true_cls in enumerate(true_classes):
        matched_pred_cls = pred_classes[optimal_pairs[i][1]]
        aligned_predictions[y_pred == matched_pred_cls] = true_cls

    return aligned_predictions


def cluster_acc(y_true, y_pred):
    matched_predictions = align_cluster_labels(y_true, y_pred)

    if matched_predictions is None:
        return None

    acc = metrics.accuracy_score(y_true - np.min(y_true), matched_predictions)
    f1 = metrics.f1_score(
        y_true - np.min(y_true),
        matched_predictions,
        average="macro"
    )

    return acc, f1


def eva(y_true, y_pred, show_details=True):
    acc, f1 = cluster_acc(y_true, y_pred)

    nmi = nmi_score(
        y_true,
        y_pred,
        average_method="arithmetic"
    )

    ari = ari_score(y_true, y_pred)

    if show_details:
        print(
            ":acc {:.4f}".format(acc),
            ", nmi {:.4f}".format(nmi),
            ", ari {:.4f}".format(ari),
            ", f1 {:.4f}".format(f1)
        )

    return acc, nmi, ari, f1


def load_graph_data(dataset_name, show_details=False):
    base_path = f"dataset/{dataset_name}/{dataset_name}"

    features = np.load(base_path + "_feat.npy", allow_pickle=True)
    labels = np.load(base_path + "_label.npy", allow_pickle=True)
    adjacency = np.load(base_path + "_adj.npy", allow_pickle=True)

    node_count = features.shape[0]

    if show_details:
        edge_count = int(np.nonzero(adjacency)[0].shape[0] / 2)
        class_count = max(labels) - min(labels) + 1

        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", features.shape)
        print("label shape:    ", labels.shape)
        print("adj shape:      ", adjacency.shape)
        print("undirected edge num:   ", edge_count)
        print("category num:          ", class_count)
        print("category distribution: ")

        for label_id in range(max(labels) + 1):
            print("label", label_id, end=":")
            print(len(labels[np.where(labels == label_id)]))

        print("++++++++++++++++++++++++++++++")

    return features, labels, adjacency, node_count


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, true_labels, cluster_num):
    predicted_labels, _ = kmeans(
        X=feature,
        num_clusters=cluster_num,
        distance="euclidean",
        device="cuda"
    )

    predicted_numpy = predicted_labels.numpy()

    acc, nmi, ari, f1 = eva(
        true_labels,
        predicted_numpy,
        show_details=False
    )

    return (
        round(acc * 100, 2),
        round(nmi * 100, 2),
        round(ari * 100, 2),
        round(f1 * 100, 2),
        predicted_numpy
    )