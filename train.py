from utils import *
from tqdm import tqdm
from torch import optim
from model import DualProjectionModel
from model import ContrastiveObjective
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

DATASET_SETTINGS = {
    "cora": {
        "seed": 4, "cluster_num": 7, "gnnlayers_l": 3, "gnnlayers_h": 7,
        "sigma_X": 0.01, "sigma": 0.001, "gama": 0.6,
        "lr": 2e-3, "dims": [1300], "alpha1": 5
    },
    "citeseer": {
        "seed": 2, "cluster_num": 6, "gnnlayers_l": 5, "gnnlayers_h": 2,
        "sigma_X": 0.01, "sigma": 0.1, "gama": 0.01,
        "lr": 3e-4, "dims": [1000], "alpha1": 1
    },
    "amap": {
        "seed": 1, "cluster_num": 8, "gnnlayers_l": 7, "gnnlayers_h": 4,
        "sigma_X": 0.0001, "sigma": 0.1, "gama": 0.5,
        "lr": 2e-3, "dims": [800], "alpha1": 3
    },
    "bat": {
        "seed": 2, "cluster_num": 4, "gnnlayers_l": 60, "gnnlayers_h": 8,
        "sigma_X": 0.01, "sigma": 0.001, "gama": 0.5,
        "lr": 5e-3, "dims": [800], "alpha1": 0.5
    },
    "eat": {
        "seed": 2, "cluster_num": 4, "gnnlayers_l": 20, "gnnlayers_h": 5,
        "sigma_X": 0.001, "sigma": 0.01, "gama": 0.5,
        "lr": 1e-3, "dims": [1000], "alpha1": 1
    },
    "uat": {
        "seed": 5, "cluster_num": 4, "gnnlayers_l": 1, "gnnlayers_h": 3,
        "sigma_X": 0.01, "sigma": 0.1, "gama": 0.9,
        "lr": 1e-3, "dims": [600], "alpha1": 0.5
    },
    "others": {
        "seed": None, "cluster_num": 7, "gnnlayers_l": 3, "gnnlayers_h": 2,
        "sigma_X": 0.01, "sigma": 0.1, "gama": 0.5,
        "lr": 1e-3, "dims": [800], "alpha1": 1
    }
}


def apply_dataset_config(args, dataset_name):
    config = DATASET_SETTINGS[dataset_name]

    for key, value in config.items():
        if key not in ["seed", "alpha1"]:
            setattr(args, key, value)

    return config["seed"], config["alpha1"]


def save_to_file(filename, *values):
    with open(filename, "a+") as f:
        print(*values, file=f)


def prepare_graph_inputs(dataset_name, args):
    X, y, A, node_num = load_graph_data(dataset_name, show_details=False)

    identity_matrix = np.eye(A.shape[0], dtype=A.dtype)
    adjacency_with_self_loop = A + identity_matrix

    sparse_adj = sp.csr_matrix(A)
    sparse_adj = sparse_adj - sp.dia_matrix(
        (sparse_adj.diagonal()[np.newaxis, :], [0]),
        shape=sparse_adj.shape
    )
    sparse_adj.eliminate_zeros()

    smooth_features = preprocess_graph(
        X,
        sparse_adj,
        args.gnnlayers_l,
        args.gnnlayers_h,
        norm="sym",
        renorm=True
    )

    target_adj = (sparse_adj + sp.eye(sparse_adj.shape[0])).toarray()

    return X, y, adjacency_with_self_loop, smooth_features, target_adj


def evaluate_embedding(embedding, labels, cluster_count):
    acc, nmi, ari, f1, predicted = clustering(
        embedding,
        labels,
        cluster_count
    )
    return {
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "f1": f1,
        "labels": predicted
    }


for dataset_name in ["cora"]:
    args.dataset = dataset_name
    print(f"Using {dataset_name} dataset")
    save_to_file("result_baseline.csv", dataset_name)

    seed, alpha1 = apply_dataset_config(args, dataset_name)

    features, true_labels, A_hat, smoothed_feature_list, adj_target = prepare_graph_inputs(
        dataset_name,
        args
    )

    setup_seed(seed)

    model = DualProjectionModel([features.shape[1]] + args.dims)

    noisy_features = add_gaussian_noise(
        features,
        args.sigma_X,
        A_hat,
        alpha1,
        0.1,
        1e-8
    )

    weighted_features, _, _ = calculate_weighted_features_z(
        model.alpha,
        model.beta,
        smoothed_feature_list,
        noisy_features
    )

    best_result = evaluate_embedding(
        weighted_features,
        true_labels,
        args.cluster_num
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    contrastive_criterion = ContrastiveObjective()

    model = model.to(args.device)
    noisy_features = noisy_features.to(args.device)
    weighted_features = weighted_features.to(args.device)
    adj_target = torch.FloatTensor(adj_target).to(args.device)

    print("Start Training...")

    learned_parameters = []

    for epoch in tqdm(range(args.epochs)):
        model.train()

        weighted_features, alpha_value, beta_value = calculate_weighted_features_z(
            model.alpha,
            model.beta,
            smoothed_feature_list,
            noisy_features
        )

        learned_parameters.append([alpha_value, beta_value])

        embedding_1, embedding_2, smooth_embedding = model(
            weighted_features,
            is_train=True
        )

        fused_embedding = 0.5 * (embedding_1 + embedding_2)
        reconstructed_adj = 0.5 * (embedding_1 @ embedding_2.T)

        reconstruction_loss = F.mse_loss(reconstructed_adj, adj_target)
        contrastive_loss = contrastive_criterion(
            embedding_1,
            embedding_2,
            smooth_embedding
        )

        total_loss = args.gama * reconstruction_loss + contrastive_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 4 == 0:
            model.eval()

            with torch.no_grad():
                eval_z1, eval_z2, _ = model(weighted_features, is_train=True)
                current_embedding = 0.5 * (eval_z1 + eval_z2)

            current_result = evaluate_embedding(
                current_embedding,
                true_labels,
                args.cluster_num
            )

            if current_result["acc"] >= best_result["acc"]:
                best_result = current_result
                final_hidden_embedding = current_embedding
                final_predicted_labels = current_result["labels"]

    tqdm.write(
        "acc: {}, nmi: {}, ari: {}, f1: {}".format(
            best_result["acc"],
            best_result["nmi"],
            best_result["ari"],
            best_result["f1"]
        )
    )

    metric_history = {
        "acc": np.array([best_result["acc"]]),
        "nmi": np.array([best_result["nmi"]]),
        "ari": np.array([best_result["ari"]]),
        "f1": np.array([best_result["f1"]])
    }

    save_to_file(
        "result_baseline.csv",
        best_result["acc"],
        best_result["nmi"],
        best_result["ari"],
        best_result["f1"]
    )

    best_index = np.argmax(metric_history["acc"])

    print(
        "\nbest acc:", metric_history["acc"][best_index],
        ", best nmi:", metric_history["nmi"][best_index],
        ", best ari:", metric_history["ari"][best_index],
        ", best f1:", metric_history["f1"][best_index]
    )

    save_to_file(
        "result_baseline.csv",
        args.gnnlayers_l,
        args.gnnlayers_h,
        args.lr,
        args.dims,
        args.sigma_X,
        args.sigma,
        args.gama
    )

    for metric_name in ["acc", "nmi", "ari", "f1"]:
        values = metric_history[metric_name]
        save_to_file(
            "result_baseline.csv",
            round(values.mean(), 2),
            round(values.std(), 2)
        )