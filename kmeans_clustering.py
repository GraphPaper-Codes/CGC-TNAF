import torch
import numpy as np


def choose_starting_centers(data, k):
    total_points = data.shape[0]
    chosen_ids = np.random.choice(total_points, size=k, replace=False)
    return data[chosen_ids]


def get_distance_function(metric):
    distance_methods = {
        "euclidean": squared_euclidean_distance,
        "cosine": cosine_distance
    }

    if metric not in distance_methods:
        raise NotImplementedError(f"Unsupported distance metric: {metric}")

    return distance_methods[metric]


def kmeans(
    X,
    num_clusters,
    distance="euclidean",
    tol=1e-4,
    device=torch.device("cuda")
):
    distance_fn = get_distance_function(distance)

    samples = X.float().to(device)

    best_centers = None
    lowest_total_distance = float("inf")

    for _ in range(20):
        candidate_centers = choose_starting_centers(samples, num_clusters)
        candidate_distance = distance_fn(samples, candidate_centers, device).sum()

        if candidate_distance < lowest_total_distance:
            lowest_total_distance = candidate_distance
            best_centers = candidate_centers.clone()

    centers = best_centers
    labels = None

    for _ in range(501):
        distance_matrix = distance_fn(samples, centers, device)
        labels = torch.argmin(distance_matrix, dim=1)

        previous_centers = centers.clone()

        updated_centers = []

        for cluster_id in range(num_clusters):
            member_mask = labels == cluster_id
            member_points = samples[member_mask]

            if member_points.shape[0] > 0:
                updated_centers.append(member_points.mean(dim=0))
            else:
                updated_centers.append(previous_centers[cluster_id])

        centers = torch.stack(updated_centers, dim=0)

        movement = torch.sqrt(((centers - previous_centers) ** 2).sum(dim=1)).sum()

        if movement ** 2 < tol:
            break

    return labels.cpu(), centers.cpu()


def kmeans_predict(
    X,
    cluster_centers,
    distance="euclidean",
    device=torch.device("cuda")
):
    distance_fn = get_distance_function(distance)

    samples = X.float().to(device)
    centers = cluster_centers.float().to(device)

    distances = distance_fn(samples, centers, device)
    assigned_clusters = torch.argmin(distances, dim=1)

    return assigned_clusters.cpu()


def squared_euclidean_distance(data_points, centers, device=torch.device("cuda")):
    data_points = data_points.to(device)
    centers = centers.to(device)

    expanded_points = data_points[:, None, :]
    expanded_centers = centers[None, :, :]

    return ((expanded_points - expanded_centers) ** 2).sum(dim=-1)


def cosine_distance(data_points, centers, device=torch.device("cuda")):
    data_points = data_points.to(device)
    centers = centers.to(device)

    expanded_points = data_points[:, None, :]
    expanded_centers = centers[None, :, :]

    normalized_points = expanded_points / expanded_points.norm(dim=-1, keepdim=True)
    normalized_centers = expanded_centers / expanded_centers.norm(dim=-1, keepdim=True)

    similarity = (normalized_points * normalized_centers).sum(dim=-1)

    return 1.0 - similarity