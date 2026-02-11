from __future__ import annotations
from typing import List, Tuple

import math


def k_means_clustering(
    points: list[Tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int,
) -> list[tuple[float, ...]]:

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if k != len(initial_centroids):
        raise ValueError(" k must match len(initial_centroids)")
    if not points:
        return [tuple(round(x, 4) for x in c) for c in initial_centroids]

    dim = len(points[0])
    if dim == 0:
        raise ValueError("Points must have at least one dimension.")

    for p in points:
        if len(p) != dim:
            raise ValueError("All points have to have the same dimensions")
    for c in initial_centroids:
        if len(c) != dim:
            raise ValueError("All centroids should have the same dimensions as the points.")

    centroids = [tuple(float(x) for x in c) for c in initial_centroids]

    def squared_euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b))

    for _ in range(max_iterations):
        clusters: list[list[tuple[float, ...]]] = [[] for _ in range(k)]
        for p in points:
            best_idx = 0
            best_dist = squared_euclidean(p, centroids[0])
            for i in range(1, k):
                d = squared_euclidean(p, centroids[i])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            clusters[best_idx].append(p)

        new_centroids: list[tuple[float, ...]] = []
        for i in range(k):
            if not clusters[i]:
                new_centroids.append(centroids[i])
                continue

            sums = [0.0] * dim
            for p in clusters[i]:
                for d in range(dim):
                    sums[d] += p[d]
            n = float(len(clusters[i]))
            new_centroids.append(tuple(s / n for s in sums))

        if new_centroids == centroids:
            break

        centroids = new_centroids

    final_centroids = [tuple(round(x, 4) for x in c) for c in centroids]
    return final_centroids


if __name__ == "__main__":
    points = [
        (1.0, 2.0),
        (1.5, 1.8),
        (5.0, 8.0),
        (8.0, 8.0),
        (1.0, 0.6),
    ]
    k = 3
    initial_centroids = [(1.0, 2.0), (5.0, 8.0), (8.0, 2.0)]
    max_iterations = 100

    result = k_means_clustering(points, k, initial_centroids, max_iterations)
    print("Final centroids:")
    for i, c in enumerate(result):
        print(f"  Cluster {i}: {c}")
