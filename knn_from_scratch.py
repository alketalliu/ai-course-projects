"""knn_from_scratch.py — k-Nearest Neighbors implemented by hand (stdlib only).

Written as university AI/ML coursework to learn the algorithm from the inside.
Generates a synthetic 2D two-class dataset, splits it into train/test,
and reports accuracy for several values of k.

Usage:
    python3 knn_from_scratch.py
"""

import math
import random


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(train_x, train_y, point, k):
    ranked = sorted(
        ((euclidean(point, x), y) for x, y in zip(train_x, train_y)),
        key=lambda t: t[0],
    )
    votes = [y for _, y in ranked[:k]]
    return max(set(votes), key=votes.count)


def make_dataset(n=200, seed=7):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        # two gaussian blobs, one per class
        label = rng.choice([0, 1])
        cx, cy = (2.0, 2.0) if label == 0 else (-2.0, -2.0)
        xs.append((rng.gauss(cx, 1.0), rng.gauss(cy, 1.0)))
        ys.append(label)
    return xs, ys


def main():
    xs, ys = make_dataset()
    split = int(0.8 * len(xs))
    train_x, train_y = xs[:split], ys[:split]
    test_x, test_y = xs[split:], ys[split:]

    for k in (1, 3, 5):
        correct = sum(
            knn_predict(train_x, train_y, p, k) == y
            for p, y in zip(test_x, test_y)
        )
        print(f"k={k}: {correct}/{len(test_y)} correct "
              f"({100 * correct / len(test_y):.1f}%)")


if __name__ == "__main__":
    main()
