# Sinkhorn Distances on MNIST with k-NN

This project explores the use of Sinkhorn distances for MNIST image classification with a k-nearest neighbors classifier.

The project was inspired by Marco Cuturi's 2013 paper, [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895). However, this is not a full reproduction of the paper. Instead, the goal is to build a simpler experiment in order to understand the main idea behind Sinkhorn distances and apply them to a concrete machine learning task.

## Main idea

Optimal transport compares two probability distributions by measuring the minimal effort needed to move the mass of one distribution onto the other.

In this project, MNIST images are interpreted as probability distributions. Each image is flattened into a vector and normalized so that its pixel intensities sum to one. The geometry of the pixel grid is then used to define a cost matrix between pixels.

Using this cost matrix, we compute Sinkhorn distances between images and use these distances inside a k-nearest neighbors classifier.

## Notebook

The main explanation and experiment are available in the notebook:

[Read the notebook here](https://github.com/Samuel-Vangu/sinkhorn-mnist-knn/blob/main/notebooks/sinkhorn_mnist_knn.ipynb)

The notebook contains:

* a short theoretical introduction to Sinkhorn distances;
* the interpretation of MNIST images as probability distributions;
* the construction of the cost matrix;
* a k-NN classification experiment using Sinkhorn distance;
* a comparison with other distances such as Hellinger, chi-square, Euclidean, and Manhattan distances.

## Results

Because computing Sinkhorn distances is expensive, the experiment is performed on a reduced subset of MNIST.

The Sinkhorn distance obtained the best result among the tested distances in this simplified setting.

![Results](https://github.com/Samuel-Vangu/sinkhorn-mnist-knn/blob/main/notebooks/sinkhorn_mnist_knn.ipynb)

These results suggest that Sinkhorn distance can be an effective way to compare images seen as probability distributions, although it is more computationally expensive than classical distances.

## Code structure

The `src` directory contains helper Python scripts that can be reused to reproduce or extend the experiment.

```text
src/
└── sinkhorn_mnist_knn/
    ├── distances.py
    └── evaluation.py
```

### `distances.py`

This file contains implementations of several distances used in the experiment, including:

* Sinkhorn distance;
* Hellinger distance;
* chi-square distance.

### `evaluation.py`

This file contains a helper function used to train and evaluate k-nearest neighbors classifiers on MNIST. It performs cross-validation to select the best value of `k`, then evaluates the selected model on the test set.

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd sinkhorn-mnist-knn
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the project in editable mode:

```bash
pip install -e .
```

## Usage

After installation, the helper functions can be imported in the notebook with:

```python
from sinkhorn_mnist_knn.distances import sinkhorn_distance, hellinger, chi2_distance
from sinkhorn_mnist_knn.evaluation import train_test
```

## Limitations

This project is meant as a learning-oriented experiment. It does not aim to reproduce the full experimental setup of Cuturi's paper.

In particular:

* the experiment is run on a reduced subset of MNIST;
* the classifier used here is k-nearest neighbors, not the full method used in the original paper;
* the focus is on understanding and experimenting with Sinkhorn distances rather than achieving state-of-the-art performance.

## Possible improvements

Possible extensions of this project include:

* testing the method on a larger subset of MNIST;
* improving the efficiency of the Sinkhorn distance computation;
* using optimized optimal transport libraries such as POT;
* comparing with more advanced classifiers;
* reproducing more closely the experimental setup of Cuturi's paper.

## Reference

Marco Cuturi.
*Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances*.
Advances in Neural Information Processing Systems, 2013.
https://arxiv.org/abs/1306.0895
