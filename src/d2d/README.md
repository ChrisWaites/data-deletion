# References

> [Descent-to-Delete: Gradient-Based Methods for Machine Unlearning](https://arxiv.org/abs/2007.02923)\
> Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi\
> _arXiv:2007.02923_

# Experiment 1

To run: `python classifier.py`

## Dataset

Idea: train a full model on public points, treat all but the last layer as a fixed feature extractor, treat the last layer as logistic regression, and then perform descent-to-delete on new private points.

[MNIST](https://en.wikipedia.org/wiki/MNIST_database)

![alt text](../../resources/mnist.png)

Public points:

```
X: (60000, 28, 28, 1)
y: (60000, 10)
```

Private points:

```
X: (10000, 28, 28, 1)
y: (10000, 10)
```

## Model

Feature extractor:

```
stax.serial(
  Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
  Relu,
  MaxPool((2, 2), (1, 1)),
  Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
  Relu,
  MaxPool((2, 2), (1, 1)),
  Flatten,
  Dense(32),
  Relu
)
```

Logistic regression:

```
stax.serial(
  Dense(10),
  LogSoftmax
)
```

## Training

### Step 1: Feature extractor training

```
iterations = 5000
batch_size = 64
step_size = 0.001
```

Results:

```
Accuracy on public points: 0.9902
Accuracy on private points: 0.9895
```

### Step 2: Finetuning on private points

```
iterations = 150
step_size = 0.5
projection_radius = 1.0
l2 = 0.05
```

Privacy parameters:

```
Epsilon: 1.0
Delta: 3.981-05
Sigma: 0.0152
```

Results:

```
Accuracy (not published): 0.9895
Accuracy (published): 0.9900
```

### Step 3: Handling deletion requests

```
iterations = 150 # iterations per deletion request
step_size = 0.5
num_requests = 25
```

Results:

```
Accuracy (not published): 0.8637
Accuracy (published): 0.8392
```
