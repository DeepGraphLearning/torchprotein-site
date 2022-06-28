---
title: Tutorial 2 - Sequence-based Protein Property Prediction
layout: page
permalink: /tutorial_2
---

In this tutorial, we will learn how to use TorchProtein to solve different 
sequence-based protein property prediction tasks. 
This tutorial will include five different types of tasks including protein-wise property prediction, 
residue-wise property prediction, contact prediction, protein-protein interaction (PPI) prediction 
and protein-ligand interaction (PLI) prediction.

- TOC
{:toc} 

# Protein Sequence Representation Model

TorchProtein defines diverse sequence-based models to learn protein sequence representations. 
In this tutorial, we use a two-layer 1D CNN as the protein sequence representation model for all considered tasks. 
First, let's define such a model via the `models.ProteinCNN` module.

```python
from torchdrug import models

model = models.ProteinCNN(input_dim=21,
                          hidden_dims=[1024, 1024],
                          kernel_size=5, padding=2, readout="max")
```

# Task 1: Protein-wise Property Prediction

The first type of tasks we would like to solve is to predict protein-wise properties. 
We take the **Beta-lactamase activity prediction** task as an example, which aims to predict mutation effects on the TEM-1 beta-lactamase protein. 

Before defining the dataset, we need to first define the transformations we want to perform on proteins. 
We consider two transformations. (1) To lower the memory cost of sequence-based models, it is a common practice to truncate overlong protein sequences. 
In TorchProtein, we can define a protein truncation transformation by specifying the maximum length of proteins via the `max_length` parameter and whether to truncate from random residue or the first residue via the `random` parameter. 
(2) Besides, since we want to use residue features as node features for sequence-based models, we also need to define a view change transformation for proteins.
During dataset construction, we can pass the composition of two transformations as an argument.

```python
from torchdrug import transforms

truncate_transform = transforms.TruncateProtein(max_length=200, random=False, residue=True)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])
```

We then build the dataset via `datasets.BetaLactamase`, in which the dataset file will be automatically downloaded. 
In this dataset, the label of each sample is a real number indicating the fitness value of the protein.
By turning on `residue_only` option, TorchProtein will use `data.Protein.from_sequence_fast` to load proteins and thus be much faster. 
We can get the pre-specified training, validation and test splits by the `split()` method.

```python
from torchdrug import datasets

dataset = datasets.BetaLactamase("~/protein-datasets/", residue_only=True, transform=transform)
train_set, valid_set, test_set = dataset.split()
print("The label of first sample: ", dataset[0][dataset.target_fields[0]])
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))
```

```bash
The label of first sample:  0.9426838159561157
train samples: 4158, valid samples: 520, test samples: 520
```

To perform Beta-lactamase activity prediction, we wrap the CNN encoder into the `tasks.PropertyPrediction` module which appends a task-specific MLP prediction head upon CNN.  

```python
from torchdrug import tasks

task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="mse", metric=("mae", "rmse", "spearmanr"),
                                normalization=False, num_mlp_layer=2)
```

Now we can train our model. We set up an optimizer for our model, and put everything together into an Engine instance. It takes about 2 minutes to train our model for 10 epochs on this task. 
We finally evaluate its performance on the validation set.

```python
import torch
from torchdrug import core

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, 
                     gpus=[0], batch_size=64)
solver.train(num_epoch=10)
solver.evaluate("valid")
```
 
The evaluation result may be similar to the following.

```bash
mean absolute error [scaled_effect1]: 0.249482
root mean squared error [scaled_effect1]: 0.304326
spearmanr [scaled_effect1]: 0.44572
```

# Task 2: Residue-wise Property Prediction

The second type of tasks we consider is to predict residue-wise properties. We take the **secondary structure prediction** task as an example. 

We first build the dataset via `datasets.SecondaryStructure`, in which we use the `cb513` test set. For residue-level tasks, we usually keep the whole sequence of proteins, so we only use the `ProteinView` transform here. 
The `target` field denotes the secondary structure (coil, strand or helix) of each residue, and the `mask` field denotes whether each secondary structure label is valid or not. 
Both fields are with the same length of protein sequence.

```python
dataset = datasets.SecondaryStructure("~/protein-datasets/", residue_only=True, transform=protein_view_transform)
train_set, valid_set, test_set = dataset.split(["train", "valid", "cb513"])
print("SS3 label: ", dataset[0]["graph"].target[:10])
print("Valid mask: ", dataset[0]["graph"].mask[:10])
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))
```

```bash
SS3 label:  tensor([2, 2, 2, 0, 0, 0, 0, 0, 2, 2])
Valid mask:  tensor([True, True, True, True, True, True, True, True, True, True])
train samples: 8678, valid samples: 2170, test samples: 513
```

To perform secondary structure prediction, we wrap the CNN encoder into the `tasks.NodePropertyPrediction` module which appends a task-specific MLP prediction head upon CNN. 

```python
task = tasks.NodePropertyPrediction(model, criterion="ce", 
                                    metric=("micro_acc", "macro_acc"),
                                    num_mlp_layer=2, num_class=[3])
```

We train the model for 5 epochs, taking about 5 minutes, and finally evaluate it on the validation set.

```python
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=128)
solver.train(num_epoch=5)
solver.evaluate("valid")
```

The evaluation result may be similar to the following.

```bash
macro_acc: 0.629119
micro_acc: 0.624727
```

# Task 3: Contact Prediction

The third task we would to solve is to predict whether any pair of residues contact or not in the folded structure, *i.e.*, performing **contact prediction**.

We first build the dataset via `datasets.ProteinNet`. 
The `residue_position` field denotes the 3D coordinates of each residue, 
and the `mask` field denotes whether each residue position is valid or not. 
Both fields are with the same length of protein sequence.

```python
dataset = datasets.ProteinNet("~/protein-datasets/", residue_only=True, transform=protein_view_transform)
train_set, valid_set, test_set = dataset.split()
print("Residue position: ", dataset[0]["graph"].residue_position[:3])
print("Valid mask: ", dataset[0]["graph"].mask[:3])
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))
```

```bash
Residue position:  tensor([[ 2.0940e+00,  2.0000e-03, -1.2420e+00],
        [ 5.1260e+00, -2.0210e+00, -2.3290e+00],
        [ 7.5230e+00,  6.1500e-01, -3.6610e+00]])
Valid mask:  tensor([True, True, True])
train samples: 25299, valid samples: 224, test samples: 34
```

To perform contact prediction, we wrap the CNN encoder into the `tasks.ContactPrediction` module which appends a task-specific MLP prediction head upon CNN. 
Two residues with sequence gap larger than `gap` are seen as interacted if their Euclidean distance is within `threshold`. 
Different from previous tasks, the maximum truncation length `max_length` is defined in the task now, since the truncation behavior is different on the test set in the contact prediction task. 
For the test set, to save memory, we will split the test sequences into several blocks according to `max_length`. 

```python
task = tasks.ContactPrediction(model, max_length=500, random_truncate=True, threshold=8.0, gap=6,
                               criterion="bce", metric=("accuracy", "prec@L5", "prec@5"), num_mlp_layer=2)
```

Since the raw training set contains a lot of samples, and only small batch size can be used in this task, we use a subset of the raw training set with 1000 samples for training.
We train the model for 1 epoch, taking about 4 minutes, and finally evaluate it on the validation set.

```python
from torch.utils import data as torch_data

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
sub_train_set = torch_data.random_split(train_set, [1000, len(train_set) - 1000])[0]
solver = core.Engine(task, sub_train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=1)
solver.train(num_epoch=1)
solver.evaluate("valid")
```

The evaluation result may be similar to the following.

```bash
accuracy: 0.973398
prec@5: 0.123214
prec@L5: 0.0894395
```

# Task 4: Protein-Protein Interaction (PPI) Prediction

The fourth task we consider is to predict the binding affinity of two interacting proteins, *i.e.*, performing **PPI affinity prediction**.

We first build the dataset via `datasets.PPIAffinity`, in which each sample is a pair of proteins, and it is associated with a continuous label indicating the binding affinity. 
Since we now need to perform transformation on both proteins, we need to specify `keys` in the transformation function.

```python
truncate_transform_ = transforms.TruncateProtein(max_length=200, residue=True, keys=("graph1", "graph2"))
protein_view_transform_ = transforms.ProteinView(view="residue", keys=("graph1", "graph2"))
transform_ = transforms.Compose([truncate_transform_, protein_view_transform_])
dataset = datasets.PPIAffinity("~/protein-datasets/", residue_only=True, transform=transform_)
train_set, valid_set, test_set = dataset.split()
print("The label of first sample: ", dataset[0][dataset.target_fields[0]])
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))
```

```bash
The label of first sample:  -12.2937
train samples: 2421, valid samples: 203, test samples: 326
```

To perform PPI affinity prediction, we wrap the CNN encoder into the `tasks.InteractionPrediction` module which appends a task-specific MLP prediction head upon CNN. 

```python
task = tasks.InteractionPrediction(model, task=dataset.tasks,
                                   criterion="mse", metric=("mae", "rmse", "spearmanr"),
                                   normalization=False, num_mlp_layer=2)
```

We train the model for 10 epochs, taking about 2 minutes, and finally evaluate it on the validation set.

```python
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=64)
solver.train(num_epoch=10)
solver.evaluate("valid")
```

The evaluation result may be similar to the following.

```bash
mean absolute error [interaction]: 2.0611
root mean squared error [interaction]: 2.6741
spearmanr [interaction]: 0.502957
```

# Task 5: Protein-Ligand Interaction (PLI) Prediction

The fourth task we consider is to predict the binding affinity of a protein and a small molecule (*i.e.*, ligand). We take the **PLI prediction on BindingDB** as an example. 

We first build the dataset via `datasets.BindingDB`, in which each sample is a pair of protein and ligand, 
and it is associated with a continuous label indicating the binding affinity. 
We use the `holdout_test` set for test.

```python
truncate_transform_ = transforms.TruncateProtein(max_length=200, residue=True, keys="graph1")
protein_view_transform_ = transforms.ProteinView(view="residue", keys="graph1")
transform_ = transforms.Compose([truncate_transform_, protein_view_transform_])
dataset = datasets.BindingDB("~/protein-datasets/", residue_only=True, transform=transform_)
train_set, valid_set, test_set = dataset.split(["train", "valid", "holdout_test"])
print("The label of first sample: ", dataset[0][dataset.target_fields[0]])
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))
```

```bash
The label of first sample:  5.823908740944319
train samples: 7900, valid samples: 878, test samples: 5230
```

To perform PLI prediction, we require an additional ligand graph encoder to extract the representations of ligands. 
We define a 4-layer [Graph Isomorphism Network (GIN)] as the ligand graph encoder.
After that, we wrap the CNN encoder and the GIN encoder into the `tasks.InteractionPrediction` module which appends a task-specific MLP prediction head upon CNN and GIN. 

[Graph Isomorphism Network (GIN)]: https://arxiv.org/pdf/1810.00826.pdf

```python
model2 = models.GIN(input_dim=66,
                    hidden_dims=[256, 256, 256, 256],
                    batch_norm=True, short_cut=True, concat_hidden=True)

task = tasks.InteractionPrediction(model, model2=model2, task=dataset.tasks,
                                   criterion="mse", metric=("mae", "rmse", "spearmanr"),
                                   normalization=False, num_mlp_layer=2)
```

We train the model for 5 epochs, taking about 3 minutes, and finally evaluate it on the validation set.

```python
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=16)
solver.train(num_epoch=5)
solver.evaluate("valid")
```

The evaluation result may be similar to the following.

```bash
mean absolute error [affinity]: 0.916763
root mean squared error [affinity]: 1.22093
spearmanr [affinity]: 0.60658
```
