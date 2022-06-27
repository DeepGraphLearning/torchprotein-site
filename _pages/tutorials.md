---
title: Tutorials
layout: page
permalink: /tutorials
---

TorchProtein is a [PyTorch]- and [TorchDrug]-based machine learning toolbox designed for several purposes of protein science.

- Universal representation of proteins through a unified data structure with GPU support
- Rapid prototyping of machine learning based protein research with a large collection of flexible building blocks
- Maintaining a comprehensive set of datasets, models and tasks for benchmarking existing and future methods

[PyTorch]: https://pytorch.org/
[TorchDrug]: https://torchdrug.ai/
[overview of TorchDrug]: https://torchdrug.ai/get_started

Here, we provide four tutorials to show new users a general picture of TorchProtein from four perspectives, 
the protein data structure, the solution to sequence-based protein property prediction, 
the solution to structure-based property prediction and how to learn protein representations from unlabeled protein structures. 

### [Tutorial 1 - Protein Data Structure]

In this tutorial, we will learn TorchProtein from following aspects:
- How TorchProtein represent the sequence and structure of a protein with a unified data structure;
- How to construct the protein data represented by such data structure;
- What operations can we perform to analyze the protein data;
- What atom-, residue- and protein-level attributes are incorporated in the data by default;
- How can we register customized attributes of proteins.

### [Tutorial 2 - Sequence-based Protein Property Prediction]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch a protein sequence dataset and specify the transformation functions we would to perform on each sample;
- How to construct a sequence-based model to extract protein sequence representations;
- Five types of protein sequence understanding tasks considered in TorchProtein;
- How can we solve a typical task of each task type by warping the protein sequence encoder into a task-specific module;
- How to instantiate an engine to conduct training and evaluation.

### [Tutorial 3 - Structure-based Protein Property Prediction]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch a protein structure dataset for function prediction and specify the transformation functions applied on each sample;
- How to better represent the geometric structures of proteins with various dynamic graph construction methods;
- How to construct a superior protein structure encoder;
- How can we solve the function prediction task by warping the structure encoder into a task-specific module;
- How to define an engine that accommodates training and evaluation.

### [Tutorial 4 - Pretrained Protein Structure Representations]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch an unlabeled protein structure dataset for pretraining and specify the transformation functions applied on each sample;
- Effectively representing the geometric structures of proteins through dynamic graph construction methods;
- The definition of a superior protein structure encoder;
- How to pretrain the protein structure encoder via two typical self-supervised learning approaches;
- How to finetune the pretrained encoder on a structure-based protein function prediction task.

[Tutorial 1 - Protein Data Structure]: tutorial_1

[Tutorial 2 - Sequence-based Protein Property Prediction]: tutorial_2

[Tutorial 3 - Structure-based Protein Property Prediction]: tutorial_3

[Tutorial 4 - Pretrained Protein Structure Representations]: tutorial_4

**Note.** For more details about the interfaces involved in these tutorials, please refer to the [document].

[document]: {{ "/docs/api/" | relative_url }}
[tutorial_1]: {{ "/tutorial_1/" | relative_url }}
[tutorial_2]: {{ "/tutorial_2/" | relative_url }}
[tutorial_3]: {{ "/tutorial_3/" | relative_url }}
[tutorial_4]: {{ "/tutorial_4/" | relative_url }}
