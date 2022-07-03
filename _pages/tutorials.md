---
title: Tutorials
layout: page
permalink: /tutorials
---

Here, we provide four tutorials to show new users a general picture of TorchProtein from four perspectives, 
the protein data structure, the solution to sequence-based protein property prediction, 
the solution to structure-based property prediction and how to learn protein representations from unlabeled protein structures. 

### [Tutorial 1 - Protein Data Structure]

In this tutorial, we will learn TorchProtein from following aspects:
- How TorchProtein represents the sequence and structure of a protein with a unified data structure;
- How to read/write a protein with such data structure from/to a file;
- What operations we can perform to analyze the protein data;
- What atom-, residue- and protein-level attributes are incorporated in the data by default;
- How we can register customized attributes of proteins.

### [Tutorial 2 - Sequence-based Protein Property Prediction]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch a protein sequence dataset and specify the transformation functions we would to perform on each sample;
- How to construct a sequence-based model to extract protein sequence representations;
- Five types of protein sequence understanding tasks considered in TorchProtein;
- How can we solve each type of tasks by wraping a protein sequence encoder into a task-specific module;
- How to instantiate an engine to conduct training and evaluation.

### [Tutorial 3 - Structure-based Protein Property Prediction]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch a protein structure dataset for function prediction and specify the transformation functions applied on each sample;
- How to better represent the geometric structures of proteins with various dynamic graph construction methods;
- How to construct a superior protein structure encoder;
- How can we solve the function prediction task by wraping the structure encoder into a task-specific module;
- How to define an engine that accommodates training and evaluation.

### [Tutorial 4 - Pre-trained Protein Structure Representations]

In this tutorial, we will learn TorchProtein from following aspects:
- How to fetch an unlabeled protein structure dataset for pre-training and specify the transformation functions applied on each sample;
- Effectively representing the geometric structures of proteins through dynamic graph construction methods;
- The definition of a superior protein structure encoder;
- How to pre-train the protein structure encoder via two typical self-supervised learning approaches;
- How to fine-tune the pre-trained encoder on a structure-based protein function prediction task.

[Tutorial 1 - Protein Data Structure]: tutorial_1

[Tutorial 2 - Sequence-based Protein Property Prediction]: tutorial_2

[Tutorial 3 - Structure-based Protein Property Prediction]: tutorial_3

[Tutorial 4 - Pre-trained Protein Structure Representations]: tutorial_4

**Note.** For more details about the interfaces involved in these tutorials, please refer to the [document].

[document]: https://torchdrug.ai/docs/api
[tutorial_1]: {{ "/tutorial_1/" | relative_url }}
[tutorial_2]: {{ "/tutorial_2/" | relative_url }}
[tutorial_3]: {{ "/tutorial_3/" | relative_url }}
[tutorial_4]: {{ "/tutorial_4/" | relative_url }}
