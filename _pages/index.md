---
layout: landing
permalink: /
title:
landing:
  title: An open source machine learning library for proteins
  excerpt: > 
    TorchProtein is an extension of TorchDrug to easily write and train deep learning models for a wide range of applications related to protein science. It covers various fundamental tasks (function prediction, structure prediction, binding affinity prediction) and models (sequence- and structure-based models) for protein representation learning.

image_grid:
  title: Why TorchProtein?
  excerpt: TorchProtein is an open source library for protein representation learning. Based on flexbile building blocks in TorchDrug, it provides data structures, tasks and models designed specifically for proteins that let researchers easily build ML powered applications on proteins.
  items:
    - title: Unified Data Structures
      image: assets/images/task/generation.png
      excerpt: >
        Provide user-friendly interface for loading and interpreting proteins and empower efficient idea exploration by a large collection of common datasets.
    - title: Flexible Building Blocks
      image: assets/images/task/generation.png
      excerpt: >
        Boost flexible model building by generic building blocks and dynamic graph construction designed specifically for proteins.
    - title: Extensive Benchmarks
      image: assets/images/task/generation.png
      excerpt: >
        Comprehensively compare different deep learning models for protein structure and sequence understanding.
    - title: A Protein ML Model Zoo
      image: assets/images/task/generation.png
      excerpt: >
        Maintain an abundant model zoo for extracting informative protein representations.
  url: features

card_grid:
  title: Learn TorchProtein with Examples
  items:
    - title: Protein Data Structure
      url: /tutorial_1
      image: assets/images/task/protein.png
      excerpt: >
        Start with basic data structures for manipulating proteins in this beginner tutorial.
    - title: Sequence-based Protein Property Prediction
      url: /tutorial_2
      image: assets/images/task/sequence_encoder.png
      excerpt: >
        Train a sequence-based encoder to predict properties of proteins, like fluoresence and stability.
    - title: Structure-based Protein Property Prediction
      url: /tutorial_3
      image: assets/images/task/structure_encoder.png
      excerpt: >
        Utilize the 3D structures of proteins via building structure-based encoders for property prediction.
    - title: Pre-train Protein Structure Encoders
      url: /tutorial_4
      image: assets/images/task/pretrain.png
      excerpt: >
        Pre-train a structure-based encoder with unlabeled proteins to obtain informative protein representations.

intro_box: "pip install torchdrug"
intro_image: "assets/images/intro.svg"
intro_image_absolute: true
intro_image_hide_on_mobile: true
---

{% include image-grid.html id="image_grid" %}

{% include card-grid.html id="card_grid" %}