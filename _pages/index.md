---
layout: landing
permalink: /
title:
landing:
  title: Protein machine learning accessible to everyone
  excerpt: > 
    TorchProtein is a machine learning library for protein science, built on top of TorchDrug. It provides representation learning models for both protein sequences and structures, as well as fundamental protein tasks like function prediction, structure prediction. Taking the advantage of TorchDrug, it is also easy to reuse abundant models from small molecules for proteins, or solve protein-molecule tasks such as binding affinity prediction.
    <br><br>
    <b>Available as a part of TorchDrug.</b>

image_grid:
  title: Why TorchProtein?
  excerpt: TorchProtein is an open source library for protein representation learning. It encapsulates common protein machine learning demands in human-friendly data structures, models and tasks, to ease the process of building applications on proteins.
  url: features
  items:
    - title: Unified Data Structures
      image: assets/images/feature/data_structure.png
      excerpt: >
        Provide intuitive interface for mainuplating proteins and a large collection of common datasets for development.
    - title: Flexible Building Blocks
      image: assets/images/feature/building_block.png
      excerpt: >
        Boost model construction with generic building blocks and dynamic graph construction tailored to proteins.
    - title: Extensive Benchmarks
      image: assets/images/feature/benchmark.png
      excerpt: >
        Comprehensively compare different representation learning models for protein structure and sequence understanding.
    - title: A Protein ML Model Zoo
      image: assets/images/feature/model_zoo.png
      excerpt: >
        Power the performance with an abundant model zoo of protein representation learning models.

card_grid:
  title: Learn TorchProtein with Examples
  items:
    - title: Protein Data Structure
      url: /tutorial_1
      image: assets/images/tutorial/protein.png
      excerpt: >
        Start with basic data structures for manipulating proteins in this beginner tutorial.
    - title: Sequence-based Protein Property Prediction
      url: /tutorial_2
      image: assets/images/tutorial/sequence_encoder.png
      excerpt: >
        Train a sequence-based encoder to predict properties of proteins, like fluoresence and stability.
    - title: Structure-based Protein Property Prediction
      url: /tutorial_3
      image: assets/images/tutorial/structure_encoder.png
      excerpt: >
        Utilize the 3D structures of proteins via building structure-based encoders for property prediction.
    - title: Pre-train Protein Structure Encoders
      url: /tutorial_4
      image: assets/images/tutorial/pretrain.png
      excerpt: >
        Pre-train a structure-based encoder with unlabeled proteins to obtain informative protein representations.

intro_box: "pip install torchdrug"
intro_image: "assets/images/intro.svg"
intro_image_absolute: true
intro_image_hide_on_mobile: true
---

{% include image-grid.html id="image_grid" %}

{% include card-grid.html id="card_grid" %}