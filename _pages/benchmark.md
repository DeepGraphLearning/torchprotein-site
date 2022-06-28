---
title: Benchmark for Protein Sequence Understanding (PEER)
layout: page
permalink: /benchmark
---

Here, we summarize the benchmark results in the paper [PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding].
We maintain a leaderboard for each of the 14 considered protein understanding tasks. 
All benchmark results can be reproduced in the [PEER benchmark codebase]. 
We also maintain an **integrated leaderboard** among different methods by taking the mean reciprocal rank (MRR) as the metric.
In the future, we will open the entrance to receive new benchmark results of new methods from the community. 

**Note that**, all benchmark results reported here are averaged over three runs with seeds 0, 1 and 2, 
and the standard deviation of three runs is also reported.

[PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding]: https://arxiv.org/pdf/2206.02096.pdf
[PEER benchmark codebase]: https://github.com/DeepGraphLearning/PEER_Benchmark

- TOC
{:toc}

# Integrated Leaderboard

- **Evaluation metric** - Mean Reciprocal Rank (MRR) on all applicable benchmark tasks

| Rank |           Method            |  MRR  |         Ranks: Fluorescence &rarr; BindingDB          | Reference |             External data              |
|:----:|:---------------------------:|:-----:|:-----------------------------------------------------:|:---------:|:--------------------------------------:|
|  1   |   **[MTL] ESM-1b + Contact**    | **0.517** |      [4, 4, 1, 2, 2, 1, /, 1, 1, 5, 4, 2, 13, 5]      |[paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL |
|  2   |        ESM-1b (fix)         | 0.401 |    [17, 3, 12, 14, 1, 5, 2, 2, 2, 1, 1, 19, 4, 15]    | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          |
|  3   |     [MTL] CNN + Contact     | 0.277 |     [6, 11, 5, 1, 9, 9, /, 7, 8, 9, 12, 1, 3, 8]      | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |
|  4   |       [MTL] CNN + SSP       | 0.272 |    [1, 7, 6, 8, 13, 10, 13, 6, /, 11, 11, 6, 1, 3]    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |
|  5   |           ESM-1b            | 0.270 |     [9, 8, 4, 3, 4, 2, 1, 4, 3, 6, 6, 7, 15, 12]      | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          |
|  6   |     [MTL] ESM-1b + SSP      | 0.269 |      [5, 2, 3, 6, 5, 3, 5, 3, /, 4, 3, 4, 7, 4]       | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   |
|  7   |     [MTL] ESM-1b + Fold     | 0.250 |      [8, 5, 2, 15, 3, 4, 4, /, 4, 2, 5, 3, 8, 9]      | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   |
|  8   |          ProtBert           | 0.231 |     [7, 1, 9, 12, 6, 6, 3, 5, 5, 3, 7, 5, 16, 11]     | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            |
|  9   |      [MTL] CNN + Fold       | 0.226 |   [2, 17, 8, 10, 14, 12, 12, /, 10, 16, 8, 8, 2, 1]   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |
|  10  |             CNN             | 0.127 |   [3, 14, 7, 16, 10, 8, 11, 8, 9, 8, 15, 13, 5, 7]    | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |
|  11  |       ProtBert (fix)        | 0.121 |  [19, 6, 11, 18, 8, 11, 7, 9, 12, 14, 2, 17, 11, 17]  | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            |
|  12  |  [MTL] Transformer + Fold   | 0.116 | [11, 9, 14, 11, 11, 15, 14, /, 14, 13, 10, 10, 14, 2] | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |
|  13  |            LSTM             | 0.104 |  [16, 16, 19, 4, 7, 7, 6, 14, 7, 15, 13, 14, 12, 16]  | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    |
|  14  |   [MTL] Transformer + SSP   | 0.091 | [10, 10, 16, 9, 12, 17, 10, 15, /, 12, 18, 11, 6, 10] | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |
|  15  |         Transformer         | 0.090 | [12, 13, 15, 5, 15, 16, 9, 13, 13, 10, 17, 9, 10, 14] | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    |
|  16  |           ResNet            | 0.084 | [15, 19, 17, 13, 17, 13, 8, 12, 6, 19, 9, 18, 9, 13]  | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    |
|  17  | [MTL] Transformer + Contact | 0.082 | [13, 15, 18, 7, 16, 18, /, 11, 11, 18, 16, 12, 17, 6] | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |
|  18  |             DDE             | 0.082 |  [14, 12, 10, 17, 18, 14, /, 10, /, 7, 14, 15, /, /]  | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |
|  19  |            Moran            | 0.058 | [18, 18, 13, 19, 19, 19, /, 16, /, 17, 19, 16, /, /]  | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |


---

# Protein Function Prediction

## Leaderboard for Fluorescence Prediction

- **Task type** - Protein-wise Regression
- **Dataset statistics** - #Train: 21,446 &nbsp; #Validation: 5,362 &nbsp; #Test: 27,217
- **Evaluation metric** - Spearman's Rho on the test set (the higher, the better)
- **Dataset splitting scheme** - Train & Validation: mutants with three or less mutations; Test: mutants with four or more mutations.
- **Description** - Models are asked to predict the fitness of green fluorescent protein mutants. The prediction target is a real number indicating the logarithm of fluorescence intensity.

| Rank |           Method            |   Test Spearman's Rho    | Reference |             External data              |   #Params   |           Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:----------------------------:|
|  1   |     **[MTL] CNN + SSP**     | **0.683 &plusmn; 0.001** | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,455,748  | 4 &times; Tesla V100 (32GB)  |
|  2   |      [MTL] CNN + Fold       |   0.682 &plusmn; 0.001   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,677,548  | 4 &times; Tesla V100 (32GB)  |
|  3   |             CNN             |   0.682 &plusmn; 0.002   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,403,073  | 4 &times; Tesla V100 (32GB)  |
|  4   |   [MTL] ESM-1b + Contact    |   0.681 &plusmn; 0.001   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,279,416 | 4 &times; Tesla V100 (32GB)  |
|  5   |     [MTL] ESM-1b + SSP      |   0.681 &plusmn; 0.002   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,643,578 | 4 &times; Tesla V100 (32GB)  |
|  6   |     [MTL] CNN + Contact     |   0.680 &plusmn; 0.001   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,502,274  | 4 &times; Tesla V100 (32GB)  |
|  7   |          ProtBert           |   0.679 &plusmn; 0.001   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |
|  8   |     [MTL] ESM-1b + Fold     |   0.679 &plusmn; 0.001   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,170,530 | 4 &times; Tesla V100 (32GB)  |
|  9   |           ESM-1b            |   0.679 &plusmn; 0.002   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 4 &times; Tesla V100 (32GB)  |
|  10  |   [MTL] Transformer + SSP   |   0.656 &plusmn; 0.002   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,810,180  | 4 &times; Tesla V100 (32GB)  |
|  11  |  [MTL] Transformer + Fold   |   0.648 &plusmn; 0.004   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,421,676  | 4 &times; Tesla V100 (32GB)  |
|  12  |         Transformer         |   0.643 &plusmn; 0.005   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,545,985  | 4 &times; Tesla V100 (32GB)  |
|  13  | [MTL] Transformer + Contact |   0.642 &plusmn; 0.017   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,071,298  | 4 &times; Tesla V100 (32GB)  |
|  14  |             DDE             |   0.638 &plusmn; 0.003   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   468,481   | 4 &times; Tesla V100 (32GB)  |
|  15  |           ResNet            |   0.636 &plusmn; 0.021   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,300,354  | 4 &times; Tesla V100 (32GB)  |
|  16  |            LSTM             |   0.494 &plusmn; 0.071   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,080,328  | 4 &times; Tesla V100 (32GB)  |
|  17  |        ESM-1b (fix)         |   0.430 &plusmn; 0.002   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 4 &times; Tesla V100 (32GB)  |
|  18  |            Moran            |   0.400 &plusmn; 0.001   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   386,561   | 4 &times; Tesla V100 (32GB)  |
|  19  |       ProtBert (fix)        |   0.339 &plusmn; 0.003   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |

***

## Leaderboard for Stability Prediction

- **Task type** - Protein-wise Regression
- **Dataset statistics** - #Train: 53,571 &nbsp; #Validation: 2,512 &nbsp; #Test: 12,851
- **Evaluation metric** - Spearman's Rho on the test set (the higher, the better)
- **Dataset splitting scheme** - Train & Validation: proteins from four rounds of experimental design; Test: top candidates with single mutations.
- **Description** - Models are asked to predict the stability of proteins under natural environment. The prediction target is a real number indicating the experimental measurement of stability.

| Rank |           Method            |   Test Spearman's Rho    | Reference |             External data              |   #Params   |           Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:----------------------------:|
|  1   |        **ProtBert**         | **0.771 &plusmn; 0.020** | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |
|  2   |     [MTL] ESM-1b + SSP      |   0.759 &plusmn; 0.002   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,643,578 | 4 &times; Tesla V100 (32GB)  |
|  3   |        ESM-1b (fix)         |   0.750 &plusmn; 0.010   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 4 &times; Tesla V100 (32GB)  |
|  4   |   [MTL] ESM-1b + Contact    |   0.733 &plusmn; 0.007   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,279,416 | 4 &times; Tesla V100 (32GB)  |
|  5   |     [MTL] ESM-1b + Fold     |   0.728 &plusmn; 0.002   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,170,530 | 4 &times; Tesla V100 (32GB)  |
|  6   |       ProtBert (fix)        |   0.697 &plusmn; 0.013   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |
|  7   |       [MTL] CNN + SSP       |   0.695 &plusmn; 0.016   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,455,748  | 4 &times; Tesla V100 (32GB)  |
|  8   |           ESM-1b            |   0.694 &plusmn; 0.073   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 4 &times; Tesla V100 (32GB)  |
|  9   |  [MTL] Transformer + Fold   |   0.672 &plusmn; 0.010   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,421,676  | 4 &times; Tesla V100 (32GB)  |
|  10  |   [MTL] Transformer + SSP   |   0.667 &plusmn; 0.063   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,810,180  | 4 &times; Tesla V100 (32GB)  |
|  11  |     [MTL] CNN + Contact     |   0.661 &plusmn; 0.006   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,502,274  | 4 &times; Tesla V100 (32GB)  |
|  12  |             DDE             |   0.652 &plusmn; 0.033   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   468,481   | 4 &times; Tesla V100 (32GB)  |
|  13  |         Transformer         |   0.649 &plusmn; 0.056   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,545,985  | 4 &times; Tesla V100 (32GB)  |
|  14  |             CNN             |   0.637 &plusmn; 0.010   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,403,073  | 4 &times; Tesla V100 (32GB)  |
|  15  | [MTL] Transformer + Contact |   0.620 &plusmn; 0.004   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,071,298  | 4 &times; Tesla V100 (32GB)  |
|  16  |            LSTM             |   0.533 &plusmn; 0.101   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,080,328  | 4 &times; Tesla V100 (32GB)  |
|  17  |      [MTL] CNN + Fold       |   0.472 &plusmn; 0.170   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,677,548  | 4 &times; Tesla V100 (32GB)  |
|  18  |            Moran            |   0.322 &plusmn; 0.011   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   386,561   | 4 &times; Tesla V100 (32GB)  |
|  19  |           ResNet            |   0.126 &plusmn; 0.094   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,300,354  | 4 &times; Tesla V100 (32GB)  |

## Leaderboard for Beta-lactamase Activity Prediction

- **Task type** - Protein-wise Regression
- **Dataset statistics** - #Train: 4,158 &nbsp; #Validation: 520 &nbsp; #Test: 520
- **Evaluation metric** - Spearman's Rho on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split.
- **Description** - Models are asked to predict the activity among first-order mutants of the TEM-1 beta-lactamase protein. The prediction target is the experimentally tested fitness score (a real number) which records the scaled mutation effect for each mutant.

| Rank |           Method            |   Test Spearman's Rho    | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   | **[MTL] ESM-1b + Contact**  | **0.899 &plusmn; 0.001** | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,279,416 | 4 &times; Tesla V100 (32GB) |
|  2   |     [MTL] ESM-1b + Fold     |   0.882 &plusmn; 0.007   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,170,530 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + SSP      |   0.881 &plusmn; 0.001   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,643,578 | 4 &times; Tesla V100 (32GB) |
|  4   |           ESM-1b            |   0.839 &plusmn; 0.053   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 1 &times; Tesla V100 (32GB) |
|  5   |     [MTL] CNN + Contact     |   0.835 &plusmn; 0.009   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,502,274  | 4 &times; Tesla V100 (32GB) |
|  6   |       [MTL] CNN + SSP       |   0.811 &plusmn; 0.014   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,455,748  | 4 &times; Tesla V100 (32GB) |
|  7   |             CNN             |   0.781 &plusmn; 0.011   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,403,073  | 1 &times; Tesla V100 (32GB) |
|  8   |      [MTL] CNN + Fold       |   0.736 &plusmn; 0.012   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,677,548  | 4 &times; Tesla V100 (32GB)  |
|  9   |          ProtBert           |   0.731 &plusmn; 0.226   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |
|  10  |             DDE             |   0.623 &plusmn; 0.019   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   468,481   | 4 &times; Tesla V100 (32GB)  |
|  11  |       ProtBert (fix)        |   0.616 &plusmn; 0.002   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,981,761 | 4 &times; Tesla V100 (32GB)  |
|  12  |        ESM-1b (fix)         |   0.528 &plusmn; 0.009   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,000,055 | 4 &times; Tesla V100 (32GB)  |
|  13  |            Moran            |   0.375 &plusmn; 0.008   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   386,561   | 4 &times; Tesla V100 (32GB)  |
|  14  |  [MTL] Transformer + Fold   |   0.276 &plusmn; 0.029   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,421,676  | 4 &times; Tesla V100 (32GB)  |
|  15  |         Transformer         |   0.261 &plusmn; 0.015   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,545,985  | 4 &times; Tesla V100 (32GB)  |
|  16  |   [MTL] Transformer + SSP   |   0.197 &plusmn; 0.017   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,810,180  | 4 &times; Tesla V100 (32GB)  |
|  17  |           ResNet            |   0.152 &plusmn; 0.029   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,300,354  | 4 &times; Tesla V100 (32GB)  |
|  18  | [MTL] Transformer + Contact |   0.142 &plusmn; 0.063   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,071,298  | 4 &times; Tesla V100 (32GB)  |
|  19  |            LSTM             |   0.139 &plusmn; 0.051   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,080,328  | 4 &times; Tesla V100 (32GB)  |

## Leaderboard for Solubility Prediction

- **Task type** - Protein-wise Classification
- **Dataset statistics** - #Train: 62,478 &nbsp; #Validation: 6,942 &nbsp; #Test: 1,999
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 30% sequence identity cutoff against the test set.
- **Description** - Models are required to predict whether a protein is soluble or not (binary classification).

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |   **[MTL] CNN + Contact**   | **70.63 &plusmn; 0.34** | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,503,299  | 4 &times; Tesla V100 (32GB) |
|  2   |   [MTL] ESM-1b + Contact    |   70.46 &plusmn; 0.16   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,280,697 | 4 &times; Tesla V100 (32GB) |
|  3   |           ESM-1b            |   70.23 &plusmn; 0.75   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,001,336 | 4 &times; Tesla V100 (32GB) |
|  4   |            LSTM             |   70.18 &plusmn; 0.63   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,080,969  | 4 &times; Tesla V100 (32GB) |
|  5   |         Transformer         |   70.12 &plusmn; 0.31   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,546,498  | 4 &times; Tesla V100 (32GB) |
|  6   |     [MTL] ESM-1b + SSP      |   70.03 &plusmn; 0.15   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,644,859 | 4 &times; Tesla V100 (32GB) |
|  7   | [MTL] Transformer + Contact |   70.03 &plusmn; 0.42   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,071,811  | 4 &times; Tesla V100 (32GB) |
|  8   |       [MTL] CNN + SSP       |   69.85 &plusmn; 0.62   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,456,773  | 4 &times; Tesla V100 (32GB) |
|  9   |   [MTL] Transformer + SSP   |   69.81 &plusmn; 0.46   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,810,693  | 4 &times; Tesla V100 (32GB) |
|  10  |      [MTL] CNN + Fold       |   69.23 &plusmn; 0.10   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,678,573  | 4 &times; Tesla V100 (32GB) |
|  11  |  [MTL] Transformer + Fold   |   68.85 &plusmn; 0.43   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,422,189  | 4 &times; Tesla V100 (32GB) |
|  12  |          ProtBert           |   68.15 &plusmn; 0.92   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,982,786 | 4 &times; Tesla V100 (32GB) |
|  13  |           ResNet            |   67.33 &plusmn; 1.46   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,300,867  | 4 &times; Tesla V100 (32GB) |
|  14  |        ESM-1b (fix)         |   67.02 &plusmn; 0.40   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,001,336 | 4 &times; Tesla V100 (32GB) |
|  15  |     [MTL] ESM-1b + Fold     |   64.80 &plusmn; 0.49   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,171,811 | 4 &times; Tesla V100 (32GB) |
|  16  |             CNN             |   64.43 &plusmn; 0.25   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,404,098  | 4 &times; Tesla V100 (32GB) |
|  17  |             DDE             |   59.77 &plusmn; 1.21   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   468,994   | 4 &times; Tesla V100 (32GB)  |
|  18  |       ProtBert (fix)        |   59.17 &plusmn; 0.21   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,982,786 | 4 &times; Tesla V100 (32GB) |
|  19  |            Moran            |   57.73 &plusmn; 1.33   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   387,074   | 4 &times; Tesla V100 (32GB)  |

# Protein Localization Prediction

## Leaderboard for Subcellular Localization Prediction

- **Task type** - Protein-wise Classification
- **Dataset statistics** - #Train: 8,945 &nbsp; #Validation: 2,248 &nbsp; #Test: 2,768
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 30% sequence identity cutoff against the test set.
- **Description** - Models are required to predict where a natural protein locates in the cell. The label denotes 10 possible locations.

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |      **ESM-1b (fix)**       | **79.82 &plusmn; 0.18** | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,011,584 | 4 &times; Tesla V100 (32GB) |
|  2   |   [MTL] ESM-1b + Contact    |   78.86 &plusmn; 0.75   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,290,945 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + Fold     |   78.43 &plusmn; 0.28   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,182,059 | 4 &times; Tesla V100 (32GB) |
|  4   |           ESM-1b            |   78.13 &plusmn; 0.49   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,011,584 | 4 &times; Tesla V100 (32GB) |
|  5   |     [MTL] ESM-1b + SSP      |   78.00 &plusmn; 0.34   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,655,107 | 4 &times; Tesla V100 (32GB) |
|  6   |          ProtBert           |   76.53 &plusmn; 0.93   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,990,986 | 4 &times; Tesla V100 (32GB) |
|  7   |            LSTM             |   62.98 &plusmn; 0.37   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,086,097  | 4 &times; Tesla V100 (32GB) |
|  8   |       ProtBert (fix)        |   59.44 &plusmn; 0.16   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,990,986 | 4 &times; Tesla V100 (32GB) |
|  9   |     [MTL] CNN + Contact     |   59.07 &plusmn; 0.45   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,511,499  | 4 &times; Tesla V100 (32GB) |
|  10  |             CNN             |   58.73 &plusmn; 1.05   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,412,298  | 4 &times; Tesla V100 (32GB) |
|  11  |  [MTL] Transformer + Fold   |   56.74 &plusmn; 0.29   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,426,293  | 4 &times; Tesla V100 (32GB) |
|  12  |   [MTL] Transformer + SSP   |   56.70 &plusmn; 0.16   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,814,797  | 4 &times; Tesla V100 (32GB) |
|  13  |       [MTL] CNN + SSP       |   56.64 &plusmn; 0.33   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,464,973  | 4 &times; Tesla V100 (32GB) |
|  14  |      [MTL] CNN + Fold       |   56.54 &plusmn; 0.65   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,686,773  | 4 &times; Tesla V100 (32GB) |
|  15  |         Transformer         |   56.02 &plusmn; 0.82   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,550,602  | 4 &times; Tesla V100 (32GB) |
|  16  | [MTL] Transformer + Contact |   52.92 &plusmn; 0.64   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,075,915  | 4 &times; Tesla V100 (32GB) |
|  17  |           ResNet            |   52.30 &plusmn; 3.51   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,304,971  | 4 &times; Tesla V100 (32GB) |
|  18  |             DDE             |   49.17 &plusmn; 0.40   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   473,098   | 4 &times; Tesla V100 (32GB)  |
|  19  |            Moran            |   31.13 &plusmn; 0.47   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   391,178   | 4 &times; Tesla V100 (32GB)  |

## Leaderboard for Binary Localization Prediction

- **Task type** - Protein-wise Classification
- **Dataset statistics** - #Train: 5,161 &nbsp; #Validation: 1,727 &nbsp; #Test: 1,746
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 30% sequence identity cutoff against the test set.
- **Description** - Models are asked to predict whether a protein is "membrane-bound" or "soluble" (binary classification).

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   | **[MTL] ESM-1b + Contact**  | **92.50 &plusmn; 0.26** | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,280,697 | 4 &times; Tesla V100 (32GB) |
|  2   |           ESM-1b            |   92.40 &plusmn; 0.35   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,001,336 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + SSP      |   92.26 &plusmn; 0.20   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 655,644,859 | 4 &times; Tesla V100 (32GB) |
|  4   |     [MTL] ESM-1b + Fold     |   91.83 &plusmn; 0.20   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,171,811 | 4 &times; Tesla V100 (32GB) |
|  5   |        ESM-1b (fix)         |   91.61 &plusmn; 0.10   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,001,336 | 4 &times; Tesla V100 (32GB) |
|  6   |          ProtBert           |   91.32 &plusmn; 0.89   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,982,786 | 4 &times; Tesla V100 (32GB) |
|  7   |            LSTM             |   88.11 &plusmn; 0.14   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,080,969  | 4 &times; Tesla V100 (32GB) |
|  8   |             CNN             |   82.67 &plusmn; 0.32   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,404,098  | 4 &times; Tesla V100 (32GB) |
|  9   |     [MTL] CNN + Contact     |   82.67 &plusmn; 0.72   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,503,299  | 4 &times; Tesla V100 (32GB) |
|  10  |       [MTL] CNN + SSP       |   81.83 &plusmn; 0.86   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  7,456,773  | 4 &times; Tesla V100 (32GB) |
|  11  |       ProtBert (fix)        |   81.54 &plusmn; 0.09   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,982,786 | 4 &times; Tesla V100 (32GB) |
|  12  |      [MTL] CNN + Fold       |   81.14 &plusmn; 0.40   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,678,573  | 4 &times; Tesla V100 (32GB) |
|  13  |           ResNet            |   78.99 &plusmn; 4.41   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,300,867  | 4 &times; Tesla V100 (32GB) |
|  14  |             DDE             |   77.43 &plusmn; 0.42   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   468,994   | 4 &times; Tesla V100 (32GB)  |
|  15  |  [MTL] Transformer + Fold   |   76.27 &plusmn; 0.57   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,422,189  | 4 &times; Tesla V100 (32GB) |
|  16  |         Transformer         |   75.74 &plusmn; 0.74   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,546,498  | 4 &times; Tesla V100 (32GB) |
|  17  |   [MTL] Transformer + SSP   |   75.20 &plusmn; 1.23   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 21,810,693  | 4 &times; Tesla V100 (32GB) |
|  18  | [MTL] Transformer + Contact |   74.98 &plusmn; 0.77   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,071,811  | 4 &times; Tesla V100 (32GB) |
|  19  |            Moran            |   55.63 &plusmn; 0.85   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   387,074   | 4 &times; Tesla V100 (32GB)  |

# Protein Structure Prediction

## Leaderboard for Contact Prediction

- **Task type** - Residue-pair Classification
- **Dataset statistics** - #Train: 25,299 &nbsp; #Validation: 224 &nbsp; #Test: 40
- **Evaluation metric** - L/5 Precision (L: protein sequence length) on the test set (the higher, the better)
- **Dataset splitting scheme** - Adopt the splits of ProteinNet; use the data of CASP12 for test. 
- **Description** - Models are asked to estimate whether each pair of residues contact or not (binary classification).

| Rank |          Method          |   Test L/5 Precision    | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |        **ESM-1b**        | **45.78 &plusmn; 2.73** | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,638,455 | 4 &times; Tesla V100 (32GB) |
|  2   |       ESM-1b (fix)       |   40.37 &plusmn; 0.22   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,638,455 | 4 &times; Tesla V100 (32GB) |
|  3   |         ProtBert         |   39.66 &plusmn; 1.21   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,030,337 | 4 &times; Tesla V100 (32GB) |
|  4   |   [MTL] ESM-1b + Fold    |   35.86 &plusmn; 1.27   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,808,930 | 4 &times; Tesla V100 (32GB) |
|  5   |    [MTL] ESM-1b + SSP    |  32.03 &plusmn; 12.25   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,281,978 | 4 &times; Tesla V100 (32GB) |
|  6   |           LSTM           |   26.34 &plusmn; 0.65   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 29,948,808  | 4 &times; Tesla V100 (32GB) |
|  7   |      ProtBert (fix)      |   24.35 &plusmn; 0.44   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,030,337 | 4 &times; Tesla V100 (32GB) |
|  8   |          ResNet          |   20.43 &plusmn; 0.74   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,562,498  | 4 &times; Tesla V100 (32GB) |
|  9   |       Transformer        |   17.50 &plusmn; 0.77   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,808,129  | 4 &times; Tesla V100 (32GB) |
|  10  | [MTL] Transformer + SSP  |   12.76 &plusmn; 1.62   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,072,324  | 4 &times; Tesla V100 (32GB) |
|  11  |           CNN            |   10.00 &plusmn; 0.20   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,451,649  | 4 &times; Tesla V100 (32GB) |
|  12  |     [MTL] CNN + Fold     |   5.87 &plusmn; 0.21    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  9,726,124  | 4 &times; Tesla V100 (32GB) |
|  13  |     [MTL] CNN + SSP      |   5.73 &plusmn; 0.66    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,504,324  | 4 &times; Tesla V100 (32GB) |
|  14  | [MTL] Transformer + Fold |   2.04 &plusmn; 0.31    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,683,820  | 4 &times; Tesla V100 (32GB) |

## Leaderboard for Fold Classification

- **Task type** - Protein-wise Classification
- **Dataset statistics** - #Train: 12,312 &nbsp; #Validation: 736 &nbsp; #Test: 718
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Adopt data from SCOP 1.75 database; entire superfamilies are held out from training to compose the test set.
- **Description** - Models are required to classify the global structural topology of a protein on the fold level. The label indicates 1195 different folding topologies. Models are expected to detect the proteins with similar structures but dissimilar sequences, *i.e.*, performing remote homology detection.

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   | **[MTL] ESM-1b + Contact**  | **32.10 &plusmn; 0.72** | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 658,808,930 | 4 &times; Tesla V100 (32GB) |
|  2   |        ESM-1b (fix)         |   29.95 &plusmn; 0.21   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,529,569 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + SSP      |   28.63 &plusmn; 1.55   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,173,092 | 4 &times; Tesla V100 (32GB) |
|  4   |           ESM-1b            |   28.17 &plusmn; 2.05   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,529,569 | 4 &times; Tesla V100 (32GB) |
|  5   |          ProtBert           |   16.94 &plusmn; 0.42   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,205,611 | 4 &times; Tesla V100 (32GB) |
|  6   |       [MTL] CNN + SSP       |   11.67 &plusmn; 0.56   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,679,598  | 4 &times; Tesla V100 (32GB) |
|  7   |     [MTL] CNN + Contact     |   11.07 &plusmn; 0.38   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  9,726,124  | 4 &times; Tesla V100 (32GB) |
|  8   |             CNN             |   10.93 &plusmn; 0.35   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,626,923  | 1 &times; Tesla V100 (32GB) |
|  9   |       ProtBert (fix)        |   10.74 &plusmn; 0.93   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,205,611 | 4 &times; Tesla V100 (32GB) |
|  10  |             DDE             |   9.57 &plusmn; 0.46    | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |  1,081,003  | 4 &times; Tesla V100 (32GB) |
|  11  | [MTL] Transformer + Contact |   9.16 &plusmn; 0.91    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,683,820  | 4 &times; Tesla V100 (32GB) |
|  12  |           ResNet            |   8.89 &plusmn; 1.45    | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,912,876  | 4 &times; Tesla V100 (32GB) |
|  13  |         Transformer         |   8.52 &plusmn; 0.63    | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 22,158,507  | 4 &times; Tesla V100 (32GB) |
|  14  |            LSTM             |   8.24 &plusmn; 1.61    | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,845,682  | 4 &times; Tesla V100 (32GB) |
|  15  |   [MTL] Transformer + SSP   |   8.14 &plusmn; 0.76    | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,422,702  | 4 &times; Tesla V100 (32GB) |
|  16  |            Moran            |   7.10 &plusmn; 0.56    | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   999,083   | 4 &times; Tesla V100 (32GB) |

## Leaderboard for Secondary Structure Prediction

- **Task type** - Residue-wise Classification
- **Dataset statistics** - #Train: 8,678 &nbsp; #Validation: 2,170 &nbsp; #Test: 513
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Training & validation: from NetSurfP; Test: CB513 dataset.
- **Description** - Models are asked to predict the secondary structure (*i.e.*, coil, strand or helix) of each residue.

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   | **[MTL] ESM-1b + Contact**  | **83.21 &plusmn; 0.32** | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 657,281,978 | 4 &times; Tesla V100 (32GB) |
|  2   |        ESM-1b (fix)         |   83.14 &plusmn; 0.10   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,002,617 | 4 &times; Tesla V100 (32GB) |
|  3   |           ESM-1b            |   82.73 &plusmn; 0.21   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 654,002,617 | 4 &times; Tesla V100 (32GB) |
|  4   |     [MTL] ESM-1b + Fold     |   82.27 &plusmn; 0.23   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 657,173,092 | 4 &times; Tesla V100 (32GB) |
|  5   |          ProtBert           |   82.18 &plusmn; 0.05   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,983,811 | 4 &times; Tesla V100 (32GB) |
|  6   |           ResNet            |   69.56 &plusmn; 0.20   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,301,380  | 4 &times; Tesla V100 (32GB) |
|  7   |            LSTM             |   68.99 &plusmn; 0.76   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 28,312,970  | 4 &times; Tesla V100 (32GB) |
|  8   |     [MTL] CNN + Contact     |   66.13 &plusmn; 0.06   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  8,504,324  | 4 &times; Tesla V100 (32GB) |
|  9   |             CNN             |   66.07 &plusmn; 0.06   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  6,405,123  | 1 &times; Tesla V100 (32GB) |
|  10  |      [MTL] CNN + Fold       |   65.93 &plusmn; 0.04   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  8,679,598  | 4 &times; Tesla V100 (32GB) |
|  11  | [MTL] Transformer + Contact |   63.10 &plusmn; 0.43   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,072,324  | 4 &times; Tesla V100 (32GB) |
|  12  |       ProtBert (fix)        |   62.51 &plusmn; 0.06   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 420,983,811 | 4 &times; Tesla V100 (32GB) |
|  13  |         Transformer         |   59.62 &plusmn; 0.94   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,547,011  | 4 &times; Tesla V100 (32GB) |
|  14  |  [MTL] Transformer + Fold   |   50.93 &plusmn; 0.20   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,422,702  | 4 &times; Tesla V100 (32GB) |

# Protein-Protein Interaction (PPI) Prediction

## Leaderboard for Yeast PPI Prediction

- **Task type** - Protein-pair Classification
- **Dataset statistics** - #Train: 1,668 &nbsp; #Validation: 131 &nbsp; #Test: 373
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 40% sequence identity cutoff against the test set.
- **Description** - Models are asked to predict whether two yeast proteins interact or not (binary classification).

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |      **ESM-1b (fix)**       | **66.07 &plusmn; 0.58** | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,639,736 | 4 &times; Tesla V100 (32GB) |
|  2   |     [MTL] ESM-1b + Fold     |   64.76 &plusmn; 1.42   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,810,211 | 4 &times; Tesla V100 (32GB) |
|  3   |          ProtBert           |   63.72 &plusmn; 2.80   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,031,362 | 4 &times; Tesla V100 (32GB) |
|  4   |     [MTL] ESM-1b + SSP      |   62.06 &plusmn; 5.98   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,283,259 | 4 &times; Tesla V100 (32GB) |
|  5   |   [MTL] ESM-1b + Contact    |   58.50 &plusmn; 2.15   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 658,919,097 | 4 &times; Tesla V100 (32GB) |
|  6   |           ESM-1b            |   57.00 &plusmn; 6.38   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,639,736 | 4 &times; Tesla V100 (32GB) |
|  7   |             DDE             |   55.83 &plusmn; 3.13   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   731,138   | 4 &times; Tesla V100 (32GB) |
|  8   |             CNN             |   55.07 &plusmn; 0.02   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,452,674  | 1 &times; Tesla V100 (32GB) |
|  9   |     [MTL] CNN + Contact     |   54.50 &plusmn; 1.61   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  9,551,875  | 4 &times; Tesla V100 (32GB) |
|  10  |         Transformer         |   54.12 &plusmn; 1.27   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,808,642  | 4 &times; Tesla V100 (32GB) |
|  11  |       [MTL] CNN + SSP       |   54.12 &plusmn; 2.87   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,505,349  | 4 &times; Tesla V100 (32GB) |
|  12  |   [MTL] Transformer + SSP   |   54.00 &plusmn; 1.17   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,072,837  | 4 &times; Tesla V100 (32GB) |
|  13  |  [MTL] Transformer + Fold   |   54.00 &plusmn; 2.58   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,684,333  | 4 &times; Tesla V100 (32GB) |
|  14  |       ProtBert (fix)        |   53.87 &plusmn; 0.38   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,031,362 | 4 &times; Tesla V100 (32GB) |
|  15  |            LSTM             |   53.62 &plusmn; 2.72   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,490,569  | 4 &times; Tesla V100 (32GB) |
|  16  |      [MTL] CNN + Fold       |   53.28 &plusmn; 1.91   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  9,727,149  | 4 &times; Tesla V100 (32GB) |
|  17  |            Moran            |   53.00 &plusmn; 0.50   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   649,218   | 4 &times; Tesla V100 (32GB) |
|  18  | [MTL] Transformer + Contact |   52.86 &plusmn; 1.15   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,333,955  | 4 &times; Tesla V100 (32GB) |
|  19  |           ResNet            |   48.91 &plusmn; 1.78   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,563,011  | 4 &times; Tesla V100 (32GB) |

## Leaderboard for Human PPI Prediction

- **Task type** - Protein-pair Classification
- **Dataset statistics** - #Train: 6,844 &nbsp; #Validation: 277 &nbsp; #Test: 227
- **Evaluation metric** - Accuracy on the test set (the higher, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 40% sequence identity cutoff against the test set.
- **Description** - Models are asked to predict whether two human proteins interact or not (binary classification).

| Rank |           Method            |        Test Acc         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:-----------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |      **ESM-1b (fix)**       | **88.06 &plusmn; 0.24** | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,639,736 | 4 &times; Tesla V100 (32GB) |
|  2   |       ProtBert (fix)        |   83.61 &plusmn; 1.34   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,031,362 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + SSP      |   83.00 &plusmn; 0.88   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,283,259 | 4 &times; Tesla V100 (32GB) |
|  4   |   [MTL] ESM-1b + Contact    |   81.66 &plusmn; 2.88   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 658,919,097 | 4 &times; Tesla V100 (32GB) |
|  5   |     [MTL] ESM-1b + Fold     |   80.28 &plusmn; 1.27   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,810,211 | 4 &times; Tesla V100 (32GB) |
|  6   |           ESM-1b            |   78.17 &plusmn; 2.91   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,639,736 | 4 &times; Tesla V100 (32GB) |
|  7   |          ProtBert           |   77.32 &plusmn; 1.10   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,031,362 | 4 &times; Tesla V100 (32GB) |
|  8   |      [MTL] CNN + Fold       |   69.03 &plusmn; 2.68   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  9,727,149  | 4 &times; Tesla V100 (32GB) |
|  9   |           ResNet            |   68.61 &plusmn; 3.78   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,563,011  | 4 &times; Tesla V100 (32GB) |
|  10  |  [MTL] Transformer + Fold   |   67.33 &plusmn; 2.68   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,684,333  | 4 &times; Tesla V100 (32GB) |
|  11  |       [MTL] CNN + SSP       |   66.39 &plusmn; 0.86   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,505,349  | 4 &times; Tesla V100 (32GB) |
|  12  |     [MTL] CNN + Contact     |   65.10 &plusmn; 2.26   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  9,551,875  | 4 &times; Tesla V100 (32GB) |
|  13  |            LSTM             |   63.75 &plusmn; 5.12   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,490,569  | 4 &times; Tesla V100 (32GB) |
|  14  |             DDE             |   62.77 &plusmn; 2.30   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   731,138   | 4 &times; Tesla V100 (32GB) |
|  15  |             CNN             |   62.60 &plusmn; 1.67   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,452,674  | 1 &times; Tesla V100 (32GB) |
|  16  | [MTL] Transformer + Contact |   60.76 &plusmn; 6.87   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,333,955  | 4 &times; Tesla V100 (32GB) |
|  17  |         Transformer         |   59.58 &plusmn; 2.09   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,808,642  | 4 &times; Tesla V100 (32GB) |
|  18  |   [MTL] Transformer + SSP   |   54.80 &plusmn; 2.06   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,072,837  | 4 &times; Tesla V100 (32GB) |
|  19  |            Moran            |   54.67 &plusmn; 4.43   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   649,218   | 4 &times; Tesla V100 (32GB) |

## Leaderboard for PPI Affinity Prediction

- **Task type** - Protein-pair Regression
- **Dataset statistics** - #Train: 2,127 &nbsp; #Validation: 212 &nbsp; #Test: 343
- **Evaluation metric** - RMSE on the test set (the lower, the better)
- **Dataset splitting scheme** - Train: wild-type complexes as well as mutants with at most 2 mutations; Validation: mutants with 3 or 4 mutations; Test: mutants with more than 4 mutations.
- **Description** - Models are required to predict the binding affinity between two proteins, measured by pKd (a real number). This task performs evaluation under a multi-round protein binder design scenario.

| Rank |           Method            |        Test RMSE         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |   **[MTL] CNN + Contact**   | **1.732 &plusmn; 0.044** | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             |  9,550,850  | 4 &times; Tesla V100 (32GB) |
|  2   |   [MTL] ESM-1b + Contact    |   1.893 &plusmn; 0.064   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 658,917,816 | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] ESM-1b + Fold     |   2.002 &plusmn; 0.065   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,808,930 | 4 &times; Tesla V100 (32GB) |
|  4   |     [MTL] ESM-1b + SSP      |   2.031 &plusmn; 0.031   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,281,978 | 4 &times; Tesla V100 (32GB) |
|  5   |          ProtBert           |   2.195 &plusmn; 0.073   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,030,337 | 4 &times; Tesla V100 (32GB) |
|  6   |       [MTL] CNN + SSP       |   2.270 &plusmn; 0.041   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,504,324  | 4 &times; Tesla V100 (32GB) |
|  7   |           ESM-1b            |   2.281 &plusmn; 0.250   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,638,455 | 4 &times; Tesla V100 (32GB) |
|  8   |      [MTL] CNN + Fold       |   2.392 &plusmn; 0.041   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              |  9,726,124  | 4 &times; Tesla V100 (32GB) |
|  9   |         Transformer         |   2.499 &plusmn; 0.156   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 21,808,129  | 4 &times; Tesla V100 (32GB) |
|  10  |  [MTL] Transformer + Fold   |   2.524 &plusmn; 0.146   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 22,683,820  | 4 &times; Tesla V100 (32GB) |
|  11  |   [MTL] Transformer + SSP   |   2.651 &plusmn; 0.034   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,072,324  | 4 &times; Tesla V100 (32GB) |
|  12  | [MTL] Transformer + Contact |   2.733 &plusmn; 0.126   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 22,333,442  | 4 &times; Tesla V100 (32GB) |
|  13  |             CNN             |   2.796 &plusmn; 0.071   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,451,649  | 1 &times; Tesla V100 (32GB) |
|  14  |            LSTM             |   2.853 &plusmn; 0.124   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 27,489,928  | 4 &times; Tesla V100 (32GB) |
|  15  |             DDE             |   2.908 &plusmn; 0.043   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   730,625   | 4 &times; Tesla V100 (32GB) |
|  16  |            Moran            |   2.984 &plusmn; 0.026   | [paper](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) |                   /                    |   648,705   | 4 &times; Tesla V100 (32GB) |
|  17  |       ProtBert (fix)        |   2.996 &plusmn; 0.462   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,030,337 | 4 &times; Tesla V100 (32GB) |
|  18  |           ResNet            |   3.005 &plusmn; 0.244   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 11,562,498  | 4 &times; Tesla V100 (32GB) |
|  19  |        ESM-1b (fix)         |   3.031 &plusmn; 0.014   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,638,455 | 4 &times; Tesla V100 (32GB) |

# Protein-Ligand Interaction (PLI) Prediction

## Leaderboard for PLI Affinity Prediction on PDBbind

- **Task type** - Protein-ligand Regression
- **Dataset statistics** - #Train: 16,436 &nbsp; #Validation: 937 &nbsp; #Test: 285
- **Evaluation metric** - RMSE on the test set (the lower, the better)
- **Dataset splitting scheme** - Random split; remove redundancy in training and validation sets with 90% sequence identity cutoff against the test set.
- **Description** - Models are asked to predict the binding affinity between a protein and a ligand, measured by pKd (a real number).

| Rank |           Method            |        Test RMSE         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |     **[MTL] CNN + SSP**     | **1.295 &plusmn; 0.030** | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,984,068  | 4 &times; Tesla V100 (32GB) |
|  2   |      [MTL] CNN + Fold       |   1.316 &plusmn; 0.064   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 10,205,868  | 4 &times; Tesla V100 (32GB) |
|  3   |     [MTL] CNN + Contact     |   1.328 &plusmn; 0.033   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 10,030,594  | 4 &times; Tesla V100 (32GB) |
|  4   |        ESM-1b (fix)         |   1.368 &plusmn; 0.076   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,790,519 | 4 &times; Tesla V100 (32GB) |
|  5   |             CNN             |   1.376 &plusmn; 0.008   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,931,393  | 1 &times; Tesla V100 (32GB) |
|  6   |   [MTL] Transformer + SSP   |   1.387 &plusmn; 0.019   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,814,212  | 4 &times; Tesla V100 (32GB) |
|  7   |     [MTL] ESM-1b + SSP      |   1.419 &plusmn; 0.026   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,434,042 | 4 &times; Tesla V100 (32GB) |
|  8   |     [MTL] ESM-1b + Fold     |   1.435 &plusmn; 0.015   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,960,994 | 4 &times; Tesla V100 (32GB) |
|  9   |           ResNet            |   1.441 &plusmn; 0.064   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 12,304,386  | 4 &times; Tesla V100 (32GB) |
|  10  |         Transformer         |   1.455 &plusmn; 0.070   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 22,550,017  | 4 &times; Tesla V100 (32GB) |
|  11  |       ProtBert (fix)        |   1.457 &plusmn; 0.024   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,510,081 | 4 &times; Tesla V100 (32GB) |
|  12  |            LSTM             |   1.457 &plusmn; 0.131   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 28,215,432  | 4 &times; Tesla V100 (32GB) |
|  13  |   [MTL] ESM-1b + Contact    |   1.458 &plusmn; 0.003   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 659,069,880 | 4 &times; Tesla V100 (32GB) |
|  14  |  [MTL] Transformer + Fold   |   1.531 &plusmn; 0.181   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 23,425,708  | 4 &times; Tesla V100 (32GB) |
|  15  |           ESM-1b            |   1.559 &plusmn; 0.164   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,790,519 | 4 &times; Tesla V100 (32GB) |
|  16  |          ProtBert           |   1.562 &plusmn; 0.072   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,510,081 | 4 &times; Tesla V100 (32GB) |
|  17  | [MTL] Transformer + Contact |   1.574 &plusmn; 0.215   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 23,075,330  | 4 &times; Tesla V100 (32GB) |

## Leaderboard for PLI Affinity Prediction on BindingDB

- **Task type** - Protein-ligand Regression
- **Dataset statistics** - #Train: 7,900 &nbsp; #Validation: 878 &nbsp; #Test: 5,230
- **Evaluation metric** - RMSE on the test set (the lower, the better)
- **Dataset splitting scheme** - Four protein classes (ER, GPCR, ion channels and receptor tyrosine kinases) are held out from training and validation for generalization test. 
- **Description** - Models are asked to predict the binding affinity between a protein and a ligand, measured by pKd (a real number).

| Rank |           Method            |        Test RMSE         | Reference |             External data              |   #Params   |          Hardware           |
|:----:|:---------------------------:|:------------------------:|:---------:|:--------------------------------------:|:-----------:|:---------------------------:|
|  1   |    **[MTL] CNN + Fold**     | **1.462 &plusmn; 0.044** | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 10,205,868  | 4 &times; Tesla V100 (32GB) |
|  2   |  [MTL] Transformer + Fold   |   1.464 &plusmn; 0.007   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              Fold for MTL              | 23,425,708  | 4 &times; Tesla V100 (32GB) |
|  3   |       [MTL] CNN + SSP       |   1.481 &plusmn; 0.036   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               |  8,984,068  | 4 &times; Tesla V100 (32GB) |
|  4   |     [MTL] ESM-1b + SSP      |   1.482 &plusmn; 0.014   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |   UniRef50 for pre-train; SSP for MTL   | 657,434,042 | 4 &times; Tesla V100 (32GB) |
|  5   |   [MTL] ESM-1b + Contact    |   1.490 &plusmn; 0.033   | [paper](https://arxiv.org/pdf/2206.02096.pdf) | UniRef50 for pre-train; Contact for MTL | 659,069,880 | 4 &times; Tesla V100 (32GB) |
|  6   | [MTL] Transformer + Contact |   1.490 &plusmn; 0.058   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 23,075,330  | 4 &times; Tesla V100 (32GB) |
|  7   |             CNN             |   1.497 &plusmn; 0.022   | [paper](https://arxiv.org/pdf/2011.03443.pdf) |                   /                    |  7,931,393  | 4 &times; Tesla V100 (32GB) |
|  8   |     [MTL] CNN + Contact     |   1.501 &plusmn; 0.035   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |            Contact for MTL             | 10,030,594  | 4 &times; Tesla V100 (32GB) |
|  9   |     [MTL] ESM-1b + Fold     |   1.511 &plusmn; 0.017   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |  UniRef50 for pre-train; Fold for MTL   | 658,960,994 | 4 &times; Tesla V100 (32GB) |
|  10  |   [MTL] Transformer + SSP   |   1.519 &plusmn; 0.050   | [paper](https://arxiv.org/pdf/2206.02096.pdf) |              SSP for MTL               | 22,814,212  | 4 &times; Tesla V100 (32GB) |
|  11  |          ProtBert           |   1.549 &plusmn; 0.019   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,510,081 | 4 &times; Tesla V100 (32GB) |
|  12  |           ESM-1b            |   1.556 &plusmn; 0.047   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,790,519 | 4 &times; Tesla V100 (32GB) |
|  13  |           ResNet            |   1.565 &plusmn; 0.033   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 12,304,386  | 4 &times; Tesla V100 (32GB) |
|  14  |         Transformer         |   1.566 &plusmn; 0.052   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 22,550,017  | 4 &times; Tesla V100 (32GB) |
|  15  |        ESM-1b (fix)         |   1.571 &plusmn; 0.032   | [paper](https://www.pnas.org/content/pnas/118/15/e2016239118.full.pdf) |         UniRef50 for pre-train          | 655,790,519 | 4 &times; Tesla V100 (32GB) |
|  16  |            LSTM             |   1.572 &plusmn; 0.022   | [paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf) |                   /                    | 28,215,432  | 4 &times; Tesla V100 (32GB) |
|  17  |       ProtBert (fix)        |   1.649 &plusmn; 0.022   | [paper](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf) |            BFD for pre-train            | 422,510,081 | 4 &times; Tesla V100 (32GB) |
