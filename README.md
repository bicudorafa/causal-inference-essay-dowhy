# Causal Inference - DoWhy Samples
This repository contains contains use cases (a synthetic and a real one) on how to implement the causal inference framework to assess causal relationships on data, based on the tools contained on the [DoWhy](https://github.com/Microsoft/dowhy) Microsoft Project.
# Installation
### Environment
For most the calculations, I used the Jupyter containerised version provided by the [Jupyter Docker Stackes](https://jupyter-docker-stacks.readthedocs.io/en/latest/) project, that provides images containing docker images of jupyter ready to use environments. For this project, I used the minimal notebook image, tag: dc9744740e12 (jupyter/minimal-notebook:dc9744740e12)
### Dependencies
To install all the necessary dependencies for this project, run:

```
pip install - requirements.txt
```
# Summary
This project aims to apply the Causal Inference Methodology to estimate causal relationships in data. To reach this goal, the methodology is applied both on synthetic, and real data (WIP), to enable the possibility of better understanding the package's (and the framework) possibilities and caveats.
### DoWhy - Synthetic
Using built-in methods to generate data with linear causal relationships, this essay tries to demonstrate all the package's methods, possibilites and funcionalities in a controled envionment (as it allow us to compare the real causal relationships and the ones estimated though the analysis).
### DoWhy - Real Usa Case (WIP)
After the above demonstrastion, the attention was redirected to a practical use case to better assess DoWhy's performance on real scenarios. The analysis scope was to reapply the framework, asses what were the gains provided byt it (and the caveats), and what would be further improvements to keep increasing the quality of assesments built on the top of DoWhy.