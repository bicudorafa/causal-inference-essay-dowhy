# Causal Inference Essay - DoWhy
## Description
This repository contains two essays (a synthetic and a real one) on how to implement the causal inference framework to assess causal relationships on data, based on the tools contained on the [DoWhy](https://github.com/Microsoft/dowhy) Microsoft Project.
### Synthetic Use Case
By using built-in methods to generate synthetic data with linear causal relationships, this essay demonstrates most of the package's methods, possibilities, and functionalities in a "controlled" environment (i.e. it's know in advance what is the true Average Treatment Effect, then what would be the population counterfactuals), along with brief about Causal Inference theory behind the package approach.
### Real Use Case: Rereading of the Classic Lalonde Essay
After the above demonstration, the focus shifts to a practical use case to better understand Dowhy's capabilities in real-world scenarios. The chosen exercise was a re-edition of the famous Rajeev H. Dehejia and Sadek Wahba [paper](https://users.nber.org/~rdehejia/papers/matching.pdf) in which they had challenged the results obtained by Lalonde (tl;dr the ordinary econometric techniques aren't able to estimate causal relationships from observational data with reasonable confidence) by showing how more modern algorithms (to the time), such as the Propensity Score Matching, are able to achieve incredible performance.
## Prerequisites
Everything performed on this repository was stored in a Docker image (description on to reproduce it below). The unique basic required is to have [Docker's](https://docs.docker.com/engine/install/) most recent version installed.
## Installation
You can either download the repository and rebuild the Docker image from the scratch on your machine by running on your shell (inside your local directory in which the repo is stored):
```
docker build -t myaccount/new_project .
docker run -p 8888:8888 myaccount/new_project
```
As it's possible to directly pull it from the [Docker Hub](https://hub.docker.com/):
```
docker pull bicudorafa/dowhy-env:0.1
docker run -p 8888:8888 bicudorafa/dowhy-env:0.1
```
## Contact
If you have any questions, suggestions, or issues to report, feel free to open an issue, or reach me out through my profile.