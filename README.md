
# Online Model Adaptation with Feedforward Compensation

This repository contains the code for Feedforward and Online Model Adaptation, as demonstrated in the following papers:

Abulikemu Abuduweili, and Changliu Liu, "[Online Model Adaptation with Feedforward Compensation](https://openreview.net/forum?id=4x2RUQ99sGz)," CoRL, 2023.


## Abstract
To cope with distribution shifts or non-stationarity in system dynamics, online adaptation algorithms have been introduced to update offline-learned prediction models in real-time. Existing online adaptation methods focus on optimizing the prediction model by utilizing feedback from the latest prediction error. Unfortunately, this feedback-based approach is susceptible to forgetting past information. This work proposes an online adaptation method with feedforward compensation, which uses critical data samples from a memory buffer, instead of the latest samples, to optimize the prediction model. We prove that the proposed approach achieves a smaller error bound compared to previously utilized methods in slow time-varying systems.  Furthermore, our feedforward adaptation technique is capable of estimating an uncertainty bound for predictions. 


## About Code

Install Requirments
```bash
pip install numpy pandas scikit-learn torch

```
Training the Model on etth1/ill/exchange:
```bash
python train.py --data etth1

```
Adapting the trained model with Feedforward Adaptation:
```bash
python adap.py --data etth1 --adapt sgd --buffer_size 1000

```


## Citation
If you find the code helpful in your research or work, please cite the following papers.
```BibTex
@inproceedings{
abuduweili2023online,
title={Online Model Adaptation with Feedforward Compensation},
author={ABULIKEMU ABUDUWEILI and Changliu Liu},
booktitle={7th Annual Conference on Robot Learning},
year={2023},
}

```




