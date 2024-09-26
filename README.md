# weavenet for keras

<div align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/>
    </a>
    <a href="https://github.com/jwwhangbo/keras_weavenet">
        <img src="https://img.shields.io/github/v/release/jwwhangbo/keras_weavenet"
        />
    </a>
    <img src="https://img.shields.io/badge/keras-v.3.4.1-green"/>
<div/>

📆 last edited: Sep 26, 2024

This repository is a keras compatible training code for weavenet, heavily inspired from [deepchem](https://github.com/deepchem/deepchem). The code here is based on their proprietary epoch algorithm(which doesn't work at the moment) so I decided to write the model for keras.

The model is from the paper "Molecular Graph Convolutions: Moving Beyond Fingerprints" from google available [here](https://arxiv.org/abs/1603.00856)

In short, the model takes two inputs; atom features and pair features. The fundamental operations rely on atom-to-atom, atom-to-pair, pair-to-atom, and pair-to-pair convolutions. These are illustrated in the figures below, taken from the paper.

<img width="402" alt="Screenshot 2024-09-26 at 6 05 41 PM" src="https://github.com/user-attachments/assets/3f3547d6-59ba-4f2b-a8c7-d4ae3a846fb1">

atom-to-pair

<img width="406" alt="Screenshot 2024-09-26 at 6 05 51 PM" src="https://github.com/user-attachments/assets/e57c28e7-f52b-4f92-a1e2-3fe5762dd872">

pair-to-atom

<img width="380" alt="Screenshot 2024-09-26 at 6 05 58 PM" src="https://github.com/user-attachments/assets/160c5489-9aab-4050-9b84-146368add092">

convolution gathering

<img width="409" alt="Screenshot 2024-09-26 at 6 06 09 PM" src="https://github.com/user-attachments/assets/963c51ea-5c39-48f0-8fb3-2b853884c226">

convolutional structure

This code was written to be run on colab


