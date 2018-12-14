The goal of the project is to be able to predict the 3-dimensional structure of a folded protein given its primary structure – a sequence of amino acids that it’s build from.

## Overview
Predicting proteins’ tertiary structure from their primary structure is one of the most important unsolved problems of biochemistry. Current methods are inaccurate and expensive. Machine Learning offers new toolset that promises cheaper and much more efficient solutions. One of the recent breakthroughs is Mohammed AlQuraishi’s paper on END-TO-END DIFFERENTIABLE LEARNING OF PROTEIN STRUCTURE , which this project aims to partially reproduce. 

The project is part of an NLP and Deep Learning course at ITU University Copenhagen.

## Technical Details
We use ProteinNet dataset as introduced by AlQuraishi and write the entire data processing pipeline in Tensorflow. The heart of the model is a multilayer bidirectional LSTM network. The model is trained using ADAM optimizer on an MSE and MAE losses between predicted and true dihedral (torsional) angles. 
Source: AlQuraishi, End-to-end differentiable learning of protein structure

The main challenge in going from a sequence of letters representing amino acids to a 3-dimensional protein structure is 1) an efficient loss calculation and 2) output of the network being angular.

In the paper, protein’s tertiary structure is approximated by 3 torsional angles per amino acid, which then are used to reproduce the 3-dimensional structure to compute loss in that space. That process though is very computationally expensive, thus we’re focusing on a regression task that minimizes the loss between angles directly.
We need to angularize the output of the network to compare it with true angles. As one approach, we predict 3 values directly squeezed into the range of [-pi, pi] by a scaled tanh. Another approach is to predict 6 values split into 3 pairs of 2 where each pair represents a vector in a 2-dimensional space, that can then be converted into an angle using atan2 function.

## Technologies Used
Python, Tensorflow and Jupyter Notebook

## Configuration Details
For details on the model configuration required to run the entire pipeline, a readme file is attached to the model folder.
