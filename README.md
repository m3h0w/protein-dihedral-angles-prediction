# Protein Tertiary Structure Prediction
This project aims at reproducing selected part of Mohammed AlQuraishi's work on End-to-end differentiable learning of protein structure (https://www.biorxiv.org/content/early/2018/08/29/265231), and Gao et al. on RaptorX (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2065-x).

Report with the results: https://drive.google.com/file/d/1-SFavU5i6XlHK2sswy60k5TowgButezy/view

In the main folder you'll find notebooks that show examples of how to use the model.

Model details and configuration can be found in the "model" folder.

# Model configuration details

This tesnorflow model uses ProteinNet dataset (in the tensor version) as available in the preliminary release here: https://github.com/aqlaboratory/proteinnet

## Input

Input is comprised of aminoacid sequences and evolutionary data (PSSM) and parsing is done through the DataHandler object, which
is written in the old queue paradigm (instead of the new tensorflow Data Pipeline).

### Files

Files to consider as trianing inputs are decided based on following variables:
- data_path: path to the ProteinNet containing casps (each casp then contains training, validation and test folders)
- casps: a list of strings defining which casps should be loaded 
- percentages: a list of integers defining which structure identity clusters should be loaded

### Features

Controlled fully by a boolean: include_evo, that controlls if evolutionary features should be used together with aminoacid sequences.

## Model

The model is controlled using the Model class defined in the model.py.

Its behaviour is fully determined by
a set of arguments passed to the contructors: <i>n_angles, model_type, prediction_mode, ang_mode, loss_mode,dropout_rate.</i>

- n_angles: 2 if should predict only phi and psi and 3 if phi, psi, and omega
- model_type: see <b>Core</b> (below)
- prediction_mode: see <b>Prediction</b> (below)
- ang_mode: see <b>Predictions and corresponding loss modes -> Angularization</b>
- loss_mode: see <b>Predictions and corresponding loss modes</b>
- dropout_rate: controlls the regularization applied to the core model
- regularize_vectors: controlls if regularization loss should be applied to vectors to keep them on unit circle (only available in 'regression_vectors' mode)

### Core

Both CNNs are composed of resnet type architecture with residual connections in between layers,
batch normalization after each layer and dropout after every second layer.

Filter numbers (neurons per layer) start at 32 and are incrementally doubled every 2 layers. Filter size is fixed at 5.

Modes:
- cnn_big: 8 layers
- cnn_small: 6 layers
- bilstm: bidirectional lstm. 1 layer, 128 neurons

### Predictions and corresponding loss modes

Modes:
- regression_angles

    <i>n_angles</i> values predicted in a dense layer, piped through tanh or cos and multiplied by pi to fit radian range

    Available Angularization Modes: 'tanh' or 'cos'
    
    Available loss modes: 'angular_mae' or 'mae', both are applied to angles

- regression_vectors

    <i>n_angles*2</i> values predicted in a dense layer, converted to angles by passing through an atan2 function
    
    Available loss modes: 'angular_mae' or 'mae'. Angular mae is applied to angles, mae is applied to vectors.

- alphabet_angles

    n_angles values predicted by calculating a weighted average of an alpahabet of n_clusters size and a probability disttribution over that alphabet

    Available loss modes: 'angular_mae' or 'mae', both are applied to angles

- alphabet_vectors

    As in alphabet_angles but first the network predicts 2 values per angle and then atan2 is applied as in regression vectors.

    Available loss modes: 'angular_mae' or 'mae'. Angular mae is applied to angles, mae is applied to vectors.

#### Angularization

Depending on the Prediction mode, angualrization mode might need to be specified.

regression_vectors: has angularization included in its transformations and has no options to choose from.

regression_angles: continuous value predicted by a linear dense layer is piped through either tanh or cos, specified by the ang_mode argument.
