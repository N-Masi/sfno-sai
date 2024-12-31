### *Climate Modeling with Spherical Fourier Neural Operators for Stratospheric Aerosol Injections*

This repo is the accompanying code for *Climate Modeling with Spherical Fourier Neural Operators for Stratospheric Aerosol Injections* by Nick Masi and Mason Lee. The preprint can be viewed [here](https://drive.google.com/file/d/15NX22cAoqskW0SgYecXcCNb-o6FB3LTr/view?usp=sharing).

Use ```data/scripts/data_serializer.py``` to preprocess, normalize, and download the data. Then ```src/sfno_train_val.py``` can be run to train the SNFO, with ```src/sfno_test_rollout.py``` being subsequently used to reproduce the autoregressive climate forecasting of §3. ```src/sfno_mim.py``` pretrains the SFNO encoder on masked image modeling as in §4.1, and ```src/sfno_train_val_finetune.py``` transfers this to the forecasting task as in §4.2.

```requirements.txt``` is provided for reproducibility to create a python environment with the same dependencies.
