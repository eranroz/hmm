hmm_kit
===

Simple Hidden Markov Models library.

Supports discrete/continuous emissions of one/multiple dimensions.

## Install
python setup.py install

## Usage
See examples in tests/hmm_test.py

## Standalone script
To run it as a standalone script call simple_hmm.py
* Training model `simple_hmm.py train_model train_data.csv --kernel gaussian --states 3 --model_name my_model`
* Decode - `simple_hmm.py decode train_data.csv my_model decode.csv`
* Posterior - `simple_hmm.py posterior train_data.csv my_model posterior.csv`




