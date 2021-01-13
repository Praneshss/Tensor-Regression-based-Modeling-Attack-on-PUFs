# Tensor-Regression-based-Modeling-Attack-on-PUFs
A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR Arbiter PUF and its Variants (ECP-TRN)

## Introduction
This repository contains the Tensorflow implementation of the IEEE TCAD paper [A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR Arbiter PUF and its Variants](https://ieeexplore.ieee.org/abstract/document/9233262). Please follow the instructions below for usage.


## Folder Structure
The folder structures should be the same as following
```
ECP_TRN
├── out/
├── dataset/
├── ecp_trn_xor7.py
├── TRL.py 
```




## Requirements
The modeling experiments are carried out with 

Python 2.7

Tensorflow 1.12


## Usage
The code needs to be polished and parameterized.  :( 

The current repository code performs TRN based modeling attack on 64-bit 7-XOR Arbiter PUF. To run the code,
```bash

python ecp_trn_xor7.py

```

Important hyperparameters are:
1. Rank :
Replace the your rank value with 1000
```
run("./out/log22.txt", 1000		, 1)
```

2. Batch size
set the batch size value to variable ```batch_size```



We list the required changes for the code to run on different XORs
```
1. df1 - parity features (input features)

   df2 - PUF response  (target label)

2. set the size of y1 to x for modeling x-XOR APUF

```

## Citation
 Please cite any of our below paper :
 
 ```text
@article{santikellurtensorattack,
   author={P. {Santikellur} and R. S. {Chakraborty}},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR Arbiter PUF and its Variants}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCAD.2020.3032624}
}

@inproceedings{santikellur2019computationally,
  title={A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR APUF},
  author={Santikellur, Pranesh and Prakash, Shashi Ranjan and Chakraborty, Rajat Subhra and others},
  booktitle={2019 Asian Hardware Oriented Security and Trust Symposium (AsianHOST)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```



