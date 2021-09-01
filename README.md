# NetB

Codes for the paper [arXiv:...]()

## Install

Requirement: 
* python 3.8
* CUDA 11.0

Install all required packages
1. Clone this repo and change the direction to this repo in your terminal.
2. Run `pip install -r Requirements.txt` to install all required packages.

## Usage

`net_train.py` is the code to train the CAE.

There are three parameters for you to make a better a training:

T: Step of training set.

BC: Step of training.

XLLC: How many turn need to train.

We have show a set of value of these three parameters in the file, and you do not need to change them, you can just use it directly.

When finish training, we can get the encoder `mod-lx1-100-1.pt` and decoder `mod-lx2-100-1.pt`. (We have save them in the 'mod-lx1-100-1.pt' and 'mod-lx2-100-1.pt.', you can use them directly.)

Use `Try-CAE.py` to test the accuracy of the CAE on the test set.

Use `Try-LCAE.py` to test the accuracy of the LCAE on the test set.

`Try-LCAE-calculate-probability.py` can calculate the probability of a random samples in the test set.

`CAE-draw the number.py` can draw the input and output of the CAE, if you want to use this code, make sure you run this '.py' file on the pycharm. 

`LCAE-draw the number.py` can draw the input and output of the LCAE, if you want to use this code, make sure you run this '.py' file on the pycharm. 

We have show a set of the output of `CAE-draw the number.py` and `LCAE-draw the number.py ` in the  `CAE-draw.png` and  `LCAE-draw.png`.
