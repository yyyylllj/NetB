# NetB
'net_train.py' is the code to train the CAE.

There are three parameters for you to make a better a training:

T: Step of training set.

BC: Step of training.

XLLC: How many turn need to train.

We have show a set of value of these three parameters in the file, and you do not need to change them, you can just use it directly.

When finish training, we can get the encoder 'mod-lx1-100-1.pt' and decoder 'mod-lx2-100-1.pt'. (We have save them in the 'mod-lx1-100-1.pt' and 'mod-lx2-100-1.pt.', you can use them directly.)

Use 'accuracy on test set.py' to test the accuracy of the CAE.

Use 'drew the number.py' to see the output of decoder.
