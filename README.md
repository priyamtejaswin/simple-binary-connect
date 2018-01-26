# Simple Binary Connect
A bare-bones instructional implementation of BinaryConnect. The rest of the post is organised as follows:

0. Introduction
1. Objective
2. Some Theory
3. Forward Pass
4. Backward Pass
5. Testing
6. Adding Binarization
7. Testing
8. Discussion

## 0. Introduction
A conversation at work spawned a discussion about the difficulties of successfully deploying Deep Neural Networks on embedded and mobile devices. A [very]rough review reveals three broad approaches:
- Training shallower/smaller architectures to reduce parameters.
- Pruning redundant network connections.
- Quantization of weights while preserving accuracy.

The focus of this tutorial is the BinaryConnect paper, which falls in the third category. The paper proposes a training procedure to learn binarized weights (+1, -1) as opposed to full precision weights (32bits or 64bits) with minimal loss in precision.

## 1. Objective
We will not be replicating the results of the paper. Rather, this will serve as a proof-of-concept for the ... concept. These are the concession that we will be making for time.
- Working with MNIST and only MNIST.
- Training a VERY simple model (logistic regression).
- Using slow and un-optimised Python.

The last point it actually more important that you might think. The true potential of such approaches is unlocked while using specialised hardware and software which actually exploits the single bit weights. But you won't get to see that with Python because I will still use 32 bit integers to represent +1 and -1. So there!

## 2. Some Theory
For those unfamiliar with back-propagation, there's a excellent post on the topic by [Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html). Here's a <INSERT_FAVORITE_BIG_NUMBER> foot overview using logistic regression model.

