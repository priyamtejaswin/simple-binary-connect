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
We will not be replicating the results of the paper. Rather, this will serve as a proof-of-concept for the ...umm, concept. These are the concession that we will be making in the interest of time.
- Working with MNIST and only MNIST.
- Training a VERY simple model (logistic regression).
- Using slow and un-optimised Python.

The last point it actually more important that you might think. The true potential of such approaches is unlocked while using specialised hardware and software which actually exploits the single bit weights. But you won't get to see that with Python because I will still use 32 bit integers to represent +1 and -1. So there!

## 2. Some Theory
For those unfamiliar with back-propagation, there's a excellent post on the topic by [Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html). Here's a <INSERT_FAVORITE_BIG_NUMBER> foot overview using logistic regression model.

### 2.1. Model for multi-class logistic regression.
I define a multi-class classification problems as follows:
$$
\begin{align}
X & \sim input\ matrix[batch, features] \\
W & \sim weight\ matrix[features, classes] \\
\textbf{b} & \sim bias\ vector[1, classes]
\end{align}
$$

The logistic model $f$ is defined as 
$$
f(X, W, \textbf{b}) = softmax(X.W + \textbf{b})
$$
where
$$
softmax(v_{ij}) = \frac{e^{v_{ij}}}{\sum_k e^{v_{ik}}}
$$

$f$ returns the model output, which in this case will be a probability distribution for every sample against every possible class. We can calculate the error from the correct class label using any number of loss/distance measures. In the spirit of needlessly complicating things, let's continue with [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy). Amongst other things, a cross-entropy as an error measure is helpful as it allows weight updates even when the activations are close to saturation, i.e. when the error gradient is very close to 0.

We define cross entropy loss $XE$ for a target distribution $\textbf{t}$ and predicted distribution $\textbf{p}$ as  follows
$$
XE(\textbf{t}, \textbf{p}) = - \sum_i t_i log(p_i)
$$


