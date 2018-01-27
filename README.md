# Simple Binary Connect

*A bare-bones instructional implementation of BinaryConnect.*

A conversation at work spawned a discussion about the difficulties of successfully deploying Deep Neural Networks on embedded and mobile devices. A [very]rough review revealed three broad approaches:
- Training shallower/smaller architectures to reduce parameters.
- Pruning redundant network connections.
- Quantization of weights while preserving accuracy.

The focus of this tutorial is the BinaryConnect paper, which falls in the third category. The paper proposes a training procedure to learn binarized weights (+1, -1) as opposed to full precision weights (32bits or 64bits) with minimal loss in precision.
The content in the notebook organised as follows:

1. Introduction (this README file)
2. Objective
3. Some Theory
4. Forward Pass
5. Backward Pass
6. Testing
7. Adding Binarization
8. Testing
9. Discussion
