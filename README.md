# Knowledge Injection in DL Models for Transprecision Computing

The injection of domain knowledge in deep learning models can boost their
performance, especially in the case of complicated functions to be learned and
scarcity of training data. For instance, in the domain of transprecision
computing an open issue is learning the relationship between the precision
assigned to the Floating-Point (FP) variables composing a benchmark and the
associated error (measured as the difference w.r.t. the execution of the
benchmark with all FP variable at maximum precision). DL models can be used to
learn this relationships, albeit it is a very complex function, non-linear,
non-monotonic and with plenty of local minima. For this reason, injecting domain
knowledge in these DL models can increase their performance.

This repository contains the source code of the experiments conducted to
demonstrate the benefits of injecting domain knowledge in DL model for
transprecision computing. 
The experimental analysis has been described in detail in the paper "Injective
Domain Knowledge in Neural Networks for Transprecision Computing", Borghesi et
al., 2020, presented at LOD2020 (arXiv version:
https://arxiv.org/abs/2002.10214).

For more details on transprecision computing and how to use DL models in that
settings we refer to "Combining Learning and Optimization for Transprecision
Computing", Borghesi et  al., 2020, presented at Computing Frontiers 2020 (arXiv
version: https://arxiv.org/abs/2002.10890).

# Required dependencies

* Python > 3.6
* TensorFlow 1.x
* Keras
* numpy
* scikit-learn
* pandas
* Spektral (https://graphneural.network/getting-started/)


