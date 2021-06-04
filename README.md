# Privacy Evaluator
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Gitter](https://badges.gitter.im/fairlearn/community.svg)](https://gitter.im/fairlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


The *privML Privacy Evaluator* is a tool that assesses a ML model's levels of privacy by running different privacy attacks on it. The tool builds upon the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and aims at extending it by implementing additional privacy attacks as well as providing easy-to-use Jupyter Notebooks which offer understandable output metrics even for non-ML experts.

Developers and anyone else with a PyTorch or TensorFlow neural network model at hand can use `privacy-evaluator` to evaluate their model's susceptibility to *model inversion attacks*, *membership inference attacks* and *property inference attacks*. Details on how to use this tool can be found in this README. 

## Dependencies

- [Python 3.7](https://www.python.org/)
- [Adversarial Robustness Toolbox (ART) 1.6.1](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [PyTorch](https://pytorch.org/)
- [Tensorflow v2](https://www.tensorflow.org/)
- for detailed list see [requirements.txt](requirements.txt) 

## Installation
- tbd

## How to use the Privacy Evaluator?
- tbd


### Example
- tbd


## Attacks

### Property Inference Attack
![plot](docs/Property_Interference_Attacks.png)

The Property Inference Attack aims to detect patterns in the parameters of the target model including properties 
which the model producer has not intended to reveal. Therefore, adversaries require a 
meta-classifier which predicts one of these properties they are interested in. 

The first step of the attack is to generate a set containing k datasets on which k shadow classifiers are trained. These shadow classifiers then constitute the training datset on which a meta classifier is trained. The k datasets should be drawn from a pool of data that is as similar as possible to the original dataset on which the target model was trained. If the original dataset is known, the shadow datasets could be created by sampling from that dataset. Furthermore, it is crucial that one half of the shadow classifiers 
is trained with a dataset including the property P and one half including not P (¬P). In addition, 
each shadow classifier comprises the same architecture as the target model where each is only fitted by its 
corresponding dataset. They do not need to perform as good as the target model, but demonstrate passably acceptable 
performance.
Because the parameters in neural networks are usually randomly initialized and even after training, the order of 
parameters is arbitrary. Thus, the meta classifier must be trained on this so that it is able to recognize pattern between 
the parameters of the shadow classifiers. As a result, all the parameters of each shadow classifier represent the 
feature representation F<sub>k</sub>. The feature representation form together the training set for the meta classifer 
where each is labeled correspondingly as either P or ¬P. The training algorithm for the meta-classifier can 
be arbitrarily chosen.

### Membership Inference Attack


## Getting Involved
If you want to contribute in any way, please visit our [Contribution Guidelines](./CONTRIBUTING.md) to get started. Please have also a look at our [Code of Conduct](./CODE_OF_CONDUCT.md). 

### Contact
Gitter: https://gitter.im/privML/community
You can join our [Gitter](https://gitter.im/privML/community) communication channel.

### Who is behind the project?
See a list of our [Supporters](./SUPPORTER_LIST.md).


## License
This library is available under the [MIT License](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt).
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-which is the correct link?

## References
Ganju, Karan, Qi Wang, Wei Yang, Carl A. Gunter, und Nikita Borisov. „Property Inference Attacks on Fully Connected Neural Networks Using Permutation Invariant Representations“. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, 619–33. Toronto Canada: ACM, 2018. https://doi.org/10.1145/3243734.3243834.

