# privacy-evaluator
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Gitter](https://badges.gitter.im/fairlearn/community.svg)](https://gitter.im/fairlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

The privML Privacy Evaluator is a tool that assesses ML model's levels of privacy by running different attacks on it.

## How does it work?

### Membership inference attack

The aim of a membership inference attack is to find whether a sample given to a trained machine learning model was part of the training data or not. Knowing that the sample was used in the training process can leak personal data, which is an undesirable property in various contexts.

#### Back-box Membership inference attack 

In the black-box setting, an attacker has only access to the target model’s output — the internal structure of the model is not known. Given an arbitrary input, the model produces an output which, in most cases, is a vector of probabilities. The vector of probabilities along with the label for the input is passed to the attack model (a binary classifier) which decides whether the input was part of the training set or not.

This attack makes use of the fact that machine learning models produce different outputs for the data that they’ve been trained on and the data that was never used in the training process. The attack model recognizes these differences and uses them to decide whether a given input is a member of the training set or not.

The current implementation gives the user the choice of the attack model classifier. The default model is a neural network, but a random forest and gradient boosting models can be chosen as well.

![](docs/mia_blackbox.pdf)

#### Black-box rule-based membership inference attack

Just as the case with the block-box membership attack and as the name implies, this attack works in the black-box setting where you don't have the access to the internals of the target model — you can only make observations based on the outputs of the target model for given inputs.

In contrast to the black-box model above which involves training an attack classifier, this attack is much simpler. The attack uses a simple rule: if the target model’s prediction for a given input is correct then that input is considered to be a member of the training set and not a member otherwise.

![](docs/mia_blackbox_rule_based.pdf)

#### Label-Only Inference Attack based on Decision Boundary

This is a black-box membership attack which only requires predicted label as the output of the target model and not a vector of probabilities. This fact makes the attack much more general and applicable to the cases where the attacker does not have an acces to the output probabilites of the black-box model.

![](docs/mia_blackbox_decision_boundary.pdf)

## How to use the Privacy Evaluator?

## How to contribute to the project?
If you want to contribute in any way, please visit our [Contribution Guidelines](https://github.com/privML/privacy-evaluator/CONTRIBUTING.md) to get started.

## Communication

### Who is behind the project?
See a list of our [Supporters]().

### Contact
Gitter: https://gitter.im/privML/community

## License
This library is available under the [MIT License](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt).

## References
