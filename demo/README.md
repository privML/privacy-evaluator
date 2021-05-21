## Demo 1: Image Classification on CIFAR10
This is an implemention of doing classification on CIFAR10 dataset which supports manually making the dataset unbalanced. Now you can easily control the sample size of each class! Implementations using PyTorch and Tensorflow are provided, with source code in sub-folder `./train-cifar-10-torch` and `./train-cifar-10-tf` respectively.

### How it works
#### Data Preparation
CIFAR10 consists of 60000 (50000 train + 10000 test) images of shape 32*32. The original dataset is perfectly balanced, in other words, eachof the 10 classes has 5000 training samples and 1000 test samples. 

To make the dataset unbalanced we first partition the dataset into disjoint subsets, each subset representing a class. Then if we want a specific class to be under-represented, we take a random sample from this class. Notice such sampling only applies in the training set.

#### Model
Here we apply the pre-trained `ResNet50`(default, `ResNet18` optional) model provided by `torchvision`/`keras` (for PyTorch/TensorFlow, respectively). To adapt this model to our dataset, we freeze all but the last layers, and customize the last layer so that the output size is compatible with the numbers of classes we want.


### Install
```bash
git clone git@github.com:yzchyx/privacy-evaluator.git
git checkout feat/28-cifar-train
cd ./privacy-evaluator/demo
```
All related files for the implementation are organized in the directory `privacy-evaluator/demo/`  and are totally independent from other files in `privacy-evaluator`. Therefore, this directory is portable and you can only unzip this `demo` directory and play with it without much trouble.

### Example
To set sample sizes for each class, just modify the `size_dicts`in `main.py`, which looks like 
```python
# put your designed sample distribution here
# each line corresponds to an experiment
size_dicts = [
    {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 
     5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000},
    {0: 2500, 1: 2500},
    {0: 4500, 1: 500},
    {2: 4500, 7: 500, 9: 2000},
]
```
- The first experiment corresponds to the training on the original dataset, taking 5000 samples from all the 10 classes;
- The second experiment is about training on a balanced two-class dataset (2500 samples for both `class 0` and `class 1`) , while the third trains model on unbalanced classes (4500 for `class 0` and 500 for `class 1`)
- We can also do 3-class or more-class training in a highly-customizable fashion, as the fourth experiment shows.

Finally, run under the `demo/train-cifar10-torch` directory:

```bash
python main.py
```
For TensorFlow implementation, switch to the other sub-folder:

```bash
cd ../train-cifar10-tf
python main.py
```

### Expriments
#### Experiment Settings
For both PyTorch and TensorFlow, the following hyperparamters are applied: 
```python
num_epochs = 10
batch_size = 500
learning_rate = 0.001
weight_decay = 0.002
dropout = 0.3
```

#### Results
|Dataset|Torch+ResNet50|TF+Resnet50|Torch+FC|TF+FC|
|:-:|-:|-:|-:|-:|
|{0: 5000, 1: 5000}|88.30%|92.40%|84.05%|83.95%|
|{0: 5000, 1: 4000}|87.65%|92.30%|82.00%|83.20%|
|{0: 5000, 1: 3000}|87.70%|92.15%|80.50%|81.65%|
|{0: 5000, 1: 2000}|87.85%|90.95%|80.50%|79.60%|
|{0: 5000, 1: 1000}|85.15%|89.55%|74.85%|76.10%|
|{0: 5000, 1: 500} |83.65%|87.75%|56.45%|62.55%|
|{0: 3000, 1: 1000, 2: 500}|72.83%|79.80%|53.07%|52.50%|

#### Checkpoints
The pre-trained models can be downloaded [here]()