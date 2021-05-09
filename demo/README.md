# Image Classification on CIFAR10
This is an implemention of doing classification on CIFAR10 dataset which supports manually making the dataset unbalanced. Now you can easily control the sample size of each class!

## How it works
### Data Preparation
CIFAR10 consists of 60000 (50000 training + 10000 testing) images of shape 32*32. The original dataset is perfectly balanced, in other words, eachof the 10 classes has 5000 training samples and 1000 test samples. 

To make the dataset unbalanced we first partition the dataset into 10 disjoint subsets, each representing a class. Then we randomly select from the training subset for a  certain class if we want this class to be under-represented. Notice that test set stays unchanged all the time.

### Model
Here we apply the pre-trained `ResNet` model provided by `torchvision`. To adapt this model to our dataset, we freeze all but the last layer, and modify the last layer so that the output size is identical with the numbers of classes we want.


## Install
```bash
git clone git@github.com:yzchyx/privacy-evaluator.git
git checkout feat/28-cifar-train
cd ./privacy-evaluator/demo
```
All related files for the implementation are organized in the directory `privacy-evaluator/demo/`  and are totally independent from other files in `privacy-evaluator`. Therefore, this directory is portable and you can only unzip this `demo` directory and play with it without much trouble.

## Example
To set sample sizes for each class, you just need to modify the `size_dicts`in `main.py`, which looks like 
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

Finally, run under the `demo` directory

```bash
python main.py
```

## Some results
|Dataset|Test Accuracy|
|:-:|-:|
|{0: 5000, 1: 5000}|85.90%|
|{0: 5000, 1: 4000}|85.85%|
|{0: 5000, 1: 3000}|84.95%|
|{0: 5000, 1: 2000}|83.90%|
|{0: 5000, 1: 1000}|79.80%|
|{0: 5000, 1: 500} |76.10%|
|{0: 2500, 1: 2500}|84.85%|
|{4: 1250, 5: 1250, 6: 1250, 7: 1250}| 60.3%|
|{4: 2000, 5: 1500, 6: 1000, 7: 500}| 57.53%|
|{4: 3000, 5: 1500, 6: 400, 7: 100}| 42.75%|