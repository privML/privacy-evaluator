#import privacy_evaluator.attacks.property_inference_attack_skeleton as pia
from privacy_evaluator.attacks.property_inference_attack_skeleton import (PropertyInferenceAttackSkeleton,)

#import torch
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
from privacy_evaluator.models.torch import dcti
#import privacy_evaluator.models.torch.dcti


model = dcti()
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

pia_instance = PropertyInferenceAttackSkeleton(0,0,0)

result = pia_instance.create_meta_training_set([classifier], [classifier])

