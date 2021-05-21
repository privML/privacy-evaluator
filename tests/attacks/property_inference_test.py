import privacy_evaluator.attacks.property_inference_attack_skeleton as pia
#import torch
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import privacy_evaluator.models.torch.dcti


model = dcti()
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

pia_instance = pia.PropertyInferenceAttackSkeleton(0,0,0)

result = pia_instance.create_meta_training_set([classifier], [classifier])

