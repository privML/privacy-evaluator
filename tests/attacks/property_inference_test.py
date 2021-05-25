# import privacy_evaluator.attacks.property_inference_attack_skeleton as pia
from privacy_evaluator.attacks.property_inference_attack_skeleton import (
    PropertyInferenceAttackSkeleton,
)

# import torch
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch
from privacy_evaluator.models.torch import dcti

# import privacy_evaluator.models.torch.dcti


model = dcti()
# model = torch.load("/home/florian/university/SS2021/PrivacyPreservingML/privacy-evaluator/privacy_evaluator/models/torch/torch_fc_class_0_5000_class_1_2000.pth")
# model.eval()

criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

pia_instance = PropertyInferenceAttackSkeleton(0, 0, 0)

features, labels = pia_instance.create_meta_training_set(
    [classifier, classifier, classifier], [classifier]
)

meta_classifier = pia_instance.train_meta_classifier(features, labels)

print(meta_classifier.predict(features[0].reshape(1, -1)))

"""
import os
os.chdir('/home/florian/university/SS2021/PrivacyPreservingML/privacy-evaluator')

model1 = torch.load("/home/florian/university/SS2021/PrivacyPreservingML/privacy-evaluator/privacy_evaluator/models/torch/torch_fc_class_0_5000_class_1_2000.pth")
"""
