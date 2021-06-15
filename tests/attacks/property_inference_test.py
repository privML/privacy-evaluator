from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.tf.cnn import ConvNet


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader("MNIST")
    input_shape = test_dataset[0][0].shape
    num_elements_per_classes = {0: 500, 1: 500}
    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)

    model = ConvNet(num_classes, input_shape)
    print("Start training target model ...\n")
    trainer(train_set, num_elements_per_classes, model, num_epochs=2, verbose=2)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)
    print("Start attack ...")
    attack = PropertyInferenceAttack(target_model, train_dataset, verbose=1)
    attack.attack()
