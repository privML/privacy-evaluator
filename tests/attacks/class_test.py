import pytest

from privacy_evaluator.attacks.sample_attack import SampleAttack

"""
This test only test if no error is thrown when calling the function, can be removed in the future
"""
def test_sample_attack():
    test = SampleAttack(0, 0, 0)
    test.perform_attack(0)
