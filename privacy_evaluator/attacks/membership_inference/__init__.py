"""
Module providing membership inference attacks.
"""
from privacy_evaluator.attacks.membership_inference.black_box import (
    MembershipInferenceBlackBoxAttack,
)
from privacy_evaluator.attacks.membership_inference.black_box_rule_based import (
    MembershipInferenceBlackBoxRuleBasedAttack,
)
from privacy_evaluator.attacks.membership_inference.label_only_decision_boundary import (
    MembershipInferenceLabelOnlyDecisionBoundaryAttack,
)
from privacy_evaluator.attacks.membership_inference.membership_inference import (
    MembershipInferenceAttack,
)
