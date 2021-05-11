class ARTBlackBoxRuleBasedMIA():
    
    def __init__(self, target_classifier, x_train, y_train, x_test, y_test):
        
        self.target_classifier = target_classifier
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def infer(self):
        
        print(f"[ART Black-Box-Rule-Based MIA] Preparing MIA attack.")
        
        # The rule based attack uses a simple rule to determine the membership status of data
        next_attack = MembershipInferenceBlackBoxRuleBased(self.target_classifier)
        
        # Ideally, this would return 1 for each record
        inferred_train_data = next_attack.infer(
            self.x_train,
            self.y_train
        )
        # Ideally, this would return 0 for each record
        inferred_test_data = next_attack.infer(
            self.x_test,
            self.y_test
        )
        
        print(f"[ART Black-Box-Rule-Based MIA] Attack completed, printing results...")
        
        evaluate_art_attack(inferred_train_data, inferred_test_data, f"ART Black-Box-Rule-Based MIA")