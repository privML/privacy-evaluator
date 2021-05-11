class ARTBlackBoxMIA():
    
    def __init__(self, target_classifier, x_train, y_train, x_test, y_test, attack_train_ratio = 0.5):
        
        self.target_classifier = target_classifier
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # The black box attacks trains a classifier 
        # The black box attack is trained using a portion of the training/testing data (defined by the attack_train_ratio) 
        # Then it is evaluated based on the rest portion of the provided training/testing data
        
        self.attack_train_ratio = attack_train_ratio
        self.attack_train_size = int(len(x_train) * attack_train_ratio)
        self.attack_test_size = int(len(x_test) * attack_train_ratio)
        
    def infer(self, attack_model_type):
        
        # attack_model_type can have the following values: nn (neural network, PyTorch), rf (RandomForest) or gb (GradientBoosting)
        
        print(f"[ART Black-Box MIA - {attack_model_type}] Preparing MIA attack.")
        
        # Train black-box attack model using the first portion of training/testing data
        next_attack = MembershipInferenceBlackBox(self.target_classifier, attack_model_type = attack_model_type)
        next_attack.fit(
            self.x_train[:self.attack_train_size],
            self.y_train[:self.attack_train_size],
            self.x_test[:self.attack_test_size],
            self.y_test[:self.attack_test_size]
        )
        
        # Infer membership status using the rest portion of the training/testing data
        # 1 -> attack model predicts that the record was used in training the classifier
        # 0 -> attack model predicts that the record was not used in training the classifier
        
        # Ideally, this would return 1 for each record
        inferred_train_data = next_attack.infer(
            self.x_train[self.attack_train_size:],
            self.y_train[self.attack_train_size:]
        )
        # Ideally, this would return 0 for each record
        inferred_test_data = next_attack.infer(
            self.x_test[self.attack_test_size:],
            self.y_test[self.attack_test_size:]
        )
        
        print(f"[ART Black-Box MIA - {attack_model_type}] Attack completed, printing results...")
        
        evaluate_art_attack(inferred_train_data, inferred_test_data, f"ART Black-Box MIA - {attack_model_type}")