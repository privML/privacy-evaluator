class ARTLabelOnlyDecisionBoundaryMIA():
    
    def __init__(self, target_classifier, x_train, y_train, x_test, y_test, distance_threshold):
        
        self.target_classifier = target_classifier
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.distance_threshold = distance_threshold
        
    def infer(self):
        
        print(f"[ART Label-Only-Decision-Boundary MIA] Preparing MIA attack.")
        
        # Create the label-only MIA
        next_attack = LabelOnlyDecisionBoundary(self.target_classifier, self.distance_threshold)
        
        # Calibrate the distance threshold tau to maximize the inference accuracy
        # Currently disabled since I need to read more on this attack to know what classifier to use to calibrate the threshold
        """
        next_attack.calibrate_distance_threshold(
            self.target_classifier,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        )
        """
        
        # Proceed with the actual inference
        
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
        
        print(f"[ART Label-Only-Decision-Boundary MIA] Attack completed, printing results...")
        
        evaluate_art_attack(inferred_train_data, inferred_test_data, f"ART Label-Only-Decision-Boundary MIA")