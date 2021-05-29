import numpy as np
from matplotlib import pyplot as plt 

class user_output:
    def __init__(self, attack_data_x, attack_data_y, privacy_risk):
        self.attack_data_x = attack_data_x
        self.attack_data_y = attack_data_y
        self.privacy_risk = privacy_risk

    def histogram_top_k(self, k=10):
        sorting = np.argsort(self.privacy_risk)
        sorting = np.flip(sorting)
        sorted_attack_data_x = self.attack_data_x[sorting][:k]
        sorted_attack_data_y = self.attack_data_y[sorting][:k]
        sorted_privacy_risk = self.privacy_risk[sorting][:k]
        print(sorted_attack_data_x, sorted_attack_data_y, sorted_privacy_risk)
        return np.histogram(sorted_attack_data_x, weights=sorted_attack_data_y)


data_x = np.array([1,2,3,4,5,6,7,8,9])
data_y = np.array([1,2,3,4,5,6,7,8,9])
priv_risk = np.array([1,2,3,4,5,6,7,8,9])
user_output = user_output(data_x, data_y, priv_risk)
hist= user_output.histogram_top_k(4)
plt.hist(hist) 
plt.title("histogram") 
plt.show()

