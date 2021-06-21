import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from typing import Tuple


class UserOutput:

    """User Output Class"""

    def __init__(
        self,
        attack_data_y: np.ndarray,
        privacy_risk: np.ndarray,
        all_labels: np.ndarray,
    ):
        """
        Initilaizes the Class with values
        :param attack_data_y: An Array of the labels of the attck data
        :param privacy_risk: the Privacy risk corresponding to the attack data
        :param all_labels: All labels that are in the training set
        """
        self.attack_data_y = attack_data_y
        self.privacy_risk = privacy_risk
        self.all_labels = all_labels

    def histogram_top_k(
        self, k: int = 10, show_diagram: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw histogram of class distribution of the k points with highest privacy risk score
        :param k: the number of points to consider, default 10
        :param show_diagram: determines if the diagram should be shown, default: True
        :return: All lables of the data with the number of points that are in the top k
        """
        sorting = np.argsort(self.privacy_risk)
        sorting = np.flip(sorting)
        sorted_attack_data_labels = self.attack_data_y[sorting][:k]
        df = pd.DataFrame(
            {
                "labels": sorted_attack_data_labels,
            }
        )
        df = df.groupby("labels")["labels"].agg(["count"])
        labels = df.index.to_numpy()
        count = df["count"].to_numpy()
        all_counts = np.array([])
        for label in self.all_labels:
            index = np.where(labels == label)
            if len(index[0]) == 0:
                all_counts = np.append(all_counts, 0)
            else:
                all_counts = np.append(all_counts, count[index[0]])
        if show_diagram:
            plt.bar(self.all_labels, all_counts)
            plt.title("histogram")
            plt.show()
        return self.all_labels, all_counts
