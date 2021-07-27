from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple

from .user_output import UserOutput


class UserOutputPrivacyScore(UserOutput):
    """`UserOutputPrivacyScore` class

    Contains the result of a `MembershipInferencePointAnalysis`.

    Interpretation of Outcome:

    Vulnerability of individual data points:
    The privacy risk score is an individual sample’s likelihood of being a training member, which allows an adversary to
    identify samples with high privacy risks and perform membership inference attacks with high confidence [1].

    The training data points that exhibit an increased membership privacy risk might differ from their classes mean
    samples (outliers) [2]. You could check them again, see if they have the correct label, or if they exhibit any
    non-standard properties for the class. If so, correct them. It was also shown that points with an high influence on
    the decision boundary are more vulnerable to membership inference attacks [3]. Therefore, these points might be
    important. If you want to protect them, you might add several similar training samples as they are to the class.

    References:
    [1] Song, Liwei and Prateek Mittal. “Systematic Evaluation of Privacy Risks of Machine Learning Models.” ArXiv
    abs/2003.10595 (2020): n. pag
    [2] Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, HaixuTang, Carl A. Gunter, and Kai Chen.
    2018.   Understanding Membership In-ferences on Well-Generalized Learning Models.CoRRabs/1802.04889
    (2018).arXiv:1802.04889  http://arxiv.org/abs/1802.0
    [3] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. 2019.Demystifying Membership Inference
    Attacks in Machine Learning as a Service.IEEE Transactions on Services Computing(2019)
    """

    def __init__(self, attack_data_y: np.ndarray, privacy_risk: np.ndarray):
        """Initializes a `UserOutputPrivacyScore` class.

        :param attack_data_y: An Array of the labels of the attack data.
        :param privacy_risk: The Privacy risk corresponding to the attack data.
        """
        self.attack_data_y = attack_data_y
        self.privacy_risk = privacy_risk

    def histogram_top_k(
        self,
        all_labels: np.ndarray,
        k: int = 10,
        label_names: np.ndarray = None,
        show_diagram: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw histogram of class distribution of the k points with highest privacy risk score.

        :param all_labels: All the labels of the input data.
        :param k: The number of points to consider, default 10.
        :param label_names: Labels to display on the x axis of the histogram.
        :param show_diagram: Determines if the diagram should be shown, default: True.
        :return: All labels of the data with the number of points that are in the top k.
        """
        sorting = np.argsort(self.privacy_risk)
        sorting = np.flip(sorting)
        sorted_attack_data_labels = self.attack_data_y[sorting][:k]
        df = pd.DataFrame({"labels": sorted_attack_data_labels})
        df = df.groupby("labels")["labels"].agg(["count"])
        labels = df.index.to_numpy()
        count = df["count"].to_numpy()
        all_counts = np.array([])
        for label in all_labels:
            index = np.where(labels == label)
            if len(index[0]) == 0:
                all_counts = np.append(all_counts, 0)
            else:
                all_counts = np.append(all_counts, count[index[0]])
        if show_diagram:
            plt.bar(all_labels, all_counts)
            plt.title("Histogram for top {} points".format(k))
            plt.yticks(np.arange(0, np.int_(max(all_counts)) + 1, 1))
            if (
                label_names is not None
                and type(label_names) is np.ndarray
                and len(label_names) == len(all_labels)
            ):
                plt.xticks(all_labels, label_names)
            else:
                plt.xticks(all_labels)
            plt.xlabel("Classes")
            plt.ylabel("number of points")
            plt.show()
        return all_labels, all_counts

    def histogram_top_k_relative(
        self,
        all_labels: np.ndarray,
        k: int = 10,
        label_names: np.ndarray = None,
        show_diagram: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw histogram of class distribution of the k points with highest privacy risk score, relative to the size of
         the classes.

        :param all_labels: All the labels of the input data.
        :param k: The number of points to consider, default 10.
        :param label_names: Labels to display on the x axis of the histogram.
        :param show_diagram: Determines if the diagram should be shown, default: True.
        :return: All labels of the data with the number of points that are in the top k, relative to the size of the
            classes.
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
        label_counts = np.array([])
        for label in all_labels:
            label_count = (self.attack_data_y == label).sum()
            label_counts = np.append(label_counts, label_count)
            index = np.where(labels == label)
            if len(index[0]) == 0:
                all_counts = np.append(all_counts, 0)
            else:
                all_counts = np.append(all_counts, count[index[0]])
        relative_values = np.divide(all_counts, label_counts)
        relative_values = np.nan_to_num(
            relative_values, copy=True, nan=0.0, posinf=None, neginf=None
        )
        if show_diagram:
            plt.bar(all_labels, relative_values)
            plt.title("Histogram for top {} points relative per class".format(k))
            plt.xlabel("Classes")
            plt.ylabel("proportion of points in top k")
            if (
                label_names is not None
                and type(label_names) is np.ndarray
                and len(label_names) == len(all_labels)
            ):
                plt.xticks(all_labels, label_names)
            else:
                plt.xticks(all_labels)
            plt.show()
        return all_labels, relative_values

    def histogram_distribution(
        self,
        class_name: str = None,
        show_diagram: bool = True,
    ) -> np.ndarray:
        """Draws a histogram of the privacy risk score of the given data.

        :param class_name: Name of the class shown in the diagram.
        :param show_diagram: Determines if the diagram should be shown, default: True.
        :return: The data that is being plotted.
        """
        if show_diagram:
            plt.hist(self.privacy_risk)
            if class_name is None:
                plt.title("Histogram for the whole data")
            else:
                plt.title("Histogram for data of {}".format(class_name))
            plt.ylabel("Number of occurences")
            plt.xlabel("Privacy risk score")
            plt.show()
        return self.privacy_risk

    def histogram_slices(
        self,
        slices: np.ndarray,
        slices_priv_risk: np.ndarray,
        show_diagram: bool = True,
        name: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plots the classes with their average privacy score.

        :param slices: The names of the slices to be plotted.
        :param slices_priv_risk: A list of the privacy risk scores within the slices.
        :param show_diagram: Determines if the diagram should be shown, default: True.
        :param name: Name of the data set (training/test).
        :return: All Slice names with the average privacy risk score.
        """

        if show_diagram:
            title_name = ""
            if name is not None:
                title_name = "for " + name + " data"
            plt.bar(slices, slices_priv_risk)
            plt.title("Histogram for the average score on each slice " + title_name)
            plt.xlabel("Slices")
            plt.ylabel("Average Privacy Score")
            plt.xticks(slices, rotation=(40), va="top", ha="right")
            plt.show()
        return slices, slices_priv_risk
