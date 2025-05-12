import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzySystem:
    def __init__(self):
        self.price_match = ctrl.Antecedent(np.arange(0, 501, 1), 'price_match')
        self.distance_match = ctrl.Antecedent(np.arange(0, 101, 1), 'distance_match')
        self.popularity_range = ctrl.Antecendent(np.arange(0, 101, 1), 'popularity_range')
        self.interest_match = ctrl.Antecedent(np.arange(0, 101, 1), 'interest_match')
        self.start_hour_match = ctrl.Antecedent(np.arange(0, 101, 1), 'start_hour_match')
        self.length_match = ctrl.Antecedent(np.arange(0, 101, 1), 'length_match')

        self.recommendation_match = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation_match')



