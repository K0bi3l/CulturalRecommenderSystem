import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

price_range = np.arange(0, 1000, 0.1)
distance_range = np.arange(0, 1000, 1)
popularity_range = np.arange(0, 100, 1)

price_range = ctrl.Antecedent(price_range, 'price_range')
distance_range = ctrl.Antecedent(distance_range, 'distance_range')
popularity_range = ctrl.Antecedent(popularity_range, 'popularity_range')

price_range['low'] = fuzz.trimf(price_range.universe, [0, 0, 100])
price_range['medium'] = fuzz.trimf(price_range.universe, [80, 150, 500])
price_range['high'] = fuzz.sigmf(price_range.universe,500,0.02)

distance_range['near'] = fuzz.trimf(distance_range.universe, [0, 0, 50])
distance_range['medium'] = fuzz.dsigmf(distance_range.universe,75,0.1,150,0.1)
distance_range['far'] = fuzz.sigmf(distance_range.universe,200,0.02)

popularity_range['low'] = fuzz.trimf(popularity_range.universe, [0, 0, 50])
popularity_range['medium'] = fuzz.trimf(popularity_range.universe, [0, 50, 100])
popularity_range['high'] = fuzz.trimf(popularity_range.universe, [50, 100, 100])
