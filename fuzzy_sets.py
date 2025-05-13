import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import FuzzyScorer
from db import Event, User, Preferences
from datetime import datetime, timedelta

# Optional: for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer


class FuzzySystem:
    def __init__(self):
        self.price_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'price_match')
        self.distance_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'distance_match')
        self.popularity_range = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'popularity_range')
        self.interest_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'interest_match')
        self.start_hour_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'start_hour_match')
        self.length_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'length_match')
        self.recommendation_match = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'recommendation_match')

        self.create_sets()
        self.create_rules()

        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def create_sets(self):
        self.price_match['low'] = fuzz.trimf(self.price_match.universe, [0, 0, 0.4])
        self.price_match['medium'] = fuzz.trimf(self.price_match.universe, [0.2, 0.5, 0.8])
        self.price_match['high'] = fuzz.trimf(self.price_match.universe, [0.6, 1, 1])

        self.distance_match['low'] = fuzz.trimf(self.distance_match.universe, [0, 0, 0.4])
        self.distance_match['medium'] = fuzz.trimf(self.distance_match.universe, [0.2, 0.5, 0.8])
        self.distance_match['high'] = fuzz.trimf(self.distance_match.universe, [0.6, 1, 1])

        self.popularity_range['low'] = fuzz.trimf(self.popularity_range.universe, [0, 0, 0.4])
        self.popularity_range['medium'] = fuzz.trimf(self.popularity_range.universe, [0.2, 0.5, 0.8])
        self.popularity_range['high'] = fuzz.trimf(self.popularity_range.universe, [0.6, 1, 1])

        self.interest_match['low'] = fuzz.trimf(self.interest_match.universe, [0, 0, 0.4])
        self.interest_match['medium'] = fuzz.trimf(self.interest_match.universe, [0.2, 0.5, 0.8])
        self.interest_match['high'] = fuzz.trimf(self.interest_match.universe, [0.6, 1, 1])

        self.start_hour_match['low'] = fuzz.trimf(self.start_hour_match.universe, [0, 0, 0.4])
        self.start_hour_match['medium'] = fuzz.trimf(self.start_hour_match.universe, [0.2, 0.5, 0.8])
        self.start_hour_match['high'] = fuzz.trimf(self.start_hour_match.universe, [0.6, 1, 1])

        self.length_match['low'] = fuzz.trimf(self.length_match.universe, [0, 0, 0.4])
        self.length_match['medium'] = fuzz.trimf(self.length_match.universe, [0.2, 0.5, 0.8])
        self.length_match['high'] = fuzz.trimf(self.length_match.universe, [0.6, 1, 1])

        self.recommendation_match['low'] = fuzz.trimf(self.recommendation_match.universe, [0, 0.2, 0.4])
        self.recommendation_match['medium'] = fuzz.trimf(self.recommendation_match.universe, [0.3, 0.5, 0.8])
        self.recommendation_match['high'] = fuzz.trimf(self.recommendation_match.universe, [0.7, 1, 1])

    def create_rules(self):
        self.rules = [
            ctrl.Rule(
                self.price_match['high'] & self.distance_match['high'] & self.popularity_range['high'] &
                self.interest_match['high'] & self.start_hour_match['high'] & self.length_match['high'],
                self.recommendation_match['high']
            ),
            ctrl.Rule(
                self.price_match['high'] & self.distance_match['high'] & self.popularity_range['medium'] &
                self.interest_match['high'] & self.start_hour_match['high'] & self.length_match['high'],
                self.recommendation_match['high']
            ),
            ctrl.Rule(
                self.price_match['high'] & self.distance_match['medium'] & self.popularity_range['high'] &
                self.interest_match['high'] & self.start_hour_match['high'] & self.length_match['high'],
                self.recommendation_match['high']
            ),
            ctrl.Rule(
                self.price_match['high'] & self.distance_match['high'] & self.popularity_range['high'] &
                self.interest_match['high'] & self.start_hour_match['medium'] & self.length_match['high'],
                self.recommendation_match['high']
            ),

            ctrl.Rule(
                self.price_match['medium'] & self.distance_match['medium'] & self.interest_match['medium'],
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                self.start_hour_match['high'] & self.length_match['medium'] & self.popularity_range['medium'],
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                self.price_match['high'] & self.popularity_range['high'] & self.interest_match['medium'],
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                self.interest_match['high'] & self.length_match['medium'] & self.popularity_range['low'] &
                self.price_match['medium'], self.recommendation_match['medium']
            ),

            ctrl.Rule(
                self.price_match['low'] & self.distance_match['low'] & self.popularity_range['low'] &
                self.interest_match['low'] & self.start_hour_match['low'] & self.length_match['low'],
                self.recommendation_match['low']
            ),
            ctrl.Rule(
                self.price_match['low'] & self.distance_match['medium'] & self.popularity_range['low'] &
                self.interest_match['low'] & self.start_hour_match['low'] & self.length_match['low'],
                self.recommendation_match['low']
            ),
            ctrl.Rule(
                self.price_match['medium'] | self.distance_match['medium'] | self.interest_match['medium'],
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                self.price_match['low'] | self.distance_match['low'] | self.interest_match['low'],
                self.recommendation_match['low']),
        ]

    def makeRecommendation(self, price, distance, popularity, interest, start_hour, length):
        self.simulator.input['price_match'] = price
        self.simulator.input['distance_match'] = distance
        self.simulator.input['popularity_range'] = popularity
        self.simulator.input['interest_match'] = interest
        self.simulator.input['start_hour_match'] = start_hour
        self.simulator.input['length_match'] = length

        self.simulator.compute()

        output = self.simulator.output['recommendation_match']
        return self.getRecommendationLabel(output), output * 100

    def getRecommendationLabel(self, output):
        low_membership = fuzz.interp_membership(
            self.recommendation_match.universe,
            self.recommendation_match['low'].mf,
            output
        )

        medium_membership = fuzz.interp_membership(
            self.recommendation_match.universe,
            self.recommendation_match['medium'].mf,
            output
        )

        high_membership = fuzz.interp_membership(
            self.recommendation_match.universe,
            self.recommendation_match['high'].mf,
            output
        )

        memberships = {
            'low': low_membership,
            'medium': medium_membership,
            'high': high_membership
        }

        return max(memberships, key=memberships.get)


past = [
    Event("Concert A", "music", 40, 5, 80, "A vibrant musical night with upbeat vibes", 3,
          datetime.now().replace(hour=19)),
    Event("Tech Talk", "tech", 60, 10, 70, "An insightful session on the latest in AI", 1.5,
          datetime.now().replace(hour=17))
]
user = User(past, text_profile_descriptions=[e.description for e in past])
preferred_times = [
    (datetime.now().replace(hour=18, minute=0, second=0, microsecond=0), 2),  # 6 PM for 2 hours
    (datetime.now().replace(hour=20, minute=0, second=0, microsecond=0), 1.5)  # 8 PM for 1.5 hours
]

# Define preferences
prefs = Preferences(
    max_distance=10,  # Prefers events within 10 km
    categories={"music": 0.9, "tech": 0.6, "science": 1.0, "jazz": 0.0},  # High interest in music, some in tech
    preferred_times=preferred_times,
    budget=100,  # Budget constraints
)

new_events = [
    Event("Jazz Night", "music", 45, 3, 75, "Smooth jazz evening with mellow tunes", 2,
          datetime.now().replace(hour=18)),
    Event("AI Meetup", "tech", 120, 8, 85, "Discuss AI trends and machine learning insights", 5,
          datetime.now().replace(hour=20)),
    Event("XD event", "standup", 50, 1, 85, "Discuss AI trends and machine learning insights", 5,
          datetime.now().replace(hour=21)),
    Event("Best event", "science", 0, 0, 100, "BEST DESCRIPTION", 1.5,
          datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)),
    Event("Worst event", "jazz", 110, 11, 0, "The worst event possible",
          3, datetime.now().replace(hour=2, minute=0, second=0, microsecond=0))
]


print("User Text Profile Descriptions:")
print()
print("User Preferences:")
print(f"  Max Distance: {prefs.max_distance}")
print(f"  Categories: {prefs.categories}")
print(f"  Preferred Times: {prefs.preferred_times}")
print(f"  Budget for Category: {prefs.budget}\n")

scorer = FuzzyScorer.FuzzyScorer(user, prefs)
for evt in new_events:
    print(f"Event: {evt.name}")
    print(f"  Description: {evt.description}")
    # Event features omitted for brevity
    scores = scorer.compute_features(evt)
    print(scores)

    final_score = FuzzySystem().makeRecommendation(
        price=scores['price'],
        distance=scores["distance"],
        popularity=scores["popularity"],
        interest=scores["interest"],
        start_hour=scores["start_hour"],
        length=scores["length"],
    )
    print(final_score)
    print()
