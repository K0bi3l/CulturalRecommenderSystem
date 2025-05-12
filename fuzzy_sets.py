import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from FuzzyScorer import FuzzyScorer
from db import User, Preferences, Event
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer


class FuzzySystem:
    def __init__(self):
        self.price_match = ctrl.Antecedent(np.arange(0, 101, 1), 'price_match')
        self.distance_match = ctrl.Antecedent(np.arange(0, 101, 1), 'distance_match')
        self.popularity_range = ctrl.Antecedent(np.arange(0, 101, 1), 'popularity_range')
        self.interest_match = ctrl.Antecedent(np.arange(0, 101, 1), 'interest_match')
        self.start_hour_match = ctrl.Antecedent(np.arange(0, 101, 1), 'start_hour_match')
        self.length_match = ctrl.Antecedent(np.arange(0, 101, 1), 'length_match')
        self.recommendation_match = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation_match')

        self.create_sets()
        self.create_rules()

        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def create_sets(self):
        self.price_match['low'] = fuzz.trimf(self.price_match.universe, [0, 0, 40])
        self.price_match['medium'] = fuzz.trimf(self.price_match.universe, [20, 50, 80])
        self.price_match['high'] = fuzz.trimf(self.price_match.universe, [65, 100, 100])

        self.distance_match['low'] = fuzz.trimf(self.distance_match.universe, [0, 0, 40])
        self.distance_match['medium'] = fuzz.trimf(self.distance_match.universe, [20, 50, 80])
        self.distance_match['high'] = fuzz.trimf(self.distance_match.universe, [65, 100, 100])

        self.popularity_range['low'] = fuzz.trimf(self.popularity_range.universe, [0, 0, 40])
        self.popularity_range['medium'] = fuzz.trimf(self.popularity_range.universe, [20, 50, 80])
        self.popularity_range['high'] = fuzz.trimf(self.popularity_range.universe, [65, 100, 100])

        self.interest_match['low'] = fuzz.trimf(self.interest_match.universe, [0, 0, 40])
        self.interest_match['medium'] = fuzz.trimf(self.interest_match.universe, [20, 50, 80])
        self.interest_match['high'] = fuzz.trimf(self.interest_match.universe, [65, 100, 100])

        self.start_hour_match['low'] = fuzz.trimf(self.start_hour_match.universe, [0, 0, 40])
        self.start_hour_match['medium'] = fuzz.trimf(self.start_hour_match.universe, [20, 50, 80])
        self.start_hour_match['high'] = fuzz.trimf(self.start_hour_match.universe, [65, 100, 100])

        self.length_match['low'] = fuzz.trimf(self.length_match.universe, [0, 0, 40])
        self.length_match['medium'] = fuzz.trimf(self.length_match.universe, [20, 50, 80])
        self.length_match['high'] = fuzz.trimf(self.length_match.universe, [70, 100, 100])

        self.recommendation_match['low'] = fuzz.trimf(self.recommendation_match.universe, [0, 10, 40])
        self.recommendation_match['medium'] = fuzz.trimf(self.recommendation_match.universe, [20, 50, 80])
        self.recommendation_match['high'] = fuzz.trimf(self.recommendation_match.universe, [70, 90, 100])

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
        return self.getRecommendationLabel(output)

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
    Event("Tech Talk", "tech", 0, 0, 1000, "An insightful session on the latest in AI", 1,
          datetime.now().replace(hour=17))
]
user = User(past, text_profile_descriptions=[e.description for e in past])
prefs = Preferences()
prefs.attended_events.append(past[0])

# Text vectorizer & profile
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(user.text_profile_descriptions)
user_text_profile = tfidf_matrix.mean(axis=0).A1  # 1D numpy array

new_events = [
    Event("Jazz Night", "music", 45, 3, 75, "Smooth jazz evening with mellow tunes", 2,
          datetime.now().replace(hour=18)),
    Event("AI Meetup", "tech", 0, 0, 100, "Discuss AI trends and machine learning insights", 1,
          datetime.now().replace(hour=18))
]


scorer = FuzzyScorer(user, prefs, text_vectorizer=vectorizer, user_text_profile=user_text_profile)
for evt in new_events:
    print(f"Event: {evt.name}")
    print(f"  Description: {evt.description}")
    # Event features omitted for brevity
    scores = scorer.compute_features(evt)
    print(scores)

    final_score = FuzzySystem().makeRecommendation(
        price=100,
        distance=100,
        popularity=50,
        interest=100,
        start_hour=90,
        length=100,
    )
    print(f"  Final Recommendation Score: {final_score}")
    print()
