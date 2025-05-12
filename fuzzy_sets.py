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
        # Define universes
        self.price_match = ctrl.Antecedent(np.arange(0, 101, 1), 'price_match')
        self.distance_match = ctrl.Antecedent(np.arange(0, 101, 1), 'distance_match')
        self.popularity_range = ctrl.Antecedent(np.arange(0, 101, 1), 'popularity_range')
        self.interest_match = ctrl.Antecedent(np.arange(0, 101, 1), 'interest_match')
        self.start_hour_match = ctrl.Antecedent(np.arange(0, 101, 1), 'start_hour_match')
        self.length_match = ctrl.Antecedent(np.arange(0, 101, 1), 'length_match')
        self.recommendation_match = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation_match')

        self.create_sets()
        self.create_rules()

        # Create control system
        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def create_sets(self):
        # Define membership functions
        self.price_match['low'] = fuzz.trimf(self.price_match.universe, [0, 0, 40])
        self.price_match['medium'] = fuzz.trimf(self.price_match.universe, [20, 50, 80])
        self.price_match['high'] = fuzz.trimf(self.price_match.universe, [60, 100, 100])

        self.distance_match['low'] = fuzz.trimf(self.distance_match.universe, [0, 0, 40])
        self.distance_match['medium'] = fuzz.trimf(self.distance_match.universe, [20, 50, 80])
        self.distance_match['high'] = fuzz.trimf(self.distance_match.universe, [60, 100, 100])

        self.popularity_range['low'] = fuzz.trimf(self.popularity_range.universe, [0, 0, 40])
        self.popularity_range['medium'] = fuzz.trimf(self.popularity_range.universe, [20, 50, 80])
        self.popularity_range['high'] = fuzz.trimf(self.popularity_range.universe, [60, 100, 100])

        self.interest_match['low'] = fuzz.trimf(self.interest_match.universe, [0, 0, 40])
        self.interest_match['medium'] = fuzz.trimf(self.interest_match.universe, [20, 50, 80])
        self.interest_match['high'] = fuzz.trimf(self.interest_match.universe, [60, 100, 100])

        self.start_hour_match['low'] = fuzz.trimf(self.start_hour_match.universe, [0, 0, 40])
        self.start_hour_match['medium'] = fuzz.trimf(self.start_hour_match.universe, [20, 50, 80])
        self.start_hour_match['high'] = fuzz.trimf(self.start_hour_match.universe, [60, 100, 100])

        self.length_match['low'] = fuzz.trimf(self.length_match.universe, [0, 0, 40])
        self.length_match['medium'] = fuzz.trimf(self.length_match.universe, [20, 50, 80])
        self.length_match['high'] = fuzz.trimf(self.length_match.universe, [60, 100, 100])

        # Define the output membership functions
        self.recommendation_match['low'] = fuzz.trimf(self.recommendation_match.universe, [0, 0, 40])
        self.recommendation_match['medium'] = fuzz.trimf(self.recommendation_match.universe, [20, 50, 80])
        self.recommendation_match['high'] = fuzz.trimf(self.recommendation_match.universe, [60, 100, 100])

    def create_rules(self):
        # Create fuzzy rules
        self.rules = [
            # High recommendation rules
            ctrl.Rule(
                (self.price_match['high'] & self.distance_match['high'] & self.interest_match['high']),
                self.recommendation_match['high']
            ),
            ctrl.Rule(
                (self.price_match['high'] & self.popularity_range['high'] & self.interest_match['high']),
                self.recommendation_match['high']
            ),
            ctrl.Rule(
                (self.interest_match['high'] & self.start_hour_match['high'] & self.length_match['high']),
                self.recommendation_match['high']
            ),

            # Medium recommendation rules
            ctrl.Rule(
                (self.price_match['medium'] | self.distance_match['medium']) & self.interest_match['medium'],
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                self.interest_match['high'] & (self.price_match['low'] | self.distance_match['low']),
                self.recommendation_match['medium']
            ),
            ctrl.Rule(
                (self.popularity_range['medium'] | self.length_match['medium']) & self.interest_match['medium'],
                self.recommendation_match['medium']
            ),

            # Low recommendation rules
            ctrl.Rule(
                self.interest_match['low'] & self.price_match['low'],
                self.recommendation_match['low']
            ),
            ctrl.Rule(
                self.interest_match['low'] & self.distance_match['low'],
                self.recommendation_match['low']
            ),
            # Add a "catch-all" rule for when no other rules fire
            ctrl.Rule(
                ~self.interest_match['high'] & ~self.interest_match['medium'],
                self.recommendation_match['low']
            )
        ]

    def makeRecommendation(self, price, distance, popularity, interest, start_hour, length):
        try:
            # Ensure values are within range
            self.simulator.input['price_match'] = min(100, max(0, price))
            self.simulator.input['distance_match'] = min(100, max(0, distance))
            self.simulator.input['popularity_range'] = min(100, max(0, popularity))
            self.simulator.input['interest_match'] = min(100, max(0, interest))
            self.simulator.input['start_hour_match'] = min(100, max(0, start_hour))
            self.simulator.input['length_match'] = min(100, max(0, length))

            # Compute the result
            self.simulator.compute()

            # Check if recommendation_match is in the output
            if 'recommendation_match' not in self.simulator.output:
                print("Warning: No rules were activated. Using default score of 50.")
                return 50.0

            return self.simulator.output['recommendation_match']
        except Exception as e:
            print(f"Error in fuzzy computation: {e}")
            # Return a default score on error
            return 50.0


past = [
    Event("Concert A", "music", 40, 5, 80, "A vibrant musical night with upbeat vibes", 3,
          datetime.now().replace(hour=19)),
    Event("Tech Talk", "tech", 60, 10, 70, "An insightful session on the latest in AI", 1.5,
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
    Event("AI Meetup", "tech", 120, 8, 85, "Discuss AI trends and machine learning insights", 2,
          datetime.now().replace(hour=18))
]

# Print user and prefs\    print("User Profile:")
print(f"  Mean Price: {user.mean_price:.2f}")
print(f"  Mean Distance: {user.mean_distance:.2f}")
print(f"  Mean Popularity: {user.mean_popularity:.2f}\n")
print("User Text Profile Descriptions:")
for desc in user.text_profile_descriptions:
    print(f"  - {desc}")
print()

print("User Preferences:")
print(f"  Max Distance: {prefs.max_distance}")
print(f"  Categories: {prefs.categories}")
print(f"  Preferred Times: {prefs.preferred_times}")
print(f"  Budget for Category: {prefs.budget_for_category}\n")

scorer = FuzzyScorer.FuzzyScorer(user, prefs, text_vectorizer=vectorizer, user_text_profile=user_text_profile)
for evt in new_events:
    print(f"Event: {evt.name}")
    print(f"  Description: {evt.description}")
    # Event features omitted for brevity
    scores = scorer.compute_features(evt)
    print(scores)

    final_score = FuzzySystem().makeRecommendation(
        price=scores['price'] * 100,
        distance=scores["distance"] * 100,
        popularity=scores["popularity"] * 100,
        interest=scores["interest"] * 100,
        start_hour=scores["start_hour"] * 100,
        length=scores["length"] * 100,
    )
    print(final_score)
    print()
