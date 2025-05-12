from datetime import datetime, timedelta
import math
from collections import Counter

# Optional: for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

class FuzzyScorer:
    """
    Computes normalized scores (0.0 - 1.0) for event features to feed into a fuzzy system.
    Outputs a dict with keys: price, distance, popularity, interest, start_hour, length.
    """
    def __init__(self, user, preferences, text_vectorizer=None, user_text_profile=None):
        self.user = user
        self.preferences = preferences
        self.text_vectorizer = text_vectorizer
        self.user_text_profile = user_text_profile

    def normalize(self, value, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def score_interest(self, event):
        base_sim = 1 - self._vector_similarity(event)
        cat_weight = self.preferences.get_category_interest(event.type)
        history = self._history_boost(event)
        return base_sim * 0.6 + cat_weight * 0.3 + history * 0.1

    def _vector_similarity(self, event):
        u = [self.user.mean_price, self.user.mean_distance, self.user.mean_popularity]
        e = event.get_vector()
        dist = math.sqrt(sum((ui - ei) ** 2 for ui, ei in zip(u, e)))
        max_dist = math.sqrt(sum((max(ui, ei) or 1) ** 2 for ui, ei in zip(u, e)))
        return self.normalize(dist, 0, max_dist)

    def _history_boost(self, event):
        counts = self.preferences.get_attended_category_counts()
        if not counts:
            return 0.0
        max_c = max(counts.values())
        return self.normalize(counts.get(event.type, 0), 0, max_c)

    def score_distance(self, event):
        d = event.distance
        md = self.preferences.max_distance
        if md <= 0:
            return 0.5
        return max(0.0, min(1.0, 1 - d / md))

    def score_start_hour(self, event):
        # Normalize how close the event start is to the user's preferred time window start
        if not self.preferences.preferred_times:
            return 0.5
        pref_start, _ = self.preferences.preferred_times[0]
        diff = abs((event.start_hour - pref_start).total_seconds()) / 3600.0
        # Assuming user cares within +/-3 hours
        return max(0.0, min(1.0, 1 - self.normalize(diff, 0, 3)))

    def score_length(self, event):
        # Normalize event length relative to average past event length
        avg_len = sum(e.event_length for e in self.user.events) / len(self.user.events) if self.user.events else event.event_length
        return max(0.0, min(1.0, 1 - self.normalize(abs(event.event_length - avg_len), 0, avg_len or 1)))

    def score_description(self, event):
        if not self.text_vectorizer or self.user_text_profile is None:
            return 0.5
        vec = self.text_vectorizer.transform([event.description]).toarray()[0]
        num = vec.dot(self.user_text_profile)
        den = math.sqrt(vec.dot(vec)) * math.sqrt(self.user_text_profile.dot(self.user_text_profile))
        return max(0.0, min(1.0, num / den)) if den else 0.5

    def compute_features(self, event):
        # Raw popularity expected between 0-100
        pop = max(0.0, min(100.0, event.popularity)) / 100.0
        features = {
            "price": self.score_budget(event),
            "distance": self.score_distance(event),
            "popularity": pop,
            "interest": self.score_interest(event),
            "start_hour": self.score_start_hour(event),
            "length": self.score_length(event)
        }
        return features

    def score_budget(self, event):
        cost = event.price
        max_b = self.preferences.budget_for_category.get(event.type,
                                                          self.preferences.budget_for_category.get("global", 0))
        if cost <= 0:
            return 1.0
        if max_b <= 0:
            return 0.0
        return max(0.0, min(1.0, 1 - cost / max_b))

if __name__ == "__main__":
    # Instantiate and test by passing the compute_features output to your fuzzy system
    pass
