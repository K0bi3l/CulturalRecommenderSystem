from datetime import datetime, timedelta
import math
from collections import Counter
from db import User, Preferences, Event
# Optional: for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer


class FuzzyScorer:
    """
    Computes normalized scores (0.0 - 1.0) for different aspects of an event
    based on a User and Preferences model, including optional text similarity.
    """

    def __init__(self, user, preferences, text_vectorizer=None, user_text_profile=None):
        self.user = user
        self.preferences = preferences
        self.text_vectorizer = text_vectorizer
        # Expect user_text_profile as 1D numpy array
        self.user_text_profile = user_text_profile

    def normalize(self, value, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def score_interest(self, event):
        u_vec = [self.user.mean_price, self.user.mean_distance, self.user.mean_popularity]
        e_vec = event.get_vector()
        dist = math.sqrt(sum((u - e) ** 2 for u, e in zip(u_vec, e_vec)))
        max_dist = math.sqrt(sum((max(u, e) or 1) ** 2 for u, e in zip(u_vec, e_vec)))
        similarity = 1 - self.normalize(dist, 0, max_dist)
        cat_weight = self.preferences.get_category_interest(event.type)
        attended = self.preferences.get_attended_category_counts()
        history_boost = 0.0
        if attended:
            max_count = max(attended.values())
            history_boost = self.normalize(attended.get(event.type, 0), 0, max_count)
        return 0.6 * similarity + 0.3 * cat_weight + 0.1 * history_boost

    def score_proximity(self, event):
        dist = event.distance
        max_d = self.preferences.max_distance
        if max_d <= 0:
            return 0.5
        return max(0.0, min(1.0, 1 - dist / max_d))

    def score_time(self, event):
        start = event.start_hour
        end = start + timedelta(hours=event.event_length)
        prefs = [(s, s + timedelta(hours=l)) for s, l in self.preferences.preferred_times]
        if not prefs:
            return 0.5
        overlap = 0.0
        for p_start, p_end in prefs:
            latest_start = max(start, p_start)
            earliest_end = min(end, p_end)
            if earliest_end > latest_start:
                overlap += (earliest_end - latest_start).total_seconds()
        duration = (end - start).total_seconds()
        if duration <= 0:
            return 0.0
        return max(0.0, min(1.0, overlap / duration))

    def score_budget(self, event):
        cost = event.price
        max_budget = self.preferences.budget_for_category.get(
            event.type, self.preferences.budget_for_category.get("global", 0)
        )
        if cost <= 0:
            return 1.0
        if max_budget <= 0:
            return 0.0
        if cost <= max_budget:
            return max(0.0, min(1.0, 1 - cost / max_budget))
        over_ratio = (cost - max_budget) / max_budget
        return max(0.0, min(1.0, 1 - over_ratio))

    def score_description(self, event):
        """
        Computes text similarity between the event.description and the user's text profile.
        Expects both descriptions and profile as dense numpy arrays.
        """
        if not self.text_vectorizer or self.user_text_profile is None:
            return 0.5
        desc_vec = self.text_vectorizer.transform([event.description]).toarray()[0]
        # Cosine similarity: dot / (||a|| * ||b||)
        num = desc_vec.dot(self.user_text_profile)
        den = math.sqrt(desc_vec.dot(desc_vec)) * math.sqrt(self.user_text_profile.dot(self.user_text_profile))
        if den == 0:
            return 0.5
        return max(0.0, min(1.0, num / den))

    def compute_scores(self, event):
        i = self.score_interest(event)
        p = self.score_proximity(event)
        t = self.score_time(event)
        b = self.score_budget(event)
        d = self.score_description(event)
        final = 0.35 * i + 0.2 * p + 0.2 * t + 0.15 * b + 0.1 * d
        return {"interest": i, "proximity": p, "time": t, "budget": b, "description": d, "final": final}


if __name__ == "__main__":
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

    scorer = FuzzyScorer(user, prefs, text_vectorizer=vectorizer, user_text_profile=user_text_profile)
    for evt in new_events:
        print(f"Event: {evt.name}")
        print(f"  Description: {evt.description}")
        # Event features omitted for brevity
        scores = scorer.compute_scores(evt)
        for k, v in scores.items():
            print(f"  {k}: {v:.2f}")
        print()
