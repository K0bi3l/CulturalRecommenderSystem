from datetime import datetime, timedelta
import math
from collections import Counter


class FuzzyScorer:
    """
    Computes normalized scores (0.0 - 1.0) for different aspects of an event
    based on a User and Preferences model.
    """

    def __init__(self, user, preferences):
        self.user = user
        self.preferences = preferences

    def normalize(self, value, min_val, max_val):
        """
        Normalize value to [0,1] given min and max bounds.
        Returns 0.5 if min_val == max_val to avoid division by zero.
        """
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def score_interest(self, event):
        """
        Combines similarity of (price, distance, popularity) to user's history
        with the user's category preference weight.
        """
        # Compute vector distances
        u_vec = [self.user.mean_price, self.user.mean_distance, self.user.mean_popularity]
        e_vec = event.get_vector()
        # Euclidean distance
        dist = math.sqrt(sum((u - e) ** 2 for u, e in zip(u_vec, e_vec)))
        # Max possible distance in feature space (assuming non-negative features)
        max_dist = math.sqrt(sum((max(u, e) or 1) ** 2 for u, e in zip(u_vec, e_vec)))
        similarity = 1 - self.normalize(dist, 0, max_dist)

        # Category interest weight
        cat_weight = self.preferences.get_category_interest(event.event_type)

        # History boost: more attended in same category => boost
        attended_counts = self.preferences.get_attended_category_counts()
        history_boost = 0.0
        if attended_counts:
            max_count = max(attended_counts.values())
            event_count = attended_counts.get(event.event_type, 0)
            history_boost = self.normalize(event_count, 0, max_count)

        # Combine: 60% similarity, 30% category weight, 10% history boost
        score = (0.6 * similarity + 0.3 * cat_weight + 0.1 * history_boost)
        return score

    def score_proximity(self, event):
        """
        Scores distance: 1.0 at distance 0, 0.0 at max_distance or beyond.
        """
        dist = event.distance
        max_d = self.preferences.max_distance
        if max_d <= 0:
            return 0.5
        return max(0.0, min(1.0, 1 - dist / max_d))

    def score_time(self, event):
        """
        Computes overlap ratio of event time with preferred times.
        """
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
        """
        Scores budget alignment: free events = 1.0;
        if within budget, higher for cheaper events.
        """
        cost = event.price
        max_budget = self.preferences.budget_for_category.get(event.event_type,
                                                              self.preferences.budget_for_category.get("global", 0))
        if cost <= 0:
            return 1.0
        if max_budget <= 0:
            return 0.0
        if cost <= max_budget:
            return max(0.0, min(1.0, 1 - cost / max_budget))
        # over budget penalizes to zero at 2x budget
        over_ratio = (cost - max_budget) / max_budget
        return max(0.0, min(1.0, 1 - over_ratio))

    def compute_scores(self, event):
        """
        Compute all component scores and a final aggregated score.
        Returns a dict with keys: 'interest', 'proximity', 'time', 'budget', 'final'.
        """
        i = self.score_interest(event)
        p = self.score_proximity(event)
        t = self.score_time(event)
        b = self.score_budget(event)
        # Weighted final: interest 0.4, proximity 0.2, time 0.2, budget 0.2
        final = 0.4 * i + 0.2 * p + 0.2 * t + 0.2 * b
        return {"interest": i, "proximity": p, "time": t, "budget": b, "final": final}
