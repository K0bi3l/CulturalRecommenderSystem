from datetime import datetime, timedelta
import math
from collections import Counter

# Optional: for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer


class FuzzyScorer:
    """
    Computes normalized scores (0.0 - 1.0) for event features to feed into a fuzzy system.
    These scores are used to evaluate how well events match user preferences.

    Outputs a dict with keys: price, distance, popularity, interest, start_hour, length.
    All scores are normalized between 0.0 (worst match) and 1.0 (perfect match).
    """

    def __init__(self, user, preferences, text_vectorizer=None, user_text_profile=None):
        """
        Initialize the FuzzyScorer with user data and preferences.

        Args:
            user: User object containing past event history and profile data
            preferences: Preferences object with user preference settings
            text_vectorizer: Optional TfidfVectorizer for text similarity scoring
            user_text_profile: Optional pre-computed user text profile vector
        """
        self.user = user
        self.preferences = preferences
        self.text_vectorizer = text_vectorizer
        self.user_text_profile = user_text_profile

    def normalize(self, value, min_val, max_val):
        """
        Normalize a value to range [0.0, 1.0] given min and max bounds.

        Args:
            value: The value to normalize
            min_val: Minimum possible value (maps to 0.0)
            max_val: Maximum possible value (maps to 1.0)

        Returns:
            Normalized value between 0.0 and 1.0
        """
        if max_val == min_val:
            return 0.5  # Default to middle value if range is zero
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def score_interest(self, event):
        """
        Calculate overall interest score based on multiple factors.

        Combines vector similarity, category interest weight, and historical preference
        with weighted importance (60%, 30%, 10% respectively).

        Args:
            event: Event object to score

        Returns:
            Interest score between 0.0 and 1.0
        """
        base_sim = 1 - self._vector_similarity(event)  # Convert similarity to score (higher is better)
        cat_weight = self.preferences.get_category_interest(event.type)
        history = self._history_boost(event)
        return base_sim * 0.6 + cat_weight * 0.3 + history * 0.1

    def _vector_similarity(self, event):
        """
        Calculate Euclidean distance-based similarity between user profile and event.

        Computes normalized distance in the feature space of price, distance, and popularity.

        Args:
            event: Event object to compare with user profile

        Returns:
            Normalized distance (lower means more similar)
        """
        u = [self.user.mean_price, self.user.mean_distance, self.user.mean_popularity]
        e = event.get_vector()
        # Calculate Euclidean distance between user profile and event vectors
        dist = math.sqrt(sum((ui - ei) ** 2 for ui, ei in zip(u, e)))
        # Calculate maximum possible distance for normalization
        max_dist = math.sqrt(sum((max(ui, ei) or 1) ** 2 for ui, ei in zip(u, e)))
        return self.normalize(dist, 0, max_dist)

    def _history_boost(self, event):
        """
        Calculate boost factor based on how often user has attended similar category events.

        Args:
            event: Event object to calculate boost for

        Returns:
            Normalized boost factor between 0.0 and 1.0
        """
        counts = self.preferences.get_attended_category_counts()
        if not counts:
            return 0.0  # No history data
        max_c = max(counts.values())
        return self.normalize(counts.get(event.type, 0), 0, max_c)

    def score_distance(self, event):
        """
        Score how well event distance matches user's maximum preferred distance.

        Args:
            event: Event object with distance information

        Returns:
            Distance score between 0.0 and 1.0 (1.0 = closest/best)
        """
        d = event.distance
        md = self.preferences.max_distance
        if md <= 0:
            return 0.5  # Default if max_distance not set
        # Inverse scoring: shorter distances get higher scores
        return max(0.0, min(1.0, 1 - d / md))

    def score_start_hour(self, event):
        """
        Score how well event start time matches user's preferred time window.

        Args:
            event: Event object with start_hour information

        Returns:
            Start hour score between 0.0 and 1.0
        """
        # Normalize how close the event start is to the user's preferred time window start
        if not self.preferences.preferred_times:
            return 0.5  # Default if no preferred times set
        pref_start, _ = self.preferences.preferred_times[0]
        # Calculate time difference in hours
        diff = abs((event.start_hour - pref_start).total_seconds()) / 3600.0
        # Assuming user cares within +/-3 hours (adjust this tolerance as needed)
        return max(0.0, min(1.0, 1 - self.normalize(diff, 0, 3)))

    def score_length(self, event):
        """
        Score how well event duration matches user's historical preferences.

        Args:
            event: Event object with event_length information

        Returns:
            Length score between 0.0 and 1.0
        """
        # Calculate average length of past events user has attended
        avg_len = sum(e.event_length for e in self.user.events) / len(
            self.user.events) if self.user.events else event.event_length
        # Score based on how close this event's length is to that average
        return max(0.0, min(1.0, 1 - self.normalize(abs(event.event_length - avg_len), 0, avg_len or 1)))

    def score_description(self, event):
        """
        Score text similarity between event description and user's profile.

        Uses cosine similarity between TF-IDF vectors of text content.

        Args:
            event: Event object with description text

        Returns:
            Text similarity score between 0.0 and 1.0
        """
        if not self.text_vectorizer or self.user_text_profile is None:
            return 0.5  # Default if text analysis not possible

        # Transform event description to vector space
        vec = self.text_vectorizer.transform([event.description]).toarray()[0]

        # Calculate cosine similarity
        num = vec.dot(self.user_text_profile)
        den = math.sqrt(vec.dot(vec)) * math.sqrt(self.user_text_profile.dot(self.user_text_profile))

        # Handle potential division by zero
        return max(0.0, min(1.0, num / den)) if den else 0.5

    def compute_features(self, event):
        """
        Compute all feature scores for an event.

        Args:
            event: Event object to score

        Returns:
            Dictionary of normalized scores for all features
        """
        # Ensure popularity is between 0-1
        pop = max(0.0, min(100.0, event.popularity)) / 100.0

        # Collect all feature scores
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
        """
        Score how well event price fits within user's budget preferences.

        Args:
            event: Event object with price information

        Returns:
            Budget score between 0.0 and 1.0 (1.0 = best value)
        """
        cost = event.price
        # Get budget from preferences, with fallback to global budget
        max_b = self.preferences.budget_for_category.get(
            event.type,
            self.preferences.budget_for_category.get("global", 0)
        )

        # Free events always get top score
        if cost <= 0:
            return 1.0

        # If no budget defined, score is zero
        if max_b <= 0:
            return 0.0

        # Higher score for lower price relative to budget
        return max(0.0, min(1.0, 1 - cost / max_b))


if __name__ == "__main__":
    # Test code can be added here
    pass