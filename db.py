from collections import Counter

event_types = ["music", "fine dining", "jam session", "painting", "sport", "travel", "stand up", "tech", "prelection"]


class Event:
    def __init__(self, name, event_type, price, distance, popularity, description, event_length, start_hour):
        self.name = name
        self.type = event_type
        self.price = price
        self.distance = distance
        self.popularity = popularity
        self.description = description
        self.event_length = event_length
        self.start_hour = start_hour
        self.score = 0

    def get_vector(self):
        return [self.price, self.distance, self.popularity]


class User:
    def __init__(self, events, text_profile_descriptions):
        self.description = None
        self.mean_popularity = None
        self.mean_distance = None
        self.mean_price = None
        self.events = events
        self.update()
        self.text_profile_descriptions = text_profile_descriptions

    def update(self):
        self.mean_price = sum(e.price for e in self.events) / len(self.events)
        self.mean_distance = sum(e.distance for e in self.events) / len(self.events)
        self.mean_popularity = sum(e.popularity for e in self.events) / len(self.events)

    def append_event(self, event):
        self.events.append(event)
        self.update()

    def update_description(self, description):
        self.description = description


class Preferences:
    def __init__(self,
                 max_distance=50,
                 categories=None,
                 preferred_times=None,
                 budget=None,
                 attended_events=None):
        # Maximum distance in km
        self.max_distance = max_distance
        # List of preferred categories with weights  e.g. {"music": 0.8, "jam session": 0.6}
        self.categories = categories
        # List of (start_time, length)
        self.preferred_times = preferred_times
        # Preferred price
        self.budget = budget
        # List of earlier events
        self.attended_events = attended_events

    def is_in_preferred_time(self, event_start, event_length):
        # Proper overlap check
        from datetime import timedelta
        event_end = event_start + timedelta(hours=event_length)
        for start, length in self.preferred_times:
            pref_end = start + timedelta(hours=length)
            if start < event_end and event_start < pref_end:
                return True
        return False

    def get_attended_category_counts(self):
        """Returns a Counter of attended event categories."""
        return Counter(event.type for event in self.attended_events)

    def add_attended_event(self, event):
        if event not in self.attended_events:
            self.attended_events.append(event)

    def get_category_interest(self, category):
        return self.categories.get(category, 0)

