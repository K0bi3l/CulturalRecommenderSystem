from collections import Counter
event_types = ["music", "fine dining", "jam session", "painting", "sport", "travel", "stand up", "tech", "prelection"]


class Event:
    def __init__(self, name, event_type, price, distance, popularity, description, event_length, start_hour):
        self.name = name
        self.event_type = event_type
        self.price = price
        self.distance = distance
        self.popularity = popularity
        self.type = event_type
        self.description = description
        self.event_length = event_length
        self.start_hour = start_hour

    def get_vector(self):
        return [self.price, self.distance, self.popularity]


class User:
    def __init__(self, event_list=None):
        self.events = event_list if event_list is not None else []
        if event_list is not None:
            self.update()
            return
        self.mean_price = 0
        self.mean_distance = 0
        self.mean_popularity = 0
        self.types = None,

    def update(self):
        self.mean_price = sum(event.price for event in self.events) / len(self.events)
        self.mean_distance = sum(event.distance for event in self.events) / len(self.events)
        self.mean_popularity = sum(event.popularity for event in self.events) / len(self.events)
        self.types = Counter(event.type for event in self.events).most_common(3)

    def append_event(self,event):
        self.events.append(event)
        self.update()


events = [
    Event('event1', 'music', 100, 10, 20),
    Event('event2', 'sport', 200, 20, 100)
]

users = [
   User([events[0]]),
   User([events[1]]),
]
