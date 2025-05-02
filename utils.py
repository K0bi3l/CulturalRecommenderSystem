from db import User
from db import Event
import math


def calculate_distance(user: User, event: Event):
    return math.sqrt(math.pow(user.mean_distance - event.distance, 2) +
                     math.pow(user.mean_price - event.price, 2) +
                     math.pow(user.mean_popularity - event.popularity, 2))
