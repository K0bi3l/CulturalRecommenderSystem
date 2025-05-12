from db import User
from db import Event
import math
import numpy as np


def calculate_euclidean_distance(user: User, event):
    return math.sqrt(math.pow(user.mean_distance - event.distance, 2) +
                     math.pow(user.mean_price - event.price, 2) +
                     math.pow(user.mean_popularity - event.popularity, 2))


def calculate_similarity(event1, event2):
    return 1 - np.linalg.norm(event1 - event2)


def get_similar_users(users: list[User], user: User):
    similar_users = users.copy()
    similar_users.sort(key=lambda db_user: calculate_euclidean_distance(user, db_user))
    return similar_users[:3]

