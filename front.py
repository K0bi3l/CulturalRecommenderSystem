from db import Event
import streamlit as st
from collections import namedtuple
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import FuzzyScorer
from fuzzy_sets import FuzzySystem
from db import Preferences, User

# Sample data for events and preferences
past = [
    Event("Concert A", "music", 40, 5, 80, "A vibrant musical night with upbeat vibes", 3,
          datetime.datetime.now().replace(hour=19)),
    Event("Tech Talk", "tech", 60, 10, 70, "An insightful session on the latest in AI", 1.5,
          datetime.datetime.now().replace(hour=17))
]

user = User(past)

preferred_times = [
    (datetime.datetime.now().replace(hour=18, minute=0, second=0, microsecond=0), 2),  # 6 PM for 2 hours
    (datetime.datetime.now().replace(hour=20, minute=0, second=0, microsecond=0), 1.5)  # 8 PM for 1.5 hours
]

# Define preferences
prefs = Preferences(
    max_distance=10,  # Prefers events within 10 km
    categories={"music": 0.9, "tech": 0.6, "science": 1.0},  # High interest in music, some in tech
    preferred_times=preferred_times,
    budget=100,  # Budget constraints
)


new_events = [
    Event("Jazz Night", "music", 45, 3, 75, "Smooth jazz evening with mellow tunes", 2,
          datetime.datetime.now().replace(hour=18)),
    Event("AI Meetup", "tech", 120, 8, 85, "Discuss AI trends and machine learning insights", 5,
          datetime.datetime.now().replace(hour=20)),
    Event("Funny event", "standup", 50, 1, 85, "Discuss AI trends and machine learning insights", 5,
          datetime.datetime.now().replace(hour=21)),
    Event("Best event", "science", 0, 0, 100, "Very attractive!", 1.5,
          datetime.datetime.now().replace(hour=20, minute=0, second=0, microsecond=0))
]


# Function to display events in tiles
def display_event_tiles(events):
    st.title("Event List")

    # Display user preferences
    st.subheader("User Preferences")
    st.markdown("### Categories of Interest")
    for category, score in prefs.categories.items():
        st.markdown(f"- **{category.capitalize()}**: {score:.2f}")

    st.markdown("### Time Preferences")
    for start_time, duration in prefs.preferred_times:
        st.markdown(f"- **From** {start_time.strftime('%H:%M')} **for** {duration}h")

    st.markdown(f"### Budget: **${prefs.budget}**")
    st.markdown(f"### Max Distance: **{prefs.max_distance} km**")

    st.markdown(
        """
        <style>
        .event-tile {
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            transition: box-shadow 0.3s ease;
        }
        .event-tile:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .divider {
            border-top: 1px solid #eee;
            margin: 16px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Scorer setup
    scorer = FuzzyScorer.FuzzyScorer(user, prefs)

    for event in events:
        st.markdown(f"""
        <div class="event-tile">
            <h2>{event.name}</h2>
            <p><strong>Type:</strong> {event.type}</p>
            <p><strong>Price:</strong> ${event.price}</p>
            <p><strong>Distance:</strong> {event.distance} km</p>
            <p><strong>Popularity:</strong> {event.popularity}</p>
            <p><strong>Description:</strong> {event.description}</p>
            <p><strong>Length:</strong> {event.event_length}</p>
            <p><strong>Start hour:</strong> {event.start_hour.strftime("%H:%M")}</p>
        </div>
        <div class="divider"></div>
        """, unsafe_allow_html=True)

        # Compute features and score
        scores = scorer.compute_features(event)
        final_score, percent = FuzzySystem().makeRecommendation(
            price=scores['price'],
            distance=scores["distance"],
            popularity=scores["popularity"],
            interest=scores["interest"],
            start_hour=scores["start_hour"],
            length=scores["length"],
        )

        st.markdown(f"**Final Score**: {final_score}, percent match: {percent:.2f}%")



# Sample data
events = [
    Event("Music Concert", "music", 50, 10, 8, "An exciting concert!", 2.5, datetime.datetime.now().replace(hour=19)),
    Event("Art Gallery", "art", 20, 5, 6, "An evening of beautiful art!", 2, datetime.datetime.now().replace(hour=18)),
    Event("Tech Conference", "tech", 100, 15, 9, "A cutting-edge technology conference!", 6,
          datetime.datetime.now().replace(hour=20)),
    Event("Food Festival", "food", 30, 3, 7, "Taste the world's best cuisines!", 3,
          datetime.datetime.now().replace(hour=21)),
]


# Main Streamlit app
def main():
    st.set_page_config(page_title="Event Viewer", layout="wide")
    display_event_tiles(new_events)


if __name__ == "__main__":
    main()
