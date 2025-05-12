from db import Event
import streamlit as st
from collections import namedtuple


# Function to display events in tiles
def display_event_tiles(events):
    st.title("Event List")

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

    for event in events:
        st.markdown(f"""
        <div class="event-tile">
            <h2>{event.name}</h2>
            <p><strong>Type:</strong> {event.event_type}</p>
            <p><strong>Price:</strong> ${event.price}</p>
            <p><strong>Distance:</strong> {event.distance} km</p>
            <p><strong>Popularity:</strong> {event.popularity} / 10</p>
        </div>
        <div class="divider"></div>
        """, unsafe_allow_html=True)


# Sample data
events = [
    Event("Music Concert", "Music", 50, 10, 8),
    Event("Art Gallery", "Art", 20, 5, 6),
    Event("Tech Conference", "Technology", 100, 15, 9),
    Event("Food Festival", "Food", 30, 3, 7),
    Event("Marathon", "Sports", 10, 20, 8),
]


# Main Streamlit app
def main():
    st.set_page_config(page_title="Event Viewer", layout="wide")
    display_event_tiles(events)


if __name__ == "__main__":
    main()

