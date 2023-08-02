
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import streamlit as st
import qrcode
from PIL import Image
import io
import qrcode
import streamlit as st
import io
from datetime import datetime
import qrcode
import streamlit as st
import io
from datetime import datetime
from PIL import Image
import qrcode
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import qrcode
import matplotlib.pyplot as plt
import streamlit as st
import json
import qrcode
import base64
from io import BytesIO
import qrcode
from PIL import Image
import io
import os




# Get the current working directory (root directory of the Streamlit app)
base_path = os.path.dirname(os.path.abspath(__file__))

# Authenticatication for the Spotify API
client_id = 'ff4d0656dd474a64a0576dab651082de'
client_secret = '1e625821e1a141a886c765f6dc3696ef'
redirect_uri = 'http://localhost:9000'
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Loading


dataset_filename = os.path.join(base_path, 'combined_dataset.csv')



# dataset_filename = 'combined_dataset.csv'
dataset = pd.read_csv(dataset_filename)
dataset = dataset.drop('track_id', axis=1)

# Split the dataset into features (X) and target variable (y)
X = dataset.drop('will_enjoy', axis=1)
y = dataset['will_enjoy']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training the model
model = XGBClassifier()
model.fit(X_scaled, y)

# Load the CSV file

df_1 = os.path.join(base_path, 'final_dataset_with_prices.csv')

df = pd.read_csv(df_1)

# df = pd.read_csv('final_dataset_with_prices.csv')

# Function to get top songs for an artist
def get_top_songs(artist_name):
    top_songs = []

    # Searching for the artist
    results = sp.search(q=artist_name, type='artist', limit=1)
    artist_id = results['artists']['items'][0]['id']

    # Get the top tracks for the artist
    top_tracks = sp.artist_top_tracks(artist_id, country='US')['tracks']

    # Extract the track names
    for track in top_tracks:
        top_songs.append(track['name'])

    return top_songs

# Function to get audio features for top songs
def get_audio_features(top_songs):
    audio_features = []

    # Retrieve audio features for each song
    for song in top_songs:
        results = sp.search(q=song, type='track', limit=1)
        track_id = results['tracks']['items'][0]['id']
        features = sp.audio_features(track_id)[0]

        # Map the audio features to the columns used in the training dataset
        audio_feature = {
            'danceability': features['danceability'],
            'energy': features['energy'],
            'key': features['key'],
            'loudness': features['loudness'],
            'mode': features['mode'],
            'speechiness': features['speechiness'],
            'acousticness': features['acousticness'],
            'instrumentalness': features['instrumentalness'],
            'liveness': features['liveness'],
            'valence': features['valence'],
            'tempo': features['tempo'],
            'time_signature': features['time_signature']
        }
        audio_features.append(audio_feature)

    return pd.DataFrame(audio_features)

# Function to convert normalized differences to percentages
def convert_to_percentage(difference):
    return (1 - difference) * 100

# Function to fetch weather information
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=4c6e2fed33ed85fbc2b0e82cf8b9e821"
    response = requests.get(url)
    data = response.json()

    if data["cod"] == "404":
        return None

    weather = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    climate = ''
    season = ''

    # Get climate information
    climate_info = {
        'hot': ['clear sky', 'few clouds'],
        'cold': ['snow', 'mist'],
        'rainy': ['rain', 'shower rain', 'thunderstorm'],
        'dry': ['dust', 'smoke'],
        'humid': ['overcast clouds', 'scattered clouds', 'broken clouds']
    }
    for key, values in climate_info.items():
        if any(value in weather for value in values):
            climate = key
            break

    # Determine the season based on the current date
    current_date = datetime.now()
    month = current_date.month
    season_info = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }
    for key, months in season_info.items():
        if month in months:
            season = key
            break

    return weather, temperature, humidity, climate, season


# Function to get user information
def get_user_information():
    name = st.text_input("Enter your name:")
    gender = st.radio("Select your gender:", ("Male", "Female", "Other"))
    age = st.number_input("Enter your age:", min_value=1, max_value=100, step=1)
    vegan = st.checkbox("Are you vegan?")
    city_name = st.text_input("Enter the city name for weather and event time information:")
    artist_name = st.text_input("Enter the artist name:")
    return name, gender, age, vegan, city_name, artist_name

# Function to get location coordinates
def get_coordinates(city):
    geolocator = Nominatim(user_agent="timezone_app")
    try:
        location = geolocator.geocode(city, timeout=10)
        return location.latitude, location.longitude
    except GeocoderTimedOut:
        return get_coordinates(city)

# Function to get current time period
def get_current_time_period():
    current_time = datetime.now().time()
    current_time_period = ''
    if current_time < datetime.strptime('06:00', '%H:%M').time():
        current_time_period = 'Early Morning'
    elif current_time < datetime.strptime('10:00', '%H:%M').time():
        current_time_period = 'Morning'
    elif current_time < datetime.strptime('14:00', '%H:%M').time():
        current_time_period = 'Afternoon'
    elif current_time < datetime.strptime('18:00', '%H:%M').time():
        current_time_period = 'Evening'
    elif current_time < datetime.strptime('22:00', '%H:%M').time():
        current_time_period = 'Night'
    else:
        current_time_period = 'Late Night'
    return current_time_period


   

# Dictionary with messages for each feature and similarity level
messages = {
    'danceability': {
        'high': "This suggests that you enjoy upbeat and energetic music. Consider food items that are also high in energy, such as spicy food or foods that are high in sugar.",
        'medium': "This suggests that you have a moderate likeness for upbeat and energetic music. Consider trying out dishes that have a balanced combination of flavors and textures.",
        'low': "While your likeness for danceable music is relatively low, you can still explore food options that provide a pleasant dining experience and complement your music preferences."
    },
    'energy': {
        'high': "This suggests that you are interested in food that is too heavy or filling. You should eat food with high calories to give you that energy!",
        'medium': "This suggests that you have a moderate likeness for energetic music. Consider food options that provide sustained energy, such as whole grains, lean proteins, and healthy fats.",
        'low': "While your likeness for energetic music is relatively low, you can still enjoy a variety of delicious and satisfying food options."
    },
    'key': {
        'high': "This suggests that you enjoy music that is bright and cheerful. Consider food items that are also bright and colorful, such as fruits and vegetables.",
        'medium': "This suggests that you have a moderate likeness for music with bright and cheerful tones. Consider exploring diverse food options that offer a mix of flavors and visual appeal.",
        'low': "While your likeness for music with bright and cheerful tones is relatively low, you can still explore a variety of interesting and flavorful food options."
    },
    'loudness': {
        'high': "This suggests that you enjoy music that is loud and in-your-face. Consider food items that are also bold and flavorful, such as spicy dishes or foods with strong flavors.",
        'medium': "This suggests that you have a moderate likeness for music with a moderate loudness. Consider exploring food options that offer a balanced blend of flavors and aromas.",
        'low': "While your likeness for music with a loud and in-your-face sound is relatively low, you can still enjoy a variety of delicious and satisfying food options."
    },
    'mode': {
        'high': "This suggests a preference for music with a specific modality. Consider exploring food options that have distinct characteristics or are unique in nature.",
        'medium': "This suggests that you have a moderate likeness for music with a specific modality. Consider trying out different cuisines and dishes that offer unique flavor profiles.",
        'low': "While your likeness for music with a specific modality is relatively low, you can still explore a variety of interesting and flavorful food options."
    },
    'speechiness': {
        'high': "This suggests that you enjoy music with prominent vocals. Consider exploring food options that are inspired by different cultures and regions, as they often have rich flavors and unique combinations that can be enjoyed alongside the music.",
        'medium': "This suggests that you have a moderate likeness for music with prominent vocals. Consider exploring a variety of food options that offer diverse flavors and culinary traditions.",
        'low': "While your likeness for music with prominent vocals is relatively low, you can still explore a variety of interesting and flavorful food options."
    },
    'acousticness': {
        'high': "This suggests an appreciation for music with a more organic and natural sound. Consider exploring farm-to-table or organic food options, as well as dishes made with fresh and locally sourced ingredients for a wholesome dining experience.",
        'medium': "This suggests that you have a moderate appreciation for music with a more organic and natural sound. Consider exploring food options that focus on fresh and natural ingredients.",
        'low': "While your appreciation for music with an organic and natural sound is relatively low, you can still explore a variety of interesting and flavorful food options."
    },
    'instrumentalness': {
        'high': "This indicates a liking for music without vocals or with minimal vocals. In terms of food, explore culinary experiences that focus on the art of plating, presentation, and innovative flavor combinations to enhance the dining experience.",
        'medium': "This suggests that you have a moderate likeness for instrumental music. Consider exploring food options that offer unique sensory experiences and culinary craftsmanship.",
        'low': "While your likeness for instrumental music is relatively low, you can still enjoy a variety of delicious and satisfying food options."
    },
    'liveness': {
        'high': "This suggests that you enjoy music that feels more alive and dynamic. Consider food options that offer interactive or live cooking experiences, such as teppanyaki or sushi-making.",
        'medium': "This suggests that you have a moderate likeness for music that feels alive and dynamic. Consider exploring food options that provide interactive or engaging dining experiences.",
        'low': "While your likeness for music that feels alive and dynamic is relatively low, you can still enjoy a variety of delicious and satisfying food options."
    },
    'valence': {
        'high': "This suggests an affinity for music with a positive and uplifting mood. Consider food options that evoke similar emotions, such as dishes with vibrant colors, refreshing flavors, or desserts that bring joy and indulgence.",
        'medium': "This suggests that you have a moderate affinity for music with a positive and uplifting mood. Consider exploring food options that offer a balanced combination of flavors and textures.",
        'low': "While your affinity for music with a positive and uplifting mood is relatively low, you can still enjoy a variety of delicious and satisfying food options."
    },
    'tempo': {
        'high': "This suggests a preference for music with a specific tempo. Consider food options that match the tempo, such as quick bites or snacks for upbeat music, or slow-cooked dishes for slower tempo music.",
        'medium': "This suggests that you have a moderate preference for music with a specific tempo. Consider exploring food options that offer a diverse range of flavors and textures.",
        'low': "While your preference for music with a specific tempo is relatively low, you can still explore a variety of interesting and flavorful food options."
    },
    'time_signature': {
        'high': "This suggests a preference for music with a specific time signature. Consider food options that have distinct characteristics or are unique in nature.",
        'medium': "This suggests that you have a moderate preference for music with a specific time signature. Consider exploring food options that offer diverse culinary traditions and techniques.",
        'low': "While your preference for music with a specific time signature is relatively low, you can still explore a variety of interesting and flavorful food options."
    }
}



# Function to perform food and beverage recommendations
def recommend_food_and_beverage(level, df, current_time, sorted_percentages, season):
    if level == "1":
        st.subheader("Level - 1 Recommendation - General Top Selling Items:")
        st.write("Here are the top selling food items:")
        food_items = df[(df['Stock Description'].str.contains('Food')) & (df['Stock Description'].str.contains('Beverage') == False)].nlargest(10, 'TOTAL_PRICE_INCLDISC')
        food_items = food_items.drop_duplicates(subset='STOCK_DESCRIPTION')
        for index, row in food_items.iterrows():
            stock_description = row['STOCK_DESCRIPTION']
            if '|' in stock_description:
                name = stock_description.split('|')[1].strip()
            else:
                name = stock_description
            st.write(f"Food: {name}, Calories: {row['Calories']}, Price: {row['Price']}")

        st.write("\nHere are the top selling beverage items:")
        beverage_items = df[(df['Stock Description'].str.contains('Beverage')) & (df['Stock Description'].str.contains('Food') == False)].nlargest(10, 'TOTAL_PRICE_INCLDISC')
        beverage_items = beverage_items.drop_duplicates(subset='STOCK_DESCRIPTION')
        for index, row in beverage_items.iterrows():
            stock_description = row['STOCK_DESCRIPTION']
            if '|' in stock_description:
                name = stock_description.split('|')[1].strip()
            else:
                name = stock_description
            st.write(f"Beverages: {name}, Calories: {row['Calories']}, Price: {row['Price']}")

    elif level == "2":
        st.subheader("Recommendation Level - 2: Perfect things to eat and drink based on the vibe!")

        # Determine the matching season and time period
        time_period = ''
        if current_time < datetime.strptime('06:00', '%H:%M').time():
            time_period = 'Early Morning'
        elif current_time < datetime.strptime('10:00', '%H:%M').time():
            time_period = 'Morning'
        elif current_time < datetime.strptime('14:00', '%H:%M').time():
            time_period = 'Afternoon'
        elif current_time < datetime.strptime('18:00', '%H:%M').time():
            time_period = 'Evening'
        elif current_time < datetime.strptime('22:00', '%H:%M').time():
            time_period = 'Night'
        else:
            time_period = 'Late Night'

        # Filter the dataset based on the current season and time period
        matching_items = df[(df['Season'] == season) & (df['Time Period'] == time_period)]

        st.write("\nHere are the perfect food items based on the vibe:")
        food_items = matching_items[matching_items['Stock Description'].str.contains('Food')]
        if food_items.empty:
            st.write("No matching food items found.")
        else:
            for index, row in food_items.iterrows():
                stock_description = row['STOCK_DESCRIPTION']
                if '|' in stock_description:
                    name = stock_description.split('|')[1].strip()
                else:
                    name = stock_description
                st.write(f"Food: {name}, Calories: {row['Calories']}, Price: {row['Price']}")

        st.write("\nHere are the perfect beverage items based on the vibe:")
        beverage_items = matching_items[matching_items['Stock Description'].str.contains('Beverage')]
        if beverage_items.empty:
            st.write("No matching beverage items found.")
        else:
            for index, row in beverage_items.iterrows():
                stock_description = row['STOCK_DESCRIPTION']
                if '|' in stock_description:
                    name = stock_description.split('|')[1].strip()
                else:
                    name = stock_description
                st.write(f"Beverages: {name}, Calories: {row['Calories']}, Price: {row['Price']}")

    elif level == "3":
        st.subheader("Level - 3 Recommendation - Distinct Feature Values:")

        # Create dictionaries for feature values
        feature_values = {}

        # Check if each feature is present in the sorted percentages and add corresponding values
        if "energy" in sorted_percentages.index:
            energy_values = df[df["Danceability/Energy"] == 1]
            feature_values["Energy"] = energy_values[["STOCK_DESCRIPTION", "Calories", "Price"]].to_dict(orient="records")

        if "liveness" in sorted_percentages.index:
            liveness_values = df[df["Liveness"] == 1]
            feature_values["Liveness"] = liveness_values[["STOCK_DESCRIPTION", "Calories", "Price"]].to_dict(orient="records")

        if "danceability" in sorted_percentages.index:
            danceability_values = df[df["Danceability/Energy"] == 1]
            feature_values["Danceability"] = danceability_values[["STOCK_DESCRIPTION", "Calories", "Price"]].to_dict(orient="records")

        if "loudness" in sorted_percentages.index:
            loudness_values = df[df["Loudness"] == 1]
            feature_values["Loudness"] = loudness_values[["STOCK_DESCRIPTION", "Calories", "Price"]].to_dict(orient="records")

        # Print the distinct feature values with separators
        for feature, values in feature_values.items():
            if values:  # Check if values are present for the feature
                st.subheader(f"\nRecommendations based on {feature} column:")
                for item in values:
                    st.write(f"Food/Beverage: {item['STOCK_DESCRIPTION']}, Calories: {item['Calories']}, Price: {item['Price']}")

    else:
        st.error("Invalid input! Please enter either '1', '2', or '3' for the recommendation level.")






# Generate the QR code for general information (user info, weather info, and music preferences)
def generate_general_info_qr(name, gender, age, vegan, weather_data, sorted_percentages, artist_name):
    # Combine user information and music preferences into a single string
    qr_data = f"User Information:\nName: {name}\nGender: {gender}\nAge: {age}\nVegan: {vegan}\n\n"
    qr_data += "Music Preferences:\n"
    for feature, percentage in sorted_percentages.items():
        qr_data += f"Match {percentage:.2f}% with the {feature} of {artist_name}.\n"
    qr_data += "\n"

    # Add weather information to the QR data
    weather, temperature, humidity, climate, season = weather_data
    qr_data += "Weather Information:\n"
    qr_data += f"Weather in {name}: {weather}\n"
    qr_data += f"Temperature: {temperature:.2f} K\n"
    qr_data += f"Humidity: {humidity}%\n"
    qr_data += f"Climate: {climate}\n"
    qr_data += f"Season: {season}\n\n"

    # Generate the QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Convert the PIL image to a matplotlib figure
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")  # Hide axes to remove extra whitespace
    plt.title("QR Code - General Information")
    plt.tight_layout()

    # Display the QR code in Streamlit using st.pyplot()
    st.subheader("QR Code for General Information:")
    st.pyplot(plt)















# Streamlit app
def main():
    st.title("Music and Food Recommendation App")

    # Vegan Food and Beverages Menu
    vegan_menu = {
            "Vegan Burger": {
            "Type": "Food",
            "Average Calories": 350,
            "Average Price": 8.50,
        },
        "Vegan Tacos": {
            "Type": "Food",
            "Average Calories": 280,
            "Average Price": 7.50,
        },
        "Quinoa Salad": {
            "Type": "Food",
            "Average Calories": 220,
            "Average Price": 6.50,
        },
        "Vegan Sushi Rolls": {
            "Type": "Food",
            "Average Calories": 180,
            "Average Price": 9.00,
        },
        "Vegan Pizza": {
            "Type": "Food",
            "Average Calories": 320,
            "Average Price": 10.50,
        },
        "Vegan Spring Rolls": {
            "Type": "Food",
            "Average Calories": 150,
            "Average Price": 5.00,
        },
        "Fresh Fruit Smoothies": {
            "Type": "Beverage",
            "Average Calories": 180,
            "Average Price": 6.00,
        },
        "Coconut Water": {
            "Type": "Beverage",
            "Average Calories": 45,
            "Average Price": 3.50,
        },
        "Vegan Iced Coffee": {
            "Type": "Beverage",
            "Average Calories": 90,
            "Average Price": 4.00,
        },
        "Hibiscus Iced Tea": {
            "Type": "Beverage",
            "Average Calories": 60,
            "Average Price": 3.00,
        },
        "Sparkling Fruit Infused Water": {
            "Type": "Beverage",
            "Average Calories": 15,
            "Average Price": 2.50,
        },
        "Freshly Squeezed Lemonade": {
            "Type": "Beverage",
            "Average Calories": 70,
            "Average Price": 4.50,
        },
    }
    # Wrap the entire app in a try-except block
    try:
        # Get user information
        name, gender, age, vegan, city_name, artist_name = get_user_information()

        # Define user information string
        user_info_str = f"Name: {name}\nGender: {gender}\nAge: {age}\nVegan: {vegan}"

        # Get the top songs for the artist
        top_songs = get_top_songs(artist_name)

        # Get the audio features for top songs
        audio_features_df = get_audio_features(top_songs)

        # Calculate the artist's average audio features
        user_features = audio_features_df.mean()

        # Calculate the users average audio features from the dataset
        artist_top_avg_audio_features = dataset.mean()

        # Calculate the normalized differences
        normalized_differences = (user_features - artist_top_avg_audio_features).abs()

        # Convert normalized differences to percentages
        percentages = normalized_differences.apply(convert_to_percentage)

        # Sort the percentages in descending order
        sorted_percentages = percentages.sort_values(ascending=False).head(4)

        # Display the top 4 matching features
        st.subheader("Music Preferences:")
        for feature, percentage in sorted_percentages[:4].items():
            st.write(f"Match {percentage:.2f}% with the {feature} of {artist_name}.")
            if feature in messages:
                if percentage >= 95:
                    st.info(messages[feature]['high'])
                elif percentage >= 50:
                    st.info(messages[feature]['medium'])
                else:
                    st.info(messages[feature]['low'])

        # Get weather information
        weather_data = get_weather(city_name)

        if weather_data:
            weather, temperature, humidity, climate, season = weather_data

            # Display weather, climate, and season information
            st.subheader("Weather Information:")
            st.write(f"Weather in {city_name}: {weather}")
            st.write(f"Temperature: {temperature:.2f} K")
            st.write(f"Humidity: {humidity}%")
            st.write(f"Climate: {climate}")
            st.write(f"Season: {season}")

            # Display user information
            st.subheader("User Information:")
            st.write(f"Name: {name}")
            st.write(f"Gender: {gender}")
            st.write(f"Age: {age}")
            st.write(f"Vegan: {vegan}")


            

            # Check if the user is vegan
            if vegan == True:
                # Display the vegan food and beverages menu
                st.subheader("Vegan Food and Beverages Menu:")
                for item, info in vegan_menu.items():
                    st.write(f"{item}")
                    st.write(f"Type: {info['Type']}")
                    st.write(f"Average Calories: {info['Average Calories']}")
                    st.write(f"Average Price: ${info['Average Price']}")
                    st.write("")
            else:


                # Perform food and beverage recommendations
                level_prompt = st.radio("Enter the level of recommendation you want:", ("1", "2", "3"))

                # Initialize the latest_level_output variable
                latest_level_output = None

                if level_prompt in ["1", "2", "3"]:
                    recommendations = recommend_food_and_beverage(level_prompt, df, datetime.now().time(), sorted_percentages, season)



                    # Check if recommendations is not None
                    if recommendations is not None:
                        latest_level_output = recommendations

                # Display Level 1, Level 2, or Level 3 Recommendations based on user's selection
                if level_prompt == "1" and latest_level_output:



                    st.subheader("Level 1 Food and Beverage Recommendations:")
                    level1_text = ""
                    for food_item in latest_level_output['level1_food']:
                        level1_text += f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}\n"

                    level1_text += "\nHere are the top selling beverage items:\n"
                    for beverage_item in latest_level_output['level1_beverages']:
                        level1_text += f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}\n"
                    


                    st.subheader("Level - 1 Recommendation - General Top Selling Items:")
                    for food_item in latest_level_output['level1_food']:
                        st.write(f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}")

                             # Generate the QR code for level-based outputs
                    

                    st.write("Here are the top selling beverage items:")
                    for beverage_item in latest_level_output['level1_beverages']:
                        st.write(f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}")
                
                    
                      


                elif level_prompt == "2" and latest_level_output:


                    st.subheader("Level 2 Food and Beverage Recommendations:")
                    level2_text = ""
                    for food_item in latest_level_output['level2_food']:
                        level2_text += f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}\n"

                    level2_text += "\nHere are the perfect beverage items based on the vibe:\n"
                    for beverage_item in latest_level_output['level2_beverages']:
                        level2_text += f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}\n"


                    st.subheader("Level - 2 Recommendation : Perfect things to eat and drink based on the vibe!")
                    for food_item in latest_level_output['level2_food']:
                        st.write(f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}")

                    for beverage_item in latest_level_output['level2_beverages']:
                        st.write(f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}")


                    

                     

                elif level_prompt == "3" and latest_level_output:

                    st.subheader("Level 3 Food and Beverage Recommendations:")
                    level3_text = ""
                    for food_item in latest_level_output['level3_food']:
                        level3_text += f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}\n"

                    level3_text += "\nHere are the perfect beverage items based on the vibe:\n"
                    for beverage_item in latest_level_output['level3_beverages']:
                        level3_text += f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}\n"



                    st.subheader("Level - 3 Recommendation : Advanced Personalized Recommendations!")
                    for food_item in latest_level_output['level3_food']:
                        st.write(f"Food: {food_item['Food']}, Calories: {food_item['Calories']}, Price: {food_item['Price']}")

                    for beverage_item in latest_level_output['level3_beverages']:
                        st.write(f"Beverages: {beverage_item['Beverages']}, Calories: {beverage_item['Calories']}, Price: {beverage_item['Price']}")
   




                
                generate_general_info_qr(name, gender, age, vegan, weather_data, sorted_percentages, artist_name)

                        



   
    except: 
        pass

if __name__ == "__main__":
    main()