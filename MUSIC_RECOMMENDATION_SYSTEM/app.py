from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('MODELS/mood_resnet_model.h5', compile=False)

@app.route('/')
def index():
    return render_template('index.html')


import pandas as pd
import random

def recommend_song(predicted_mood):
    moosik = pd.read_csv("data_moods.csv")
    moosik = moosik[['name', 'artist', 'mood', 'popularity', 'album', 'id']]
    RAND_SWITCH_THRESH = 0.15
    p = random.random()
    

    if((predicted_mood=='happy' or predicted_mood=='sad') and p >= RAND_SWITCH_THRESH):
        
        Play = moosik[moosik['mood'] == 'Happy']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        randidx = random.randint(0, 4)
        print(randidx)
        song = Play.iloc[randidx]

        
    if((predicted_mood=='fear' or predicted_mood=='angry') and p >= RAND_SWITCH_THRESH):

        Play = moosik[moosik['mood'] =='Calm']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        randidx = random.randint(0, 4)
        song = Play.iloc[randidx]

    if((predicted_mood=='surprise' or predicted_mood=='neutral') and p >= RAND_SWITCH_THRESH):

        Play = moosik[moosik['mood'] =='Energetic']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        randidx = random.randint(0, 4)
        song = Play.iloc[randidx]

    else:
        Play = moosik[moosik['mood'] =='Sad']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        randidx = random.randint(0, 4)
        song = Play.iloc[randidx]
    
    return song


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image_file = request.files['image']

        # Convert PNG image to JPEG
        img = Image.open(image_file)

        # Convert PNG image to JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img = image.load_img(image_file, target_size=(224, 224))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Get the class label based on your model's classes
        # Replace 'class1', 'class2', etc. with your actual class labels
        class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        predicted_label = class_labels[predicted_class]
        song = recommend_song(predicted_label)
        song_name = song['name']
        song_artist = song['artist']
        song_album = song['album']
        song_id = song['id']

        return render_template('display.html', song_name = song_name, song_album = song_album, song_artist = song_artist, song_id = song_id)
       
    else:
        return 'No image uploaded.'

if __name__ == '__main__':
    app.run()
