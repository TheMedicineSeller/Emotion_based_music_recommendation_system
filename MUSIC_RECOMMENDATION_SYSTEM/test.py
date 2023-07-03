# # from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np


# img = image.load_img('uploaded_image.png', target_size=(224, 224))
# print(np.shape(img))
# img_array = image.img_to_array(img)
# print(np.shape(img_array))
# # img_array = np.expand_dims(img_array, axis=0)
# print(np.shape(img_array))


import random
import pandas as pd
moosik = pd.read_csv("data_moods.csv")

Play = moosik[moosik['mood'] =='Happy']
Play = Play.sort_values(by="popularity", ascending=False)
Play = Play[:5].reset_index(drop=True)
randidx = random.randint(0, 4)
print(randidx)
temp = Play.iloc[randidx]
print(temp)
