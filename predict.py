from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

DR = load_model("DR.h5")

def prediction(file):
    img = file.resize((224,224)).convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    
    classes = DR.predict(img_data)
    probability = classes[0][0]  # Get the probability score
    
    result = int(probability > 0.5)  # 0.5 threshold for binary classification

    if result == 0:
        return f"It's severe "
    else:
        return f"Not severe "
