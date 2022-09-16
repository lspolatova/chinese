import codecs
import re
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from os.path import isfile, join
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
class Data:  
    def __init__(self, file_name):
        letters = set()
        with codecs.open(file_name, encoding = 'utf-8') as file:
            for chinese in re.findall(r'[\u3400-\u4BDF\u4E00-\u9FFF\uF900-\uFAFF\u3400-\u4BDF\u4E00-\u9FFF\uF900-\uFAFF]', file.read()):
                letters.add(chinese)
        self.letters = letters
        self.X_data = []
        self.Y_data = []
        self.Y_classes = []
        self.encoder = LabelEncoder()
    def drawings(self, multiplier):
        self.X_data = list(self.X_data)
        for letter in tqdm(multiplier*list(self.letters)): #обеспечивается контроль выполнения
            img = Image.new('RGB', (64, 64), 'white')
            idraw = ImageDraw.Draw(img)
            line = ImageFont.truetype(join('font/', random.choice(os.listdir('font')[1:])), size = random.choice([30,35,40,45]), encoding='utf-8')
            idraw.text((5, 5), letter, font = line, fill = 'black')
            self.X_data.append(np.asarray(img).astype('int8'))
        self.X_data = np.array(self.X_data)
        self.Y_data = self.Y_data + multiplier*list(self.letters)
    def lable(self):
        self.encoder = self.encoder.fit(self.Y_data)
        self.Y_classes = tf.keras.utils.to_categorical(np.array(self.encoder.transform(self.Y_data)),  dtype='int8')
    def hieroglyphs(self, classes):
        return self.encoder.inverse_transform(np.argmax(classes, axis = 1))
        
        