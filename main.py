import sys
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
import numpy as np
import keras
import string
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Embedding
from keras.models import Sequential, load_model
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import spacy
import random

poem = open('C:\\Users\\fedor\\Desktop\\repositories\\ulgtu\\deeplearn2\\poem.txt').read().lower()
poem = poem.replace('\n', ' ')
half_poem = poem[0:1000000]
print('Corpus full length:', len(poem))
print('Corpus length:', len(half_poem))

print(half_poem[0:200])

spec_chars = string.punctuation
clean_poem = "".join([ch for ch in half_poem if ch not in spec_chars])

print(len(clean_poem))
print(clean_poem)

# Подготавливаем данные для обучения модели генерации текста на уровне символов
# Длина извлеченных последовательностей символов
maxlen = 60  # максимальная длина последовательности

# Мы пробуем новую последовательность символов на каждом `шаге".
step = 3  # размер шага для создания последовательностей

# Здесь хранятся наши извлеченные последовательности
sentences = []  # список для хранения созданных последовательностей

# Здесь содержатся цели (последующие символы)
next_chars = []  # список для хранения следующего символа после каждой последовательности

for i in range(0, len(clean_poem) - maxlen, step):
    sentences.append(clean_poem[i: i + maxlen])
    next_chars.append(clean_poem[i + maxlen])
# количество созданных последовательностей
print('Number of sequences:', len(sentences))

# Список уникальных символов в корпусе
chars = sorted(list(set(clean_poem)))
print('Unique characters:', len(chars))
# Словарь, сопоставляющий уникальные символы с их индексом в `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

# Закодирование символов в двоичные массивы(one-hot encode).
print('Vectorization...')
# Трехмерный массив(количество предложений, maxlen, количество уникальных символов)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
# Двумерный массив с формой (количество предложений, количество уникальных символов).
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

#optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for epoch in range(1, 10):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    # Начальный текст выбирается случайным образом
    start_index = random.randint(0, len(clean_poem) - maxlen - 1)
    generated_text = clean_poem[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.4, 0.6, 1.0, 1.5]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # генерируем 200 символовЯ
        for i in range(200):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
