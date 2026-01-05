import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
import keras_nlp

data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({"values": data})

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(1,)),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

tokenizer = keras_nlp.tokenizers.WhitespaceTokenizer()
tokens = tokenizer(["hello world", "simple keras nlp"])
