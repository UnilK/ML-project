from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math

source = "training_amplitude"

input_data = pd.read_csv(source + "_input.csv", sep=",")
label_data = pd.read_csv(source + "_label.csv", sep=",")

all_data = pd.concat([input_data, label_data], axis=1)
all_data.columns = ["f"+str(i) for i in range(100)] + ["vowel"]

vowels = ["a", "e", "i", "o", "u"]
data = [all_data[all_data["vowel"] == i] for i in vowels]

print(data)

