from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math

source = "training_energy"

input_data = pd.read_csv(source + "_input.csv", sep=",")
label_data = pd.read_csv(source + "_label.csv", sep=",")

all_data = pd.concat([input_data, label_data], axis=1)
all_data.columns = ["f"+str(i) for i in range(100)] + ["vowel"]

vowels = ["a", "e", "i", "o", "u"]
data = [all_data[all_data["vowel"] == i] for i in vowels]

x = [float(x)*60 for x in range(1, 101)]

for i in range(5):

    fig = plt.figure()

    [inputs, labels] = np.split(data[i], [100], axis=1)
    
    y = np.mean(inputs.to_numpy(), axis=0)
    y = np.log10(y.tolist())*10

    ax = fig.add_subplot()
    ax.set_ylabel("dB")
    ax.set_xlabel("Hz")
    ax.set_title(vowels[i])
    ax.plot(x, y)

    plt.show()

