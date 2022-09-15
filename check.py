from matplotlib import pyplot as plt
import random
import math

source = "training_amplitude"

inputf = open(source + "_input.csv").readlines()
labels = open(source + "_label.csv").readlines()
inputs = [[float(x) for x in y.split(",")] for y in inputf]
indices = [x for x in range(len(inputs))]

samples = random.sample(indices, k=5)
x = [float(x)*60 for x in range(1, len(inputs[0])+1)]

for i in samples:
    Y = inputs[i]
    plt.plot(x, Y)
    plt.title(labels[i])
    plt.show()

