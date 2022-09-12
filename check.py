from matplotlib import pyplot as plt
import random

source = "training"

inputf = open(source + "_input.csv").readlines()
labels = open(source + "_label.csv").readlines()
inputs = [[float(x) for x in y.split(",")] for y in inputf]
indices = [x for x in range(len(inputs))]

samples = random.sample(indices, k=5)
x = [float(x) for x in range(len(inputs[0]))]

for i in samples:
    plt.plot(x, inputs[i])
    plt.title(labels[i])
    plt.show()

