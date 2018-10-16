import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('file')
arg = parser.parse_args()
fig = pickle.load(open(arg.file,'rb'))
fig.show()


# How to extract data from a figure
# for axe in fig.axes:
#     for line ine axe.lines:
#        print(line.get_data())