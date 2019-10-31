''' make database'''
from random import randrange
import csv 

NUM_POINTS = 10
DEFAULT_NUMERIC_VALUE = 10



#take in a list of dictionaries, with expected field names
#write them to a csv file with field names as features
def save_as_csv(data, outputfilename):
    print(data)

    fieldnames = ['X', 'Y', 'class']
    id = 0
    writer = csv.DictWriter(open(outputfilename, 'w'), fieldnames=fieldnames)

    writer.writeheader()
    for item in data:
        writer.writerow(item)

#generate a set number of datapoints, with three features.
#Features X and Y can be either categorical (A/B) or numeric (randomly generated)
#class is always a 1 or 0
def generateData(X_cat=True, Y_cat=True):
    data = []
    defaults = [{"X": "A", "Y":"A", 'class': 1}, {"X": "B", "Y":"A", 'class': 0},
                {"X": "A", "Y":"B", 'class': 0}, {"X": "B", "Y":"B", 'class': 1}]
    for i in range(NUM_POINTS):
        point = {}
        index = randrange(4)
        if X_cat:
            point["X"] = defaults[index]["X"]
        else:
            if defaults[index]["X"] == "A":
                point["X"] = DEFAULT_NUMERIC_VALUE + (randrange(DEFAULT_NUMERIC_VALUE + 1) - (0.5 * DEFAULT_NUMERIC_VALUE))
            else:
                point["X"] = (-1 * DEFAULT_NUMERIC_VALUE) + (randrange(DEFAULT_NUMERIC_VALUE + 1) - (0.5 * DEFAULT_NUMERIC_VALUE))
        if Y_cat:
            point["Y"] = defaults[index]["Y"]
        else:
            if defaults[index]["Y"] == "A":
                point["Y"] = DEFAULT_NUMERIC_VALUE + (randrange(DEFAULT_NUMERIC_VALUE + 1) - (0.5 * DEFAULT_NUMERIC_VALUE))
            else:
                point["Y"] = (-1 * DEFAULT_NUMERIC_VALUE) + (randrange(DEFAULT_NUMERIC_VALUE + 1) - (0.5 * DEFAULT_NUMERIC_VALUE))
        point["class"] = defaults[index]["class"]
        data.append(point)
    return data



