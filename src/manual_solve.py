#!/usr/bin/python

'''
Name: Caolan McDonagh
Student ID: 21249929
Github: https://github.com/caolanmc/ARC

Task 23b5c85d:

Task 4290ef0e:

Task caa06a1f:

Task 2dd709a:

Task 62c24649:

Summary:


'''

import os, sys
import json
import numpy as np
import re

from collections import Counter
import sklearn as sk
import matplotlib.pyplot as plt

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_23b5c85d(x):
    xFlat = x.flatten()
    count = Counter(xFlat)

    #leastCommon = min(count.values())
    digit = [key for key, value in count.items() if value == min(count.values())] #get the value of our least frequent number.
    digit = digit[0] #Extract our int

    for i in x:
        i = i[i == digit]
        print(i)
    

    print('=========================')
    print(digit)
    print('=========================')
    #easy
    return x

#def solve_4290ef0e(x):
    #easy
    return x

#def solve_caa06a1f(x):
    #easy
    return x

#def solve_2dd709a(x):
    #Medium?
    return x

#def solve_62c24649(x):
    #Easy/Medium?
    return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    #print(taskID)
    train_input, train_output, test_input, test_output = data
    #print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    #print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    #print("Input")
    #print(x)
    #print("Correct output")
    #print(y)
    #print("Our output")
    #print(yhat)
    #print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

