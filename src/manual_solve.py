#!/usr/bin/python

'''
Name: Caolan McDonagh
Student ID: 21249929
Github: https://github.com/caolanmc/ARC

Task 23b5c85d:
    Simple enough task, x number of coloured quadrilaterals generated, the idea is to return just the smallest.
    I flatten the x ndArray, to get a count of each colour (number), I now have the smallest int, I then go through the original x ndArray, removing anything that isn't our smallest int, and vStack() the remainder into a new temp array, this is then let = to x.
    Task 4290ef0e:

Task caa06a1f:
    Had a lot of trouble with this, ended up going about it wrong twice, undoing all my work. Currently in its not
    fully working state it is able to find the original pattern, and replace the non patterened colours/numbers
    with the pattern, in the correct array size. Now I need to transform this pattern. Demonstrations show
    the patter swaps, but was having difficulty with this and moved on temporarily.
    Task 2dd709a:

Task 62c24649:
    :(

Task 62c24649:
    Went about rotating all the ndArrays +90/-90/+180 degrees, took a second look at the ARC html only to note none of these are transformed count/clockwise, they're actually mirrored. So all of  that work was put in the bin, thankfully I read up on np.flip and
    and was able to implement this mirroring pretty stress free. Our original X is basically the top left quarted of our image, we flip this horizontally to get the top left, flip these two vertically to get our bottom half, now we have the total reqested output.

Task 08ed6ac7:

Summary:
    Transcriber/describer -> construction 

'''

import enum
import os, sys
import json
import numpy as np
import re

from collections import Counter
from numpy.core.fromnumeric import shape
import sklearn as sk
import matplotlib.pyplot as plt
from math import dist

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
'''
def solve_23b5c85d(x):
    #easy
    xFlat = x.flatten()
    count = Counter(xFlat)

    #leastCommon = min(count.values())
    digit = [i for i, value in count.items() if value == min(count.values())] #get the value of our least frequent number.
    digit = digit[0] #Extract our int

    temp = np.array([])

    for i in x:
        i = i[i == digit] #Remove anything that isn't our lowest digit.
        if i.any(): #Removes any empty rows
            if temp.any(): #Checks if out temp ndarray is empty, if it isn't, adds our next row of digits.
                temp = np.vstack((temp,i))
            else: #Temp found empty, so we add our first line, this allows us to get the correct size for the above vstack()
                temp = np.fromiter(i, dtype=int)
    x = temp
    
    return x
'''
#def solve_4290ef0e(x):
    #difficult?
    #return x
'''
def solve_caa06a1f(x):
    #medium
    pattern = []
    
    #The below finds the pattern, for future use in remakingthe grid.
    for index, i in enumerate(x):
        if pattern:
            for j in i:
                if(j in pattern):
                    pass
                else:
                    pattern.append(j)
        else:
            #index 0 (for getting first digit in pattern)
            pattern.append(i[0])

    digit = pattern.pop() #Labels our useless number, not in pattern.

    #Keep the shape ouf our original array, as np.tile will extend beyond what we want.
    shape = ((int(x.shape[0]), (int(x.shape[1]))))
    newGrid = np.tile(pattern,(shape))

    #Trim the ndarray columns as they are multiplied by our pattern length.
    #Spent a long time trying to fix this at the root cause, but no luck.
    #We trim the num of columns equal to num of rows, as these are always equal.
    trim = shape[1]-newGrid.shape[1]
    newGrid = newGrid[:,:trim]
    
    x = newGrid

    return x
'''
'''
def solve_2dd70a9a(x):
    #Medium?
    return x
'''
'''
def solve_62c24649(x):
    #Easy

    #Break all sections of our "mirror", numpy does all the heavy lifting with its flip* methods.
    topLeft = x
    topRight = np.fliplr(x)
    top = np.array([])
    bottom = np.array([])

    #Flip (Horizontal) our "top left" (original x) to get our mirror on the right, append this.
    #Flip (Vertical) this entire appended topleft/topright to get our full mirror bottom half
    top = np.append(topLeft,topRight,axis=1)
    bottom = np.flip(top)

    #Append these halves to get our entire mirrored array.
    x = np.append(top,bottom,axis=0)    

    return x
'''
'''
def solve_6d0aefbc(x):
    # Easier version of task 62c24649 seen above, only doing it for the sake of it.
    topLeft = x
    topRight = np.fliplr(x)
    top = np.array([])

    top = np.append(topLeft,topRight,axis=1)
    x = top

    return x

    '''
    
#def solve_08ed6ac7(x):
    #Medium?
    #return x

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
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

