#!/usr/bin/python

'''
Name: Caolan McDonagh
Student ID: 21249929
Github: https://github.com/caolanmc/ARC

Task 23b5c85d (All grids successful):
    Simple enough task, x number of coloured quadrilaterals generated, the idea is to return just the smallest.
    I flatten the x ndArray, to get a count of each colour (number), I now have the smallest int, I then go through the original x ndArray, removing anything that isn't our smallest int, and vStack() the remainder into a new temp array, this is then let = to x.

Task caa06a1f (All grids successful):
    Had a lot of trouble with this, ended up going about it wrong twice, undoing all my work. Currently in its not
    fully working state it is able to find the original pattern, and replace the non patterened colours/numbers
    with the pattern, in the correct array size. Now I need to transform this pattern. Demonstrations show
    the patter swaps, but was having difficulty with this and moved on temporarily.
    Task 2dd709a:

Task 62c24649 (All grids successful):
    Went about rotating all the ndArrays +90/-90/+180 degrees, took a second look at the ARC html only to note none of these are transformed count/clockwise, they're actually mirrored. So all of  that work was put in the bin, thankfully I read up on np.flip and
    and was able to implement this mirroring pretty stress free. Our original X is basically the top left quarted of our image, we flip this horizontally to get the top left, flip these two vertically to get our bottom half, now we have the total reqested output.

Task 6d0aefbc (All grids successful):
    Same as above task (6d0aefbc), just only one flip horizontally, none vertically. I also saw another task
    with the same concept, except it was one flip vertically, not horizontally.

Task a61f2674 (All grids successful):
    This was fun, I went about it in a gnarly way with multiple loops etc, went scrounging thorugh numpy documentation and 
    found practically everything I needed by going down a rabbit hole. The idea of the task is the find the largest and 
    smallest columns, give them a new value/colour and remove everthing else.

Task 4347f46a (All grids successful):
    I was looking into using SK learn or something other than a gross mess of loops and if statements.
    I came across Scikit image segmentation, which has edge detection modules. This task was to carve
    out the inside of given quadrilateral, sk image "Find_Boundaries" can find these quads, then use
    the "inner" mode to empty anything that != edge. Great, and I feel it could be applicable across 
    many tasks.

Summary:
    In summary of this assignment, I think this is probably some of the best testing my programming has come under, in terms of trying to think out problems. What is so easy to compute in my own head can be an absolute
    nightmare when transcribing to instructions to an interpreter. Its a great way of showing how tasks considered simple by humans are much more complex under the hood when it comes to letting a computer or some sort of 
    AI algorithm work them out.
        Working through my different solve* tasks, there was a lot of tasks, which all under the guise of the abstract and reasoning corpus, they were vastly different in how you would go about solving them via
    programming. There was also a few that would fall under teh same school of thought, where it is about max or min counts of a given colour, distance orientated tasks, simple image mirroring or rotation. These 
    tasks could all be solved by simiar solve* methods, with a bit of tweaking and allowance for some variance. This got me thinking a great way to make a more overarching soltuion/master solve* metod would be 
    to break up the work. You could have some sort of transcriber/describer, which has learnt from previous tasks to be able to classify a given task, e.g image mirroring, colour counting, pattern solving,
    distance related tasks etc... This transcriber could then issue the task to the correlating constructor that would carry out the given task. Outputs then tested, we can see if that was a good decision
    and bank the data. The commonalities between tasks are in a more catergorical level/broader perspective, rather than task to task basis. I learned this through checking dozens of tasks via the random button, they're
    all different, but a good few share the same core concept, testing different parts of the human brain.
        While working on this, I found numpy was my greatest companion. It does a lot of the heavy lifting when it comes to ndarray manipulation and saved me a lot of stress and time while working through the problems.
    Scikit image saved the day towards the end, but I found it too late in my ventures into this assignment, I feel it is much more powerful than I realize and could have been utilized in a lot of these problems. I went 
    into them looking at them as matrix related problems, but SK Image showed me I can look at some of them as images, treat the ndArrays as an image as I would have looking at the image in grid form on the html. I would
    also bet there is better use in other packages, but again, I looked too late.
        Chollet speaks on how "Generalization is not quantified" with regards to no quantitative of the generalization of the evanuation set given the test set difficulty. Some of these tasks I would say are easy and could
    solve in seconds, where someone else might struggle for a while. I could jump to the next task and have the opposite reaction, which I did infact encounter some tasks I couldn't wrap my head around. This difficulty does
    need to be quantified through human performance, which Chollet does intend on doing, going off his paper. This also factors into the task solving. I could find a task very hard to programme, as I found the task hard
    to solve with my brain, where another programmer might find it easy to solve both in brain, and so is able to implement it more efficiently. There is tricks that can be missed by the human brain in both real life interpretation
    and a programming interpretation so this can cause hiccups when trying to recreate human intelligence, as intelligence has no definite measure. It is quite the task Challet is after.



'''

import enum
import os, sys
import json
import numpy as np
import re

from collections import Counter
from skimage.segmentation import find_boundaries

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

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

def solve_6d0aefbc(x):
    # Easier version of task 62c24649 seen above, only doing it for the sake of it.
    topLeft = x
    topRight = np.fliplr(x)
    top = np.array([])

    top = np.append(topLeft,topRight,axis=1)
    x = top

    return x

def solve_a61f2674(x):
    #Medium?

    #Below could just be replaced with value = 5 (Grey), but I didn't realise that till after, this will work if columns != grey or 5
    value = np.unique(x)
    value = int(np.delete(value,np.where(value==0)))

    
    columnCheck = np.count_nonzero(x==value, axis=0) #Here we get the count of non zeros per columnd
    sortedColumns = np.sort(columnCheck) #Sort the columns for grabbing largest/smallest count later
    indices = np.argsort(columnCheck) #Returns the unsorted indices so we can correctly reference the largest/smallest in relation to the original

    smallestIndex = sortedColumns.tolist().index(next(filter(lambda x: x!=0, sortedColumns))) #returns index where smallest non 0 column is AKA first non 0 in sorted list
    smallestIndex = indices[smallestIndex] 
    largestIndex = indices[-1] #returns index where largest column is AKA last element in the sorted list

    #These are the numerical value/colour of the smallest and largest digits respectivly, not useful in this instance though.
    #smallestval = columnCheck[smallestIndex]
    #largestval = columnCheck[largestIndex]

    #These two lines will replace our largest and smallest columns, where values !=0 with the new respective number/colour.
    newSmall = np.where(x[:,smallestIndex]!=0,2,0)
    newLarge = np.where(x[:,largestIndex]!=0,1,0)

    #Replace the altered columns in the original x ndArray
    x[:,smallestIndex] = newSmall
    x[:,largestIndex] = newLarge

    #Replace all values equal to 'value' (grey) with 0.
    x = np.where(x == value, 0, x)
    
    return x

def solve_4347f46a(x):
    #difficult (without prior knowledge of sk image)?
    #I was looking at tackling like this with a more hands on approach as I have done all along, but I looked into
    #libraries like sk.learn to see if there was any tools that could help, I found sk image.
    #Studied their modu;es and there is acutally lots of useful stuff here that I wish I knew earlier: https://scikit-image.org/docs/stable/api/skimage.segmentation.html
    
    #returns the same ndArray, with true/false in place of where alternate "pixels" or digits
    findboundaries = find_boundaries(x, mode = 'inner')

    #This can simply be multiplied by the original x to result in numerical output, rather than bool.
    x = findboundaries * x

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

