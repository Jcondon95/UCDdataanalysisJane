import pandas as pd
from pandas.tests.groupby.test_value_counts import bins
from pyparsing import col

data = pd.read_csv("gapminder.csv")

# Print the last item from years and populations
print(data['year'[-1]])
print(data['pop'[-1]])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(data['year', 'pop'])

# Display the plot with plt.show()
plt.show()
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(data['year', 'pop'])

# Display the plot with plt.show()
plt.show()

# Change the line plot below to a scatter plot
plt.scatter(data['gdp_cap', 'life_exp'])

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

# Create histogram of life_exp data
plt.hist(data['life_exp'])

# Display histogram
plt.show()

# Build histogram with 5 bins
plt.hist(data['life_exp'], bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(data['life_exp'], bins=20)

# Show and clean up again
plt.show()
plt.clf()

# Basic scatter plot, log scale
plt.scatter(data['gdp_cap', 'life_exp'])
plt.xscale('log')

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()

# Scatter plot
plt.scatter(data['gdp_cap', 'life_exp'])

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# After customizing, display the plot
plt.show()

# Import numpy as np
import numpy as np

# Specify c and alpha inside plt.scatter()
plt.scatter(x=data['gdp_cap'], y=data['life_exp'], s=np.array(data['pop']) * 2, c=col, alpha=0.8)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Show the plot
plt.show()

# Scatter plot
plt.scatter(x=data['gdp_cap'], y=data['life_exp'], s=np.array(data['pop']) * 2, c=col, alpha=0.8)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()

# Store pop as a numpy array: np_pop
np_pop = np.array(data['pop'])

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(data['gdp_cap'], ['life_exp'], s=np_pop)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Display the plot
plt.show()

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'berlin', 'norway': 'oslo'}

# Print europe
print(europe)

# Dictionaries

# Dictionary of dictionaries
europe = {'spain': {'capital': 'madrid', 'population': 46.77},
          'france': {'capital': 'paris', 'population': 66.03},
          'germany': {'capital': 'berlin', 'population': 80.62},
          'norway': {'capital': 'oslo', 'population': 5.084}}

# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital': 'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country': names, 'drives_right': dr, 'cars_per_cap': cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)

import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = {'country': names, 'drives_right': dr, 'cars_per_cap': cpc}
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)

# Import pandas as pd
import pandas as pd

# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')

# Print out cars
print(cars)

# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col=0)

# Print out cars
print(cars)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Data Frames loc / iloc

# Print out observation for Japan
print(cars.loc['JPN'])

# Print out observations for Australia and Egypt
print(cars)
print(cars.loc[['AUS', 'EG']])

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Print out drives_right value of Morocco
print(cars.loc['MOR', 'drives_right'])

# Print sub-DataFrame
print(cars)
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])

# Equality

# Comparison of booleans
True == False

# Comparison of integers
-5 * 15 != 75

# Comparison of strings
"pyscript" == "PyScript"

# Compare a boolean with an integer
True == 1

# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print("test" <= y)

# Comparison of booleans
print(True > False)

# Create arrays
import numpy as np

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Extract drives_right column as Series: dr
dr = cars['drives_right']

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]

# Print medium
print(medium)

# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)

# Initialize offset
offset = -6

# Code the while loop
while offset != 0:
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
areas: float
for areas in areas:
    print(areas)

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for index, a in enumerate(areas):
    print("room" + str(index) + ":" + str(a))

    # areas list
    areas = [11.25, 18.0, 20.0, 10.75, 9.50]

    # Code the for loop
    for index, area in enumerate(areas):
        print("room " + str(index + 1) + ": " + str(area))

# house list of lists
house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]

# Build a for loop from scratch
for room in house:
    print('the ' + room[0] + ' is ' + str(room[1]) + ' sqm')

# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'berlin',
          'norway': 'oslo', 'italy': 'rome', 'poland': 'warsaw', 'austria': 'vienna'}

# Iterate over europe
for country, capital in europe.items():
    print("the capital of " + country + ' is ' + str(capital))

# Import numpy as np
import numpy as np

MLB = pd.read_csv("baseball.csv")

# For loop over np_height
for x in MLB('np_height'):
    print(str(x) + " inches")

# For loop over np_baseball
for x in np.nditer(MLB('np_baseball')):
    print(x)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    assert isinstance(row, object)
    print(row)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Adapt for loop
for lab, row in cars.iterrows():
    print(lab + ": " + str(row['cars_per_cap']))

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Print cars
assert isinstance(cars, object)
print(cars)

# Import cars data
import pandas as pd

cars = pd.read_csv('cars.csv', index_col=0)

# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)

# Random Numbers

# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

# Import numpy and set seed
import numpy as np

np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1, 7))

# Use randint() again
print(np.random.randint(1, 7))

# Numpy is imported, seed is set

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1, 7)

# Finish the control construct
if dice <= 2:
    step = step - 1
elif dice <= 5:
    step = step + 1
else:
    step = step + np.random.randint(1, 7)

# Print out dice and step
print(dice)
print(step)

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

# Complete the ___
for x in range(100):
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1, 7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

print(random_walk)

# Numpy is imported, seed is set

# Initialization
random_walk = [0]

for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()

# Numpy is imported; seed is set

# Initialize all_walks (don't change this line)
all_walks = []

# Simulate random walk 10 times
for i in range(10):

    # Code from before
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()
# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()

# numpy and matplotlib imported, seed set

# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()
