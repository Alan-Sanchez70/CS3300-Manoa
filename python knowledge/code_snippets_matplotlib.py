# A first example
import matplotlib.pyplot as plt

plt.plot([-1, -4.5, 16, 23])
plt.show()


# A first example 2
plt.plot([-1, -4.5, 16, 23], "ob")
plt.show()


# X values
import matplotlib.pyplot as plt
# our X values:
days = list(range(0, 22, 3))
# our Y values:
celsius_values = [25.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
plt.plot(days, celsius_values)
plt.show()

# X values - discrete
import matplotlib.pyplot as plt
# our X values:
days = list(range(0, 22, 3))
# our Y values:
celsius_values = [25.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
plt.plot(days, celsius_values, "ob")
plt.show()


# Labels on axes
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_values = [25.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
plt.plot(days, celsius_values)
plt.xlabel('Day')
plt.ylabel('Degrees Celsius')
plt.show()


# arbitrary number of x, y,
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
celsius_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
plt.xlabel('Day')
plt.ylabel('Degrees Celsius')
plt.plot(days, celsius_min,
         days, celsius_min, "oy",
         days, celsius_max,
         days, celsius_max, "or")
plt.show()


# arbitrary number of x, y,
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
celsius_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
plt.xlabel('Day')
plt.ylabel('Degrees Celsius')
plt.plot(days, celsius_min)
plt.plot(days, celsius_min, "oy")
plt.plot(days, celsius_max)
plt.plot(days, celsius_max, "or")
plt.show()


# Checking and Defining the Range of Axes 1
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
celsius_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
plt.xlabel('Day')
plt.ylabel('Degrees Celsius')
plt.plot(days, celsius_min,
         days, celsius_min, "oy",
         days, celsius_max,
         days, celsius_max, "or")
print("The current limits for the axes are:")
# (0.6499999999999999, 8.35, 18.62, 40.18)
print(plt.axis())
print("We set the axes to the following values:")
xmin, xmax, ymin, ymax = 0, 10, 14, 45
print(xmin, xmax, ymin, ymax)
# 0 10 14 45
plt.axis([xmin, xmax, ymin, ymax])
plt.show()


# # Checking and Defining the Range of Axes 2
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
celsius_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
plt.xlabel('Day')
plt.ylabel('Degrees Celsius')
plt.plot(days, celsius_min,
         days, celsius_min, "oy",
         days, celsius_max,
         days, celsius_max, "or")
plt.axis([0, 10, 18, 41])
plt.show()


# "linspace" to Define X Values
import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(0, 2 * np.pi, 50, endpoint=True)
F = np.sin(X)
plt.plot(X,F)
startx, endx = -0.1, 2*np.pi + 0.1
starty, endy = -1.1, 1.1
plt.axis([startx, endx, starty, endy])
plt.show()


# "linspace" to Define X Values
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=True)
F1 = 3 * np.sin(X)
F2 = np.sin(2*X)
F3 = 0.3 * np.sin(X)
startx, endx = -2 * np.pi - 0.1, 2*np.pi + 0.1
starty, endy = -3.1, 3.1
plt.axis([startx, endx, starty, endy])
plt.plot(X,F1)
plt.plot(X,F2)
plt.plot(X,F3)
plt.show()


# Adding discrete points
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=True)
F1 = 3 * np.sin(X)
F2 = np.sin(2*X)
F3 = 0.3 * np.sin(X)
startx, endx = -2 * np.pi - 0.1, 2*np.pi + 0.1
starty, endy = -3.1, 3.1
plt.axis([startx, endx, starty, endy])
plt.plot(X,F1, 'r')
plt.plot(X,F1, 'rs')
plt.plot(X,F2, 'b')
plt.plot(X,F2, 'bo')
plt.plot(X,F3, 'g')
plt.plot(X,F3, 'gx')
plt.show()


# Changing the Line Style
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 2 * np.pi, 50, endpoint=True)
F1 = 3 * np.sin(X)
F2 = np.sin(2*X)
F3 = 0.3 * np.sin(X)
F4 = np.cos(X)
plt.plot(X, F1, color="blue", linewidth=2.5, linestyle="-")
plt.plot(X, F2, color="red", linewidth=1.5, linestyle="--")
plt.plot(X, F3, color="green", linewidth=2, linestyle=":")
plt.plot(X, F4, color="grey", linewidth=2, linestyle="-.")
plt.show()


# Fill between
import numpy as np
import matplotlib.pyplot as plt

n = 256
X = np.linspace(-np.pi,np.pi,n,endpoint=True)
Y = np.sin(2*X)
plt.plot (X, Y, color='blue', alpha=1.00)
plt.fill_between(X, 0, Y, color='blue', alpha=.1)
plt.show()


# Fill between 2
import numpy as np
import matplotlib.pyplot as plt

n = 256
X = np.linspace(-np.pi,np.pi,n,endpoint=True)
Y = np.sin(2*X)
plt.plot (X, Y, color='blue', alpha=1.00)
plt.fill_between(X, Y, 1, color='blue', alpha=.1)
plt.show()


# Changing axes position
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 70, endpoint=True)
F1 = np.sin(2* X)
F2 = (2*X**5 + 4*X**4 - 4.8*X**3 + 1.2*X**2 + X + 1)*np.exp(-X**2)
# get the current axes, creating them if necessary:
ax = plt.gca()
# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
# moving left spine to the right to position x == 0:
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.plot(X, F1)
plt.plot(X, F2)
plt.show()


# Customazing ticks
import numpy as np
import matplotlib.pyplot as plt

locs, labels = plt.xticks()
print(locs, labels)
locs, labels = plt.yticks()
print(locs, labels)
plt.show()
# [0.  0.2 0.4 0.6 0.8 1. ] <a list of 6 Text xticklabel objects>
# [0.  0.2 0.4 0.6 0.8 1. ] <a list of 6 Text yticklabel objects>


plt.xticks( np.arange(10) )
locs, labels = plt.xticks()
print(locs, labels)
plt.show()
# [0 1 2 3 4 5 6 7 8 9] <a list of 10 Text xticklabel objects>


plt.xticks( np.arange(4),
           ('Berlin', 'London', 'Hamburg', 'Toronto') )
plt.show()


# Trigonometric example
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
F1 = np.sin(X**2)
F2 = X * np.sin(X)
# get the current axes, creating them if necessary:
ax = plt.gca()
# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
# moving left spine to the right to position x == 0:
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xticks( [-6.28, -3.14, 3.14, 6.28])
plt.yticks([-3, -1, 0, +1, 3])
plt.plot(X, F1)
plt.plot(X, F2)
plt.show()


# Trigonometric example 2
X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
F1 = np.sin(X**2)
F2 = X * np.sin(X)
# get the current axes, creating them if necessary:
ax = plt.gca()
# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
# moving left spine to the right to position x == 0:
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xticks( [-6.28, -3.14, 3.14, 6.28],
        [r'$-2\pi$', r'$-\pi$', r'$+\pi$', r'$+2\pi$'])
plt.yticks([-3, -1, 0, +1, 3])
plt.plot(X, F1)
plt.plot(X, F2)
plt.show()


# Adjusting the tick labels
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
F1 = np.sin(X**2)
F2 = X * np.sin(X)
# get the current axes, creating them if necessary:
ax = plt.gca()
# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
# moving left spine to the right to position x == 0:
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xticks( [-6.28, -3.14, 3.14, 6.28],
        [r'$-2\pi$', r'$-\pi$', r'$+\pi$', r'$+2\pi$'])
plt.yticks([-3, -1, 0, +1, 3])

for xtick in ax.get_xticklabels():
    xtick.set_fontsize(18)
    xtick.set_bbox(dict(facecolor='yellow', edgecolor='black', alpha=0.7 ))

for ytick in ax.get_yticklabels():
    ytick.set_fontsize(14)
    ytick.set_bbox(dict(facecolor='green', edgecolor='None', alpha=0.7 ))

plt.plot(X, F1)
plt.plot(X, F2)

plt.show()


# Legend
x = np.linspace(0, 25, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, '-b', label='sine')
plt.plot(x, y2, '-r', label='cosine')
plt.legend()
plt.show()


# Legend 2
x = np.linspace(0, 25, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, '-b', label='sine')
plt.plot(x, y2, '-r', label='cosine')
plt.legend(loc='upper left')
plt.ylim(-1.5, 2) # To make some space
plt.show()


# Legend,  best
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 70, endpoint=True)
F1 = np.sin(0.5*X)
F2 = -3 * np.cos(0.8*X)
plt.xticks( [-6.28, -3.14, 3.14, 6.28],
        [r'$-2\pi$', r'$-\pi$', r'$+\pi$', r'$+2\pi$'])
plt.yticks([-3, -1, 0, +1, 3])
plt.plot(X, F1, label="$sin(0.5x)$")
plt.plot(X, F2, label="$-3 cos(0.8x)$")
plt.legend(loc='best')
plt.show()


# Legend, best 2
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 70, endpoint=True)
F1 = np.sin(0.5*X)
F2 = 3 * np.cos(0.8*X)
plt.xticks( [-6.28, -3.14, 3.14, 6.28],
        [r'$-2\pi$', r'$-\pi$', r'$+\pi$', r'$+2\pi$'])
plt.yticks([-3, -1, 0, +1, 3])
plt.plot(X, F1, label="$sin(0.5x)$")
plt.plot(X, F2, label="$3 cos(0.8x)$")
plt.legend(loc='best')
plt.show()


# Title
X = np.linspace(-2 * np.pi, 2 * np.pi, 70, endpoint=True)
F1 = np.sin(0.5*X)
F2 = 3 * np.cos(0.8*X)
plt.xticks( [-6.28, -3.14, 3.14, 6.28])
plt.yticks([-3, -1, 0, +1, 3])
plt.plot(X, F1, label="$sin(0.5x)$")
plt.plot(X, F2, label="$3 cos(0.8x)$")
plt.legend(loc='best')
plt.title("Sin and 3 cos")
plt.show()


# Subplots
import matplotlib.pyplot as plt

X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
F1 = np.sin(0.5*X)
F2 = 3 * np.cos(0.8*X)
plt.subplot(2,2,1)
plt.plot(X, F1)
plt.subplot(2,2,4)
plt.plot(X, F2)
plt.show()


# Subplots 2
plt.subplot(2,1,1)
plt.subplot(2,3,6)
plt.show()


# Grid lines
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
x = np.linspace(0, 25, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, 'b')
plt.plot(x, y2, 'r')
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.show()


# 9 Bar chart
x = ['Now york', 'Miami', 'Colorado Springs', 'Los angeles']
y = [40000, 27000, 55000, 20000]
plt.bar(x, y)
plt.show()

# 10 Histogram
# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(500)

# num_bins = [0, 80, 100, 120, 200]
num_bins = 50  # Bins of the histogram

# the histogram of the data, density = 1 for normalized form
plt.hist(x, num_bins, density=1)

plt.xlabel('Smarts')
plt.ylabel('Probability density')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

plt.show()

# 12 Scatter plot
a = np.array([[64, 165],
              [70, 170],
              [73, 180],
              [80, 179],
              [82, 190],
              [75, 185],
              [62, 166],
              [72, 172],
              [71, 175],
              [82, 177],
              [86, 185],
              [76, 181]])
plt.scatter(a[:, 0], a[:, 1], label='Weight vs height', marker='o', s=10)
plt.legend()
plt.show()

# 13 More complex scatter plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N)) ** 2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

# 14 Pie chart
slices = [15, 6, 3, 1, 1]
activities = ['sleeping', 'playing video games',
              'eating', 'studying', 'doing homework']
explode = (0.0, 0.0, 0.1, 0.0, 0.3)
plt.pie(slices, labels=activities, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# 15 Loading data from files using numpy
x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)

plt.plot(x, y)
plt.show()

# 16 Loading data from files using csv
x = []
y = []

with open('example.txt', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')

    for row in plots:
        x.append(row[0])
        y.append(row[1])

plt.plot(x, y)
plt.show()

## 

# import csv
x = []
y = []
# var1 = ''
# var2 = ''

with open('example.txt', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')

    for row in plots:
        x.append(row[0])
        y.append(row[1])

    # var1 = x.pop(0)
    # var2 = y.pop(0)

plt.plot(x, y)
# plt.xlabel(var1)
# plt.ylabel(var2)
plt.show()

