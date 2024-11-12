# Python_Portfoli0_2
This is the portfolio of python code that I learned during BISC-450C

## Analyizing Patient data

In this analysis, we look at inflammation data for multiple patients.

```python
import numpy
```


```python
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape)
```

    (60, 40)



```python
print('first value in data:', data[0,0])
```

    first value in data: 0.0



```python
print('middle value in data:', data[29,19])
```

    middle value in data: 16.0



```python
print(data[0:4,0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10 ,0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3,36:]
```


```python
print('small is:')
```

    small is:



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Lets us a numpy function
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)

```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
maxval = numpy.amax(data)
minval = numpy.amin(data)
stdval = numpy.std(data)
```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation: 4.613833197118566



```python
# Sometimes we want to look at variation in statistial values, such as maxmimum inflammation per pateint, # or average from day one.

patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)

print('maximum inflammation for pateint_0:', numpy.amax(patient_0))


```

    maximum inflammation for pateint_0: 18.0



```python
print('maximum inflammation for pateint 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for pateint 2: 19.0



```python

```


```python

```
## Using Jupyter Notebooks

In this section, we found the mean, max, min, and std dev for each group.

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numeric_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006))
```


![png](output_11_0.png)



```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0, y = 0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'increase in mean Fortune 500 company proftis from 1955 to 2005', 'Profit (millions)')
```


![png](output_16_0.png)



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'increase in Fortune 500 company revenues from 1955 to 2005', 'Revenues (millions)')
```


![png](output_17_0.png)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols= 2)
title = 'increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14,4)
fig.tight_layout()
```


![png](output_18_0.png)



```python

```


```python

```

## Making Choices pt1

```python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53 
print('before conditional...')
if num > 100:
    print(num, 'Is greater than 100')
print('...after conditional')
```

    before conditional...
    ...after conditional



```python
num = 14

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    14 is positive



```python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```

    at least one part is false



```python
if ( 1 > 0) or (-1 >= 0):
    print('atleast one part is true')
else:
    print('both of these are false')
```

    atleast one part is true



```python
import numpy
```


```python

```


```python

```



## Making Choices pt2
```python
import numpy
```


```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```


```python
max_inflammation_0 = numpy.amax(data, axis = 0)[0]


```


```python
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
    
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero!')
else:
    print('Seems OK!')
```

    Suspicious looking maxima!



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter = ',')

max_inflammation_0 = numpy.amax (data, axis = 0)[0]
max_inflammation_20 = numpy.amax (data, axis = 0)[20]
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems ok!')
```

    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!



```python

```


```python

```


## Using Loops

```python
odds = [1, 3, 5, 7]

```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])

```

    1
    3
    5
    7



```python
odds = [1, 3, 5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-3-b48c9fadc7bf> in <module>
          3 print(odds[1])
          4 print(odds[2])
    ----> 5 print(odds[3])
    

    IndexError: list index out of range



```python
odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

for num in odds:
    print(num)
```


```python
length = 0
names = ['Curie', 'Darwin', 'Turing']
for values in names:
    length = length + 1
    print( 'There are', length, 'names in the list.')
```


```python
name = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
    print('after the loop, name is', name)
```


```python
print(len([0,1,2,3]))
```


```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```


```python

```


```python

```



 ## Storing values in Lists

```python
odds =  [1, 3, 5, 7]
print('odds are:', odds)

```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds [0])
print('last element:', odds[3])
print('"-1" element:', odds[-1])

```

    first element: 1
    last element: 7
    "-1" element: 7



```python
names = ['curie', 'Darwing', 'Turing'] # Typo in darwin's name
print('name is orignally:', names)
names[1] = 'Darwin' # Correct the name
print('final value of names:', names)

```

    name is orignally: ['curie', 'Darwing', 'Turing']
    final value of names: ['curie', 'Darwin', 'Turing']



```python
#name = 'Darwin'
#name[0] = 'd'
```


```python
odds.append(11)
print('odds after adding a value', odds)
```

    odds after adding a value [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(0)
print('odds after removing the first element', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element [3, 5, 7, 11]
    removed_element: 1



```python
odds.reverse()
print('odds after reversing:', odds)

```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)

```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3, 5, 7]
primes = list(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)

```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
binomial_name = "Drosphilia melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['x', 'y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosome:', autosomes)

last = chromosomes[-1]
print('last:', last)

```

    group: Drosphilia
    species: melanogaster
    autosome: ['2', '3', '4']
    last: 4



```python
data = 'Monday 4 January 2023'
day = data[0:6]
print('Using 0 to begin range:', day)
day = data[:6]
print('Ommiting beginning index:', day)
```

    Using 0 to begin range: Monday
    Ommiting beginning index: Monday



```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position', sond)
sond = months[0:len(months)]
print('Using len()to get last entry:', sond)
sond = ('Ommiting ending index', sond)
```

    With known last position ['sep', 'oct', 'nov', 'dec']
    Using len()to get last entry: ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']



```python

```


```python

```

## Creating Functions pt 1

```python
fahrenheit_val = 99
celsius_val = ((fahrenheit_val -32) *(5/9))


print(celsius_val)
```

    37.22222222222222



```python
fahrenheit_val2 = 43
celsius_val2 = ((fahrenheit_val2 -32) *(5/9))

print(celsius_val2)
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp -32) * (5/9))
    # Return the values of the new variable
    return converted

```


```python
def fahr_to_celsius(temp):
    # Return converted values more effciently using the return function without creating a new varaible. This code does the same thing as the previous function but it is more explicit in explaining how the return command works.

    return ((temp -32) * (5/9))


```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')
```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
def celsius_to_kelvin(temp_C):
    return temp_C + 273.15

print('Freezing point of water in Kelvin:', celsius_to_kelvin(0.))
```

    Freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(temp_F):
    temp_C = fahr_to_celsius(temp_F)
    temp_K = celsius_to_kelvin(temp_C)
    return temp_K

print('Boiling point of water in Kelvin:', fahr_to_kelvin(212.0))
```

    Boiling point of water in Kelvin: 373.15



```python
print("Again, temperature in Kelvin was:", temp_K)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-21-9c88882de1ba> in <module>
    ----> 1 print("Again, temperature in Kelvin was:", temp_K)
    

    NameError: name 'temp_K' is not defined



```python
temp_kelvin = fahr_to_kelvin(212.0)
print('Temperature in kelvin was:', temp_kelvin)
```

    Temperature in kelvin was: 373.15



```python
temp_kelvin
```




    373.15




```python
def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in kelvin was:', temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was: 212.0
    Temperature in kelvin was: 373.15



```python

```


```python

```


```python

```


```python

```


```python

```
## Creating Functions pt2

```python
import numpy
import glob
import matplotlib
import matplotlib.pyplot
```


```python
def visualize(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```


```python
def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis =0)[20] == 20:
        print('Suspicous looking maxima!')
    elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems ok!')
```


```python
filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames:
    print(filename)
    visualize(filename)
    detect_problems(filename)
```

    inflammation-01.csv



![png](output_3_1.png)


    Suspicous looking maxima!
    inflammation-02.csv



![png](output_3_3.png)


    Suspicous looking maxima!
    inflammation-03.csv



![png](output_3_5.png)


    Minima add up to zero!
    inflammation-04.csv



![png](output_3_7.png)


    Suspicous looking maxima!
    inflammation-05.csv



![png](output_3_9.png)


    Suspicous looking maxima!
    inflammation-06.csv



![png](output_3_11.png)


    Suspicous looking maxima!
    inflammation-07.csv



![png](output_3_13.png)


    Suspicous looking maxima!
    inflammation-08.csv



![png](output_3_15.png)


    Minima add up to zero!
    inflammation-09.csv



![png](output_3_17.png)


    Suspicous looking maxima!
    inflammation-10.csv



![png](output_3_19.png)


    Suspicous looking maxima!
    inflammation-11.csv



![png](output_3_21.png)


    Minima add up to zero!
    inflammation-12.csv



![png](output_3_23.png)


    Suspicous looking maxima!



```python
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```


```python
z =  numpy.zeros((2,2))
print(offset_mean(z, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

print(offset_mean(data, 0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]



```python
print('Original min, mean, and max are:', numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data, 0)
print('min, mean, and max of offset data are:',
      numpy.amin(offset_data),
       numpy.mean(offset_data),
       numpy.amax(offset_data))
```

    Original min, mean, and max are: 0.0 6.14875 20.0
    min, mean, and max of offset data are: -6.14875 2.842170943040401e-16 13.85125



```python
print('std dev before and after', numpy.std(data), numpy.std(offset_data))
```

    std dev before and after 4.613833197118566 4.613833197118566



```python
print('difference in standard deviation before and after:',
      numpy.std(data) - numpy.std(offset_data))
```

    difference in standard deviation before and after: 0.0



```python
# offset_mean (data, target_mean_value):
#return a new array containing the original data with its mean offset to match the desired value.

def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value

```


```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean offset to match the desired value"""
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value
    



```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the original data
    with its mean offset to match the desired value.
    
    Examples
    -------
    
    >>> Offset_mean[1,2,3], 0)
    array([-1., 0., 1.]) """
    
    return (data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data
        with its mean offset to match the desired value.
        
        Examples
        -------
        
        >>> Offset_mean[1,2,3], 0)
        array([-1., 0., 1.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')

```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')

```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def offset_mean(data, target_mean_value = 0.0):
    """Return a new array containing the original data
    with its mean offset to match the desired value, (0 by default).
    
    Examples
    ----------
    
    >>> Offset_mean[1,2,3], 0)
    array([-1., 0., 1.]) """
    
    return (data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))
print(offset_mean(test_data, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
print(offset_mean(test_data))
```

    [[0. 0.]
     [0. 0.]]



```python
def display(a=1, b=2, c=3):
    print('a:', a, 'b:', b, 'c:', c)

print('no perameters:')
display()
print('one perameter:')
display()
print('two perameters:')

```

    no perameters:
    a: 1 b: 2 c: 3
    one perameter:
    a: 1 b: 2 c: 3
    two perameters:



```python
print('only setting the value of c')
display(c=77)
```

    only setting the value of c
    a: 1 b: 2 c: 77



```python
help(numpy.loadtxt)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')

```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python

def std_dev(sample):
    sample_sum = 0
    for value in sample:
        sample_sum += value
    sample_mean = sample_sum / len(sample)
    
    sum_squared_devs = 0
    for value in sample:
        sum_squared_devs += (value - sample_mean) * (value - sample_mean)
```


```python

```


```python

```
## Defensive Programming

```python
numbers = [1.5, 2.3, 0.7, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
    print('total is:', total)
```

    total is: 1.5
    total is: 3.8
    total is: 4.5
    total is: 8.9



```python
def normalize_rectangle(rect):
    """Normalize a rectangle so that it is at the origin and 1.o units on its longest axis.
    input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle repectivley"""
    assert len(rect) == 4, 'Rectangles must contain 4 coordiantes'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X Coordinates'
    assert y0 < y1, 'Invalid Y Coordinates'
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dx / dy
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
            
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
        
    return (0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle( (0.0, 1.0, 2.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-9-cb5ff8ce860f> in <module>
    ----> 1 print(normalize_rectangle( (0.0, 1.0, 2.0)))
    

    <ipython-input-8-5a0b1bc9f698> in normalize_rectangle(rect)
          3     input should be of the format (x0, y0, x1, y1).
          4     (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle repectivley"""
    ----> 5     assert len(rect) == 4, 'Rectangles must contain 4 coordiantes'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X Coordinates'


    AssertionError: Rectangles must contain 4 coordiantes



```python
print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-10-f7e0d48bdfd0> in <module>
    ----> 1 print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
    

    <ipython-input-8-5a0b1bc9f698> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must contain 4 coordiantes'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X Coordinates'
          8     assert y0 < y1, 'Invalid Y Coordinates'
          9 


    AssertionError: Invalid X Coordinates



```python
print(normalize_rectangle( (0.0, 0.0, 1.0, 5.0) ))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-12-8d4a48f1d068> in <module>
    ----> 1 print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0) ))
    

    <ipython-input-8-5a0b1bc9f698> in normalize_rectangle(rect)
         18 
         19     assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    ---> 20     assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
         21 
         22     return (0, 0, upper_x, upper_y)


    AssertionError: Calculated upper y coordinate invalid



```python

```


```python

```
## Transcription

```python
# Prompt the user to enter the input fasta file name
input_file_name = input("Enter the name of the input fasta file: ")

```

    Enter the name of the input fasta file:  ubiqutin



```python
# Open the input fasta file and read the sequence

with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith (">"):
            continue
        dna_sequence += line.strip()
        
```


```python
# Transcribe the DNA and RNA
rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name

output_file_name = input( "Enter the name of the output file: ")
```

    Enter the name of the output file:  ubiqutin



```python
# Save the RNA sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print( "The RNA sequence has been saved to (output_file_name)")
    
```

    The RNA sequence has been saved to (output_file_name)



```python
print(rna_sequence)
```

    MOLSALRVEEVQNVINAMQKILECPIWSLQESURFSQLVEELLKIICAFQLOUGLEYANSYNFAKKENNSPEHLKOEVSIIQSMGYRNRAKRLLQSEPENPSLEUSLSVQLSNLGUVRULRUKQRIQPQKUSVYIELGSOSSEOUVNKAUYCSVGOQELLQIUPQGUROEISLOSAKKAACEFSEUOVUNUEHHQPSNNOLNUUEKRAAERHPEKYQGEAASGCESEUSVSEOCSGLSSQSOILUUQQROUMQHNLIKLQQEMAELEAVLEQHGSQPSNSYPSIISOSSALEOLRNPEQSUSEKAVLUSQKSSEYPISQNPEGLSAOKFEVSAOSSUSKNKEPGVERSSPSKCPSLOORWYMHSCSGSLQNRNYPSQEELIKVVOVEEQQLEESGPHOLUEUSYLPRQOLEGUPYLESGISLFSOOPESOPSEORAPESARVGNIPSSUSALKVPQLKVAESAQSPAAAHUUOUAGYNAMEESVSREKPELUASUERVNKRMSMVVSGLUPEEFMLVYKFARKHHIULUNLIUEEUUHVVMKUOAEFVCERULKYFLGIAGGKWVVSYFWVUQSIKERKMLNEHOFEVRGOVVNGRNHQGPKRARESQORKIFRGLEICCYGPFUNMPUOQLEWMVQLCGASVVKELSSFULGUGVHPIVVVQPOAWUEONGFHAIGQMCEAPVVUREWVLOSVALYQCQELOUYLIPQIPHSHY



```python

```


```python

```
## Translation

```python
# Prompt the user to enter the input RNA file name

input_file_name = input("Enter the name of the input RNA file: ")

```

    Enter the name of the input RNA file:  BRCA1.txt



```python
# Open the input RNA file and read the RNA sequence

with open(input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table

codon_table = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L" , 
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L" ,   
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M" ,   
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V" ,   
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S" ,   
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P" ,   
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T" ,   
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A" ,   
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*" ,   
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q" ,   
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K" ,      
    "GAU": "O", "GAC": "O", "GAA": "E", "GAG": "E" ,   
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W" ,   
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R" ,   
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R" , 
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G" ,   
   
    
}
```


```python
# Translate RNA to protein

protein_sequence = " "
for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
         amino_acid = codon_table[codon]
    if amino_acid == "*":
            break
    protein_sequence += amino_acid
```


```python
# Prompt the user to enter output file name

output_file_name = input("Enter the name of the output file:")

```

    Enter the name of the output file: BRCA1.txt



```python
# Save the protein sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name} ")
```

    The protein sequence has been saved to BRCA1.txt 



```python
print(protein_sequence)
```

MDLSALRVEEVQNVINAMQKILECPICLELIKEPVKSCDRMLKYLEEKVKKNYPFCNRAGVCVVCLKHLLSKKKC
KECRSRVKNCVPKLKRCRQYDSLLHLSQECMYACCLTEYALKHIKDCRKCYRCNSTESLKSYILEECLKYNKTSN
HRDTIHADLSGDKPKQLLQDDGAVESIDSSPFTFNNEHSGERKPKLPDLSLQTEFFSPEHPLLSSTEFHFGLSIS
ILKDSEDLLNTEEQIYTLKDLILEESDVTPLSLFSSRQGVTLSPVFKPPLGPASANSGSYDTTSKQLLNLADLMF
SEYGQNEPHEAVADFVFDKFSSLPIQVNHKQVCQELLRQTSSTGELQDPHALKAKEPTSEVDESTATLLGRQST
SRQNSTHEQVYLESIPRLNSVLQVRGLQTPVFYVLPNRTLSERKIFLTEQSGNESLKLPCVGKLVFVDESGEGID
LPDQPLQPHIQTMDKNSADLQLDETLQKMQALNGYDVRDSLPATIQESFYLHTPSPSAVKKQRGDLVDDTEFTKQ
TEATKQNGLEGADEDEDTAKSSSSENDDDMDTSNANLQQNALEDNESNLSEDEINDYDDQQSQDDIYDDGNEDHS
APELLSSSPVKLSSCDSSSNDAQLGTEPQGNAQEVTTSGNLGQTPPNTSTEALDVKIDFLSSVNAWAEHAPRTQA
LTSSGSSTSQNSEKSPSAKEESSKSSKLDSSTSRSLQASLTEETPKAKTSLQEKLNDCLEREDQTPVTDLQTSST
HRGLPSHPRDPLSNESVAPTYQESRQALSEPALGRASSPPAQAHLPSPRFPVLSATGTTDSAGSTTQKSTSSSGR
SSAPNGLLSRPEPSGASPLTSTLSCSSSVESTSVRTHSTGVASTSSSSDSITQDSGPVHLKKRTSLLHLPPLLCQ
FVFHPVSSLVTPSHTLTSRSGPSTSAVYSQPQPIVLPPYYQILTPLHVAHEVALGGKDRKDSRRPSEPLEESLPH
PLPKVVEALFPADSGHADMVSSVSTSQSVFPSSSGKGVNTLLRKLSESSPGLVANLEPVIAQPTQQEVSRGLKNN
IKASLQEEIEALLATYHLLVQEKSNYVGIAAVSKIMQLSERGLPYEGVLYSRKINRTPILKQDFSNPEFFSYLLR
KGIQADDAEADISESGNKSDIIVEESFSKYYNKKNTEIIRQEKITVLRKTELLTKSQSDKEVHTNSNEYPEPIQE
SAFPPLCLPTLSPENEAGLELNGYDDFDVLPTLEQVVPLYREQLQDLLHDFTSKQTSLESCGTSNSQLCYNSSVL
HKQWDDNTTEGKKLVLDDISTFKGLRPLNTSQCPAPISNPSTSNPPPASTSLTADPSQLLRGELVSTDASTVNSI
LVPLLFKMDEQELSAMMPDGLFPSQQISESHHILAKQSYFPLSYLWLSWLSPELSHKSSPCSRHFLQSRLFEDLD
DVVQQLAAHRAQALWPFHTPMEEDELDDFTFLQSNSPGVNLRDFNGLNELEVLNLQHLTKILHDLRNLQEQDTYL
SCSLTTTQLSPSCSNRNLIEERELRQEIVLTLTSHYRENNGQPGENFELHIQKGYEPLNSSSREAVLEDFEMFRQ
FFWLILNPLTISHNEDSGVHTNGRDDGHLIHLRQINSGHSLLHKLFGGTEATFNTPEVDMRFPCYWNNNRANVAQ
CLLERLHYNPTAVSSFTNIDWNNVVLQNGLDTLRTEKSTLQLVEHYSSSQSVSNSQSPAIPEERDPVHLDPRPHV
DEHPSSQSPRPSSQSSSSSSSSSESDESSNSPTSRPATTAVSPSASSQKRQDDSTSAEDSEIDNNYDKMSNKSDY
SAADADGDYNTWEDEDIIRPRRNPASLSQRQTFSIIISGLKMESNPEHIAPSPASASSSQPESTPLNSQSNGAKK
ISGPKAKRSGSSSSQTPATQQRPESDQGLVSEDEEDEDQLTEDEISIMDDLLRKQGAGDLASDFVAVQSQLRAAL
KSIMRKEADTSKLIFENSHGLKESIMSGVRLIPEIIRNESEFFTTLSKVKLYAMHQLKAVTSCEQSTQELTSPTE
NSDYLECVLKVLGEDVVNVHTGTQFSVGKLRLDCFSQEGEAEFKDVIIANDNEDGYPVRKELPYKEAFKVKKNVS
TDRPFIVSLGVGLSHGSLLVLKDLFSAENYFQSPHTLEEDFVNYGVQKVVTTDFYTSKAFGLGPVYVRDEALELV
NVLTKKHRNRVTCMEEMTLRLVKTALSICTPSIPQAEESLLSEAQRLNALFGFRNLKTLLALETSLAQISYIEYE
VSKKMRGKRTRAGKDRNKQVLHTGNSALLNDEESGSEQLHSSQPSVNIAQWSSILSTIYPENGIMTDKLHLRENN
GTLGSQESIVLSKQELNVPGIAIVTKSVPGLQECPICRLIPELVKIIRRSCTVESPSLLTLQGLVDFHYESEVVV
TTTGKVVLSRKDNYELIEEDIVQAHVIRKVLERSMSLSLTAHAPLAHKIFKHLRNSRNTTSNYELMERKRPKRLL
KRMKVLPPPPPSSEPVVPYPQLSPSASDNGNGLQNKERKKTGGGSNRASENAKVSQRPNSQPGTYQVLKTLEALS
ELVVKYFYSTIRTKYPQRVFVTSPGPQLGSSPRHINPQPPSAYSSGSLSWKLPLLASSRPGILRRKLPTEELDFV
FIAAPVDLSASTISRSVPSRVYNHVLERVFTSEKTTPVLLAQARPPHRHSQTEECESLVKLELRILGTVAVESHP
VDESSNFFPPLALPPSQSGYRSYQCPQEPLRQIKEELCRTVEDGKIFIGKILKHPDYIQNVRPITLYVQNTSKSD
PYHVNTFPPLSPPDDKEFTRQPTRYLELCALNPLDFTLYGQSHHDGEDNNLSNVCGVESNEVSLADSCISNEEDT
SEDEIDLEMDNEQNQEEIEQLERRKNKLQEERKKILESIINELKDQKPRISGSRRGEICSDYAIPLLSNCDFVGG
IQSDVDPPLKSTFSKKQSTNSFSVVKISKEEDVKLLEKIEKNLEIFQELVLDLNDAENLKKVCEVYLHYQSGWKI
FFKGLDTLQKSDISIKEEVTKIGQHEGDIQDSLYGASEKPQLHIVEALKLSAKRTWHSIHDLVPRGRKAKGKPSE
VLLEPDHYQKIFVNTQVYQKTQILRLHFDGRSCSVHEDERLEELKHLTEAQFLELDPENFQPRKYYGKFVQLVTA
NADLDDSGNFEKRRGDSPTLVVPPESLRFAEHKISESPPTSSQESKGSRSGRTPQISLNRLISLEKILEEDLQLV
EELLHHAEQTTSQQQLKMEQAQNKKKRTESDSGGSAVENGNKASSLFDLAQQPHYYNSEKALSPPSQALSRQQIK
QQQKLNLAQQNQNGSPEDLTTEQLQEIMEKLFSLLADKEEKLAALREQLQAEPSRETKQKNISGSLSRDELLTFH
SLEVFEKYGKFLTPDQKQLDDVEEALSEPLAVCRDPREEPSSSSSDEDQPDLSNRHRSSNNGSESEDPPLVHGVV
EKHRRALRECVQLVKNFNLYYINLNPAYMPGYQGYPHLNSDLLQAFNLYPEIYDIQQLKFLQDICRMYPGWDTPG
YGDRCDTQLASGQQDDGYYKNDSTDHQGVAAHGEHGGSGYTSRYGKDLPK



     



```python

```



## Python Fundamentals

```python
# Any python interpreter can be used as a calculator:
3 + 5  * 4
```




    23




```python
# Lets save a value to a variable
weight_kg= 60
```


```python
print(weight_kg)
```

    60



```python
# Weight0 = upload
# 0weight = invalid
# weight and Weight are indifferent
```


```python
# Types of data
# There are three common types of data
# Integer numbers
# floating point numbers
# Strings
```


```python
weight_kg = 60.3
```


```python
# String compromised of letters
pateint_name = "John Smith"
```


```python
# String compromised of numbers
pateint_id = '001'
```


```python
# Use variables in python

weight_lb = 2.2 * weight_kg

print(weight_lb)
```

    132.66



```python
# Lets add a prefix to our patient id


patient_id = 'inflam_' + pateint_id

print(patient_id)

```

    inflam_001



```python
# Lets combine print statements

print(patient_id, 'weight in kilograms:' , weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# we can call a function inside another function

print(type(60.3))


print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# we can also do calculations inside the print function

print('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66



```python
weight_kg = 65.0
print( 'weight in kilograms:', weight_kg)
```

    weight in kilograms: 65.0



```python

```


```python

```


## Multiple Files

```python
import glob
```


```python
print(glob.glob('inflammation*.csv'))
```

    ['inflammation-10.csv', 'inflammation-09.csv', 'inflammation-11.csv', 'inflammation-06.csv', 'inflammation-05.csv', 'inflammation-08.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-04.csv', 'inflammation-03.csv', 'inflammation-02.csv', 'inflammation-12.csv']



```python
import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
fig = matplotlib.pyplot.figure(figsize =(10.0, 3.0))

axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('average')
axes1.plot(numpy.mean(data, axis = 0))

axes2.set_ylabel('max')
axes2.plot(numpy.amax(data, axis = 0))

axes3.set_ylabel('min')
axes3.plot(numpy.amin(data, axis = 0))

fig.tight_layout()
matplotlib.pyplot.show()
```

    inflammation-01.csv
    inflammation-02.csv
    inflammation-03.csv



![png](output_2_1.png)



```python

```


```python

```









 




