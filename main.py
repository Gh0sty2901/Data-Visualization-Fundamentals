import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D

st.title('Data Visualization Fundamentals - Lesson 4')
st.write('by JC Diamante')

# Data Generation (Dummy Data)

# defining a seed ensures that random number generated are the same every time.
np.random.seed(0)

# Pre-defined list containing the categories
categories = ['A', 'B', 'C', 'D']

# Generates an array named "values" containing four random integers
# between 10(inclusive) and 100(exclusive)
values = np.random.randint(10, 100, size=4)

# Creates a DatetimeIndex named "dates" containing a sequence of
# 10 dates starting from January 1, 2024
dates = (pd.date_range('2024-01-01', periods=10)).strftime('%Y-%m-%d').tolist()

# Generates an array of 10 random numbers from a standard normal distribution
# Calculates the cumulative sum of these numbers and assign to "trend" variable
trend = np.random.randn(10).cumsum()

# Generates an array named "x" and "y", each containing
# 100 random numbers from a standard normal distribution
x = np.random.randn(100)
y = np.random.randn(100)

# Bar Chart using Matplotlib
def bar_chart():
  colors = ['skyblue', 'lightgreen', 'salmon', 'orange']  # you can define colors using a list

  # defines the cateogries, values, and the color for our chart
  plt.bar(categories, values, color=colors)
  # this displays a Title for our chart
  plt.title('Bar Chart Example')
  # this defines the label for the y axis of our chart
  plt.ylabel('Value')
  # this defines the label for the x axis of our chart
  plt.xlabel('Categories')
  # this shows the graph
  st.pyplot(plt)
  # Clears the current figure
  plt.clf()

st.header("Bar Chart Demo")
bar_chart()

# Line Chart using Matplotlib
def line_chart():
    # Defines the data used for our line chart, the markers used, line style, and the color
    plt.plot(dates, trend, marker='o', linestyle='-', color='b')
    plt.title('Line Chart Example')
    plt.xlabel('Date')
    plt.ylabel('Trend Value')
    # Rotates the dates shown in the x axis label (45 degrees) for readability
    plt.xticks(rotation=45) # rotate in 45 degrees
    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

st.header("Line Chart Demo")
line_chart()

# Pie Chart using Matplotlib
def pie_chart():
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange']  # you can define colors using a list

    # autopct defines the wedges or the numeric values shown in the pie chart
    plt.pie(values, labels=categories, autopct='%1.1f%%', colors=colors)
    plt.title('Pie Chart Example')
    plt.show()
    st.pyplot(plt)
    plt.clf()

st.header("Pie Chart Demo")
pie_chart()

# Scatter Plot using Matplotlib
def scatter_plot():
    plt.scatter(x, y, color='purple')
    plt.title('Scatter Plot Example')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    st.pyplot(plt)
    plt.clf()

st.header("Scatter Plot Demo")
scatter_plot()