import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import squarify

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

heatmap_data = np.random.rand(10, 10) # 10x10 matrix

bubble_size = np.random.rand(100) * 100 # generating the bubble sizem random values between 0 to 100 * 100

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
    st.pyplot(plt)
    plt.clf()

st.header("Scatter Plot Demo")
scatter_plot()

# Histogram using Matplotlib
def histogram():
    # bins - define the intervals into which your data is divided or counted
    # color - defines the actual color of the histogram bar
    # edgecolor - defines the stroke color of the bar
    plt.hist(x, bins=10, color='gray', edgecolor='black')
    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    plt.clf()

st.header("Histogram Demo")
histogram()

# Heatmap using Seaborn
# Note: If annotations are only visible on the first row,
# try upgrading your Seaborn library using `pip install seaborn --upgrade`

def heatmap():
    # annot=True - defines the values of the heatmap displayed on the plot
    # cmap - sets the color scheme to "coolwarm",
    # other colors: viridis, plasma, inferno, magma, cividis, Blues, Greens, Reds, etc.
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
    plt.title('Heatmap Example')
    st.pyplot(plt)
    plt.clf()
    
st.header("Heatmap Demo")
heatmap()

# Box Plot using Seaborn
def box_plot():
    # in the data we define x and y variables for the x and y data
    # palette - specifies the color used in the box plot
    # other palette colors: Set1, Set2, pastel, muted
    sns.boxplot(data=[x, y], palette="Set3")
    plt.title('Box Plot Example')
    st.pyplot(plt)
    plt.clf()

st.header("Box Plot Demo")
box_plot()

# Area Chart using Altair
def area_chart():
    df = pd.DataFrame({'Date': dates, 'Value': trend})
    # alt.Chart(df) - initializes the Altair chart object by using the DataFrame(df) as data source
    # mark_area(opacity=0.5) - specifies that the chart should be an area chart with 50% opacity
    # encode(...) - defines how the data should be mapped to visual properties of chart(channels)
      # x="Date:T" - maps the Date column to the x axis, using temporal (T) data type
      # y="Value:Q" - maps the Value column to the y-axis using quantitative (Q) data type
    # .properties(title='Area Chart Example') - sets the title of the chart.
    # st.altair_chart(area_chart) - renders the chart for the output using Streamlit
    area_chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x='Date:T',
        y='Value:Q'
    ).properties(title='Area Chart Example')
    st.altair_chart(area_chart, use_container_width=True)

st.header("Area Chart Demo")
area_chart()

# Bubble Chart using Matplotlib
def bubble_chart():
    plt.scatter(x, y, s=bubble_size, alpha=0.5, color='green')
    plt.title('Bubble Chart Example')
    plt.xlabel('X')
    plt.ylabel('Y')
    st.pyplot(plt)
    plt.clf()

st.header("Bubble Chart Demo")
bubble_chart()

# Treemap using Squarify
def treemap():
    # squarify.plot(...) - used to create the treemap
    # sizes=values - assigns the values data to determine the sizes of rectangles in the treemap
    # label=categories - assigns the labels to the rectangles
    # color - defines the color of the rectangles
    # alpha=0.7 - controls the transparency of the rectangles
    # 0 is fully transparent and 1 is opaque
    # plt.axis('off') - hides the x and y axis of the plot
    squarify.plot(sizes=values, label=categories, color=['red', 'green', 'blue', 'orange'], alpha=0.7)
    plt.title('Treemap Example')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

st.header("Treemap Demo")
treemap()

# Violin Plot using Seaborn
def violin_plot():
    # palette - specifies the color used in the box plot
    # other palette colors: Set1, Set2, Set3, pastel, and muted
    sns.violinplot(data=[x, y], palette="Set3")
    plt.title('Violin Plot Example')
    st.pyplot(plt)
    plt.clf()

st.header("Violin Plot Demo")
violin_plot()