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

# Word Cloud using WordCloud library
def word_cloud():
    # A string variable containing random words
    text = "Python Data Science Visualization WordCloud Example"

    # width=800 - sets the width of the word cloud image to 800 pixels
    # height=400 - sets the height of the word cloud image to 400 pixels
    # background_color='white' - sets the background color of the word cloud to white
    # interpolation='bilinear' - uses bilinear interpolation to smooth the edges
    # of the words in the displayed image.
    # plt.axis('off') - hides the axis lines and labels from the plot
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')
    plt.title('Word Cloud Example')
    st.pyplot(plt)
    plt.clf()

st.header("Word Cloud Demo")
word_cloud()

# 3D Surface Plot using Matplotlib
def surface_plot():
    # creates a new figure object as the container for the plot
    fig = plt.figure()

    # adds a subplot to the figure with 3D projection.
    # "111" means 1 row, 1 column, and this is the 1st subplot
    ax = fig.add_subplot(111, projection='3d')

    # Creates an array "X" and "Y" both with 100 evenly spaced values between -5 and 5
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)

    # Creates a meshgrid from "X" and "Y" arrays for 3D plotting
    X, Y = np.meshgrid(X, Y)

    # Calculates the "Z values" using the formula sin(sqrt(X^2 + Y^2))
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Creates a 3D surface plot using "X", "Y", and "Z" arrays
    # cmap="viridis" - sets the color of the map
    # other colors: plasma, inferno, magma, cividis, Greys, Blues, Greens, Oranges, Reds, etc.
    ax.plot_surface(X, Y, Z, cmap='viridis')

    plt.title('3D Surface Plot Example')
    st.pyplot(plt)
    plt.clf()

st.header("3D Surface Plot Demo")
surface_plot()

# Network Graph using NetworkX

import networkx as nx

def network_graph():
    # Generates a random graph using Erdős-Rényi model
    # 10 - the number of nodes in the graph
    # 0.5 - probability of an edge existing between any two nodes
    G = nx.erdos_renyi_graph(10, 0.5)

    # Calculates the position of nodes in the graph using force-directed algorithm
    pos = nx.spring_layout(G)

    # Draws the graph
    # G - the graph object
    # pos - the positions of nodes
    # with_labels=True - displays node labels
    # node_color='skyblue' - sets the color of nodes
    # node_size=1000 - sets the size of the nodes
    # edge_color='gray' - sets the color of the edges to gray
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray')

    plt.title('Network Graph Example')
    st.pyplot(plt)
    plt.clf()

st.header("Network Graph Demo")
network_graph()

# Sankey Diagram using Plotly

import plotly.graph_objects as go

def sankey_diagram():
    # go.Figure - Creates a Plotly figure object to hold the Sankey Diagram
    # go.Sankey - Creates a Sankey diagram object within the figure
    # node - Defines the properties of the nodes in the diagram
      # pad - Specifies the padding between nodes (15 pixels)
      # thickness - Sets the thickness of the nodes (20 pixels)
      # line - defines the appearance of the border around nodes
        # color - sets the border color to "black"
        # width - sets the border width to 0.5 pixels
      # label - provides the label for the nodes (A, B, C, and D)
    # link - Defines the connections between the nodes

      # target - specifies the target nodes of the links (using indices, 1, 2, 3)
      # value - sets the flow values for each link (8, 4, 2)
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=["A", "B", "C", "D"]),
        link=dict(source=[0, 1, 0], target=[1, 2, 3], value=[8, 4, 2])
    ))

    fig.update_layout(font_size=10)
    return fig

st.header("Sankey Diagram Demo")
fig = sankey_diagram()
st.plotly_chart(fig)


########## USING ACTUAL DATASET

st.title("Using an actual dataset 📊")

st.markdown("""
Throughout the previous code blocks, we were dealing with randomly generated values as our dummy data but how do we deal with actual datasets?  
  
**Dataset:** Titanic Dataset (Kaggle)  
https://www.kaggle.com/datasets/brendan45774/test-file/data  
  
`NOTE:` Check the slides first **"Lesson 4 - Data Visualization Fundamentals"** to see how to upload local files to Google Colab.

""")

# Read our CSV dataset.
df = pd.read_csv("datasets/tested.csv")

st.write(df)

st.write(df.info()) # Show the relevant information in the dataset such as Data Types
st.write("Show the relevant information in the dataset such as Data Types")

st.write(df.isna().sum()) # Show the null values
st.write("Show the null values")

st.write(df.describe()) # Generate Descriptive Statistics
st.write("Generate Descriptive Statistics")

st.markdown("""

`count` Number of non-null or non-missing values in a column.  
`mean:` The calculated mean or average of the column.  
`std (Standard Deviation):` Values in this row shows how much the values in their respective columns deviate from the mean (average value)  
`min:` Smallest (minimum) number in the column.  
`25% (1st Quartile):` This is the 25th percentile. The values represented in this row implies that 25% of the data points based on their respective rows are less than or equal (<=) its value.  
`50% (Median):` This is the 50th percentile. The values represented in this row are below (<) its value and half (50%) are above (>).  
`75% (3rd Quartile):` This is the 75th percentile. The values represented in this row are less than or equal (<=) to its value.  
`max:` Largest (maximum) number in the column.  

""")

st.write("""

Based on the dataset provider these are the description for the columns.  
  
`PassengerId`  
Passenger number  
`Survived`  
0 = Dead 1 = Alive  
`Pclass`  
1 = First class 2 = Second class 3 = Third class  
`Name`  
Name of passenger  
`Sex`  
Gender  
`Age`  
Age of passenger  
`SibSp`  
Number of siblings  

### `Observations`

We can see that the categorical values are:
**pclass**, **sex**, and **survived**

Numerical values are:
**age**, **fare**, **sibsp**, and **parch**

""")

st.write(df['Survived'].value_counts()) # Show the count of the survived and deceased passengers
st.write("Show the count of the survived and deceased passengers")

total_passenger_count = 266 + 152

st.write(f"Total Passenger Count: {total_passenger_count}")


# We can use pie chart to show the margin between the survived and the deceased passengers
def pie_chart_survived_deceased():
    survived = df['Survived'].value_counts()
    colors = ['salmon', 'lightgreen']
    plt.pie(survived, labels = ['Deceased', 'Survived'], autopct='%1.1f%%', colors=colors)
    plt.title('Titanic Survival Rate')
    st.pyplot(plt)
    plt.clf()
    
st.header("Passengers Survival Rate")
pie_chart_survived_deceased()
st.markdown("We can observe the proportion of passengers who survived and those who did not through our pie chart. There's a **36.4%** survival rate.")

# We can use pie chart to show the margin between the passengers based on their class
def pie_chart_passenger_based_on_sex():

    sex = df['Sex'].value_counts()
    colors = ['skyblue', 'pink']
    plt.pie(sex, labels = ['male', 'female'], autopct='%1.1f%%', colors=colors)
    plt.title('Titanic Passenger Sex Distribution')
    st.pyplot(plt)
    plt.clf()
    
st.header("Passengers Sex Distribution")
pie_chart_passenger_based_on_sex()
st.markdown("Amongst the passengers, majority were **Male accounting for 63.6%** of the total passengers (266 total). **Female passengers on the other hand are 36.4%** of the total passengers (152 total).")

# We can use pie chart to show the margin between the passengers based on their class
def pie_chart_classes_distribution():

    survived = df['Pclass'].value_counts()
    colors = ['skyblue', 'lightgreen', 'salmon']
    plt.pie(survived, labels = ['3rd', '1st', '2nd'], autopct='%1.1f%%', colors=colors)
    plt.title('Titanic Classes Distribution')
    st.pyplot(plt)
    plt.clf()
    
st.header("Passenger Classes Distribution")
pie_chart_classes_distribution()
st.markdown("Based on the Titanic Classes Distribution pie chart, we can see that majority of the passengers are **Class 3 accounting for 52.2%** of the total passengers. This is followed by **Class 1 passengers with 25.6%**, and **Class 2 with 22.2%**.")

# How about the survival based on sex? We can display that using Bar Chart
def bar_plot_survival_based_on_sex():
    plt.figure(figsize=(7, 7)) # define the width and height of the graph

    ax = sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title('Passenger Sex vs. Survival')

    # You can use this to show the actual count of each bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    st.pyplot(plt)
    plt.clf()
        
st.header("Passenger Sex vs Survival")
bar_plot_survival_based_on_sex()
st.markdown("It is evident from the bar graph that **Female passengers had the highest survival rate** (152) while **none of the male passengers survived** the tragic event, 266 deceased, and 0 survivors.")


# Now we use a bar chart to show the survival rate of the passengers based on their class
def bar_plot_survival_based_on_class():
    sns.countplot(x='Pclass', hue='Survived', data=df)
    plt.title('Passenger Class vs. Survival')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    st.pyplot(plt)
    plt.clf()
    
st.header("Passenger Class vs Survival")
bar_plot_survival_based_on_class()

# Add double spaces ("  ") after the line you want to add a line break.
st.markdown(""" 
1 - First Class  
2 - Second Class  
3 - Third Class  
  
We can see from the observations that the **third class** passengers had the **highest deceased rate** compared to the other 2 classes **but they also had the highest survival rate** amongst the 2 other classes.  
  
**First-class** passengers tend to **have higher proportion of survivors** compared to the other two classes. This may reflect the socio-economic factor in survival chances.
""")

# Let's check how the Age of the passengers are distributed using Histogram
def histogram_age_distribution():
    sns.histplot(df['Age'], bins=30, color="salmon")
    plt.title('Passenger Age Distribution')
    plt.xlabel('Passenger Age')
    plt.ylabel('Count')
    st.pyplot(plt)
    plt.clf()
    
st.header("Passenger Age Distribution")
histogram_age_distribution()
st.markdown("We can see from the graph that **peak passenger age ranges between 20 and 30**. It is also notable that there are **10 infants in in the passengers** which are below 1 year of age.")

st.header("Conclusion")
st.subheader("Insights from our Data Visualization and Data Analysis: 📊")
st.markdown("""

1. **Survival Rate:**  
- Survival rate of the passengers is fairly low accounting for **36.4%**  
- Almost **two-thirds** of the passengers did not survive.  
  
2. **Gender Distribution and Survival:**  
- Majority of the passengers were "Male" which is made up of **63.6% (266)** of the total passengers. Females on the other hand accounts for **36.4% (152)** of the total passengers  
- Even though majority of the passengers are "Male", "Female" passengers had higher survival rate **152** compared to **0** survival rate of male passengers. This can be related to the evacuation policies such as "women and children first".  
  
3. **Class Distribution and Survival:**  
- The largest proportion of passengers belonged to the third class comprising of **52.2%** based on the total passengers. This is followed by first class with **25.6%**, and **22.2%** in second class.  
- Another important observation is that **third-class passengers tend to have the highest total deceased rate** but **they also had the highest absolute number of survivors** which highlights the class disparity in survival.  
  
4. **Age Distribution:**  
- The age of the passengers **peaks around between 20 to 30 years old**.  
- **10 infants** were also on-board who were under 1 year of age. This indicates a wide range of age among the passengers  
  
5. **Survival by Class:**  
- **1st class passengers** tend to have the **higher proportion of survivors** after comparing them to the other two classes, this may be attributed to the socio-economic factor playing a vital role in the survival chances during the event.  
- **3rd Class passengers*** had the **highest deceased rate** after comparing them to the other two classes. Despite this, they also had the highest survival rate.  

""")