### Machine Learning and Statistics - GMIT
#### Jody Bradley (G00387878) | Due date 08/01/2021

----------------

#### Introduction

We have been tasked with creating a web service that uses machine learning to predict winder turbine power output from wind speed values, as in the data set provided ("powerproduction.csv").


#### Prerequisites

Before you continue, make sure that you have met the following requirements:
- You have installed Python 3.8.3
- You have installed the latest version of Anaconda.

#### Technologies
- Python 3.8.3
- Anaconda
- pandas - Python Data Analysis Library
- NumPy
- Matplotlib: Python plotting
- Seaborn: statistical data visualization
- scikit-learn 
- Flask

#### Contents 

In my github repository you will find the following Python scripts [5]:

- power.ipynb (my Jupyter notebook detailing my research and my approach;
- model.py (which develops and trains the linear regression model);
- app.py (which handles the POST requests and returns the results); and
- request.py (which send requests with the features to the servers and receives the results).

#### Running the web service
1. Once you have downloaded the repository, open a new command line, navigate to the folder and run the app.py script (by typing "python app.py" into the command line"
2. Open a new browser page and navigate to http://127.0.0.1:5000/ and the web server will open, where you will be prompted to enter your windspeed.
3. Once you type in your wind speed and hit "predict", the estimated wind power will be displayed below.




