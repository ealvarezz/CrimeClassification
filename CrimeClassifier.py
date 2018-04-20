
# coding: utf-8

# # CSE391 
# ## Assignment 2
# ##  Due: April 20th at 11:59 pm
# 
# Data Science 
# 
# 
# 
# 
# 
# This dataset provides crime reports from San Francisco's neighborhoods in 2014. Please explore the data and check what is inside.
# 
# Tasks:
# 1.	Predict the category of the crime based on the time and location information.     
# 
# 
# What to use:
# You can use any models (regressions, classification, clustering and/or their combinations). 
# Please use  Python.  
# 
# 
# Data fields Description:
# - Dates - timestamp of the crime incident
# - Category - category of the crime incident. This is the target variable you are going to predict. Descript - - detailed description of the crime incident
# - PdDistrict - name of the Police Department District
# - Address - the approximate street address of the crime incident 
# - X - Longitude
# - Y - Latitude
# 
# 
# 

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import colorsys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss 
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets

plt.style.use(['seaborn-whitegrid','seaborn-bright'])

data = pd.read_csv('/home/ealvarez/Documents/SBU/CSE 391/data_set.csv',parse_dates = ['Dates'])
data.dropna(inplace=True)
print(data.shape)
data.head(10)


# ## Some visualization on the data to see how it looks!

# ## As we can see we have too many crimes that barely appear in the spectrum and are barely noticeable. We can group the crimes that barely appear into one group.

# In[20]:


# We select the row where category appears less then 700 times and set the category to 'OTHER OFFENCES'
data.loc[data.groupby('Category')['Category'].transform('size') < 700,'Category'] = 'OTHER OFFENSES'

encoder = preprocessing.LabelEncoder()                 # Encoder used to turn categories into integer values

categories = encoder.fit_transform(data.Category)      # Turning them into numerical values
categories = pd.DataFrame(categories)                  # Encode the categorie data
X = pd.DataFrame(preprocessing.scale(data.X))          # Normalize the X values
Y = pd.DataFrame(preprocessing.scale(data.Y))          # Normalize the X values

new_data = pd.concat([categories, X,Y], axis=1)         # We make a new dataframe with our new values
new_data.columns = [ 'categories', 'X', 'Y']

shrink = new_data.sample(frac=0.05)           # Here we can see how the categories are scattered over the X and Y axis but we take a smaller sample 

plt.figure(figsize=(12,8))
#for i in range(15):
#    cur = colorsys.hsv_to_rgb(np.random.uniform(low=0.0, high=1), np.random.uniform(low=0.2, high=1),np.random.uniform(low=0.9, high=1))
#    plt.scatter(shrink[shrink['categories'] == i].X, shrink[shrink['categories'] == i].Y, c=cur, edgecolor='none')
plt.scatter(shrink.X, shrink.Y, c=shrink.categories, edgecolor='none', alpha=0.2, s=150, cmap=plt.cm.get_cmap('viridis', 15))
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Reduced Crimes')
plt.show();


# ## I want the data to have numerical values since I want to use the Naive Bayes algorithm as a classifier, for now I'll strip away the address and description data since I would have to parse them to convert them into a meaningful numerical value, I got four other homeworks and the final project due by next week so I'm good =D
# -  For the rest of the string values I will turn them into binary values.
# -  For the date I will extract the day, month and hour of the say (turn month and day into binary data).
# -  Last but not least I think it will be wise to normalize the X and Y values.
# -  For the label which is the category, I will convert it to numerical values using an encoder.

# In[21]:


# Here we will extract the values from Dates
month = pd.get_dummies(data.Dates.dt.strftime("%B"))   # Extract month by string then binary
day = pd.get_dummies(data.Dates.dt.strftime('%A'))     # Day
hour = pd.get_dummies(data.Dates.dt.hour)              # Hour
district = pd.get_dummies(data.PdDistrict)

# I get the names of the columns so that I can plot things more easily
month_names = list(month) 
day_names = list(day)
district_names = list(district)

# Reseting the index helps concatinate because we know that this point the number of rows and order are the same 
new_data.reset_index(drop=True, inplace=True)
month.reset_index(drop=True, inplace=True)
day.reset_index(drop=True, inplace=True)
hour.reset_index(drop=True, inplace=True)
district.reset_index(drop=True, inplace=True)

# I concatiate all my numerical data and now we can input this to the model
new_data = pd.concat([new_data, month, day, hour, district], axis=1)
new_data.head(10)


# ## Now that we have this data we can easily make different type of charts to better visualize the data

# In[22]:


table = pd.pivot_table(new_data, index=['categories'], values=district_names, aggfunc=np.sum)
ax = table.plot(kind='barh', title ="Crime Per District", figsize=(25, 10), legend=True, fontsize=12,stacked=True)
plt.show()


# In[23]:


table = pd.pivot_table(new_data, index=['categories'], values=month_names, aggfunc=np.sum)
ax = table.plot(kind='barh', title ="Crime Per Month", figsize=(25, 10), legend=True, fontsize=12, stacked=True)
plt.show()


# ## These are some function to help with the metric

# In[24]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def show_confusion_matrix(y_test, y_pred, class_names):    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    plt.figure(figsize=(13,9))
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(13,9))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


# # Naive Bayes

# ## Now we can divide out data into training and testing data and train our model using a Naive Bayes classifier.

# In[25]:


class_names = ["1", "2",'3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15']
labels = new_data.iloc[:, 0]
features = new_data.iloc[:, 1:]

classifier = BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
predicted_prob = np.array(classifier.predict_proba(X_test))

show_confusion_matrix(y_test, y_pred, class_names)


# In[26]:


from pandas_ml import ConfusionMatrix
y_actu = y_test

cm = ConfusionMatrix(y_actu, y_pred)
cm.print_stats()


# In[27]:


log_loss(y_actu, predicted_prob) # This is the log loss for this model


# # Logistic Regression
# ## Let's try Logistic Regression because why not!!!

# In[28]:


classifier = LogisticRegression(C=.01)
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
predicted_prob = np.array(classifier.predict_proba(X_test))

show_confusion_matrix(y_test, y_pred, class_names)


# In[29]:


y_actu = y_test

cm = ConfusionMatrix(y_actu, y_pred)
cm.print_stats()


# In[30]:


log_loss(y_actu, predicted_prob) # This is the log loss for this model


# ## Logistic Regression has a better accuracy than Naive Bayes of 0.25 and a lower log loss of 2.23. F1 and FPR for each class are shown in the confusion matrix stats.
