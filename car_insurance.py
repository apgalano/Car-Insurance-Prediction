'''
This is a classic logistic regression problem, where given a series of features
I am tasked to create a model predicting if a certain customer is likely to purchase
car insurance, denoted by a binary variable (1 is yes, and 0 is no). 
Hence, I opt for a simple logistic regression model instead of a more complex approach,
e.g. a neural network, which I would use if for instance I had a multi-class 
classification problem.
'''
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

train_data_raw = pd.read_csv('carInsurance_train.csv')
test_data_raw = pd.read_csv('carInsurance_test.csv')
working_dir = os.getcwd()

def produce_visuals():
    '''
    The purpose of this function is to plot the distributions of features of the dataset,
    as well as their relationship with the prediction variable CarInsurance.
    We can visually inspect which features have a positive or negative correlation with
    CarInsurance and spot outliers that need to be removed later on.
    '''
    fig_dir = '/Figures/'
    
    if not os.path.exists(working_dir+fig_dir):
        os.makedirs('Figures')
    os.chdir(working_dir+fig_dir)
    
    # Analyze numerical data

    sns.kdeplot(data['Age'])
    plt.savefig('age_hist.jpg', bbox_inches='tight')
    plt.clf()
    sns.kdeplot(data['Balance'])
    plt.savefig('balance_hist.jpg', bbox_inches='tight')
    plt.clf()
    sns.kdeplot(data['NoOfContacts'])
    plt.savefig('contacts_hist.jpg', bbox_inches='tight')
    plt.clf()
    sns.kdeplot(data['DaysPassed'])
    plt.savefig('dayspassed_hist.jpg', bbox_inches='tight')
    plt.clf()
    sns.kdeplot(data['PrevAttempts'])
    plt.savefig('attempts_hist.jpg', bbox_inches='tight')
    plt.clf()
    sns.kdeplot(data['CallDuration'])
    plt.savefig('duration_hist.jpg')
    plt.clf()
    
    age_plot = sns.lmplot(x='Age',y='CarInsurance', data=data, fit_reg=False)
    age_plot.savefig('age_vs_car.jpg', bbox_inches='tight')
    balance_plot = sns.lmplot(x='Balance',y='CarInsurance', data=data, fit_reg=False)
    balance_plot.savefig('balance_vs_car.jpg', bbox_inches='tight')
    contacts_plot = sns.lmplot(x='NoOfContacts',y='CarInsurance', data=data, fit_reg=False)
    contacts_plot.savefig('contacts_vs_car.jpg', bbox_inches='tight')
    dayspassed_plot = sns.lmplot(x='DaysPassed',y='CarInsurance', data=data, fit_reg=False)
    dayspassed_plot.savefig('dayspassed_vs_car.jpg', bbox_inches='tight')
    attempts_plot = sns.lmplot(x='PrevAttempts',y='CarInsurance', data=data, fit_reg=False)
    attempts_plot.savefig('attempts_vs_car.jpg', bbox_inches='tight')
    duration_plot = sns.lmplot(x='CallDuration',y='CarInsurance', data=data, fit_reg=False)
    duration_plot.savefig('duration_vs_car.jpg')
    
    '''
    From the scatterplots we can see spot a few outliers in Balance and PrevAttempts.
    Also, we see that e.g. DaysPassed probably positively predicts CarInsurance, 
    while NoOfContacts probably negatively predicts CarInsurance.
    '''
    
    
    # Analyze categorical data
    
    '''
    I firstly plot the distribution of the prediction variable to see if the 
    dataset is balanced. About 40% of samples are from successful insurance sales
    and the rest is from unsuccessful ones. It is not exactly balanced but not too
    bad either. Removing some of the unsuccesful sales samples would reduce 
    the already small dataset so I did not intervene. Next I plot the distribution
    of the other categorical features, stacking them with respect to the 
    CarInsurance variable in order to spot potential predictive indicators.
    '''
    plt.clf()
    data['CarInsurance'].value_counts().plot(kind='bar')
    plt.xlabel('Insurance')
    plt.ylabel('Count')
    plt.savefig('insurance.jpg', bbox_inches='tight')
    plt.clf()
    df1 = data.groupby('Job').apply(lambda x: x['CarInsurance'].value_counts())
    df1 = df1.unstack().fillna(0)
    df1.plot.bar(stacked=True)
    plt.xlabel('Job')
    plt.ylabel('Count')
    plt.savefig('job_hist.jpg', bbox_inches='tight')
    plt.clf()
    df2 = data.groupby('Marital').apply(lambda x: x['CarInsurance'].value_counts())
    df2.plot.bar(stacked=True)
    plt.xlabel('Marital')
    plt.ylabel('Count')
    plt.savefig('marital_hist.jpg', bbox_inches='tight')
    plt.clf()
    df3 = data.groupby('Education').apply(lambda x: x['CarInsurance'].value_counts())
    df3.plot.bar(stacked=True)
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.savefig('education_hist.jpg', bbox_inches='tight')
    plt.clf()
    df4 = data.groupby('Default').apply(lambda x: x['CarInsurance'].value_counts())
    df4.plot.bar(stacked=True)
    plt.xlabel('Default')
    plt.ylabel('Count')
    plt.savefig('default_hist.jpg', bbox_inches='tight')
    plt.clf()
    df5 = data.groupby('HHInsurance').apply(lambda x: x['CarInsurance'].value_counts())
    df5.plot.bar(stacked=True)
    plt.xlabel('House insurance')
    plt.ylabel('Count')
    plt.savefig('house_hist.jpg', bbox_inches='tight')
    plt.clf()
    df6 = data.groupby('CarLoan').apply(lambda x: x['CarInsurance'].value_counts())
    df6.plot.bar(stacked=True)
    plt.xlabel('Car loan')
    plt.ylabel('Count')
    plt.savefig('loan_hist.jpg', bbox_inches='tight')
    plt.clf()
    df7 = data.groupby('Communication').apply(lambda x: x['CarInsurance'].value_counts())
    df7.plot.bar(stacked=True)
    plt.xlabel('Communication')
    plt.ylabel('Count')
    plt.savefig('communication.jpg', bbox_inches='tight')
    plt.clf()
    df8 = data.groupby('LastContactDay').apply(lambda x: x['CarInsurance'].value_counts())
    df8 = df8.unstack().fillna(0)
    df8.plot.bar(stacked=True)
    plt.xlabel('Contact Day')
    plt.ylabel('Count')
    plt.savefig('day.jpg', bbox_inches='tight')
    plt.clf()
    df9 = data.groupby('LastContactMonth').apply(lambda x: x['CarInsurance'].value_counts())
    df9 = df9.unstack().fillna(0)
    df9.plot.bar(stacked=True)
    plt.xlabel('Contact Month')
    plt.ylabel('Count')
    plt.savefig('month.jpg', bbox_inches='tight')
    plt.clf()
    df10 = data.groupby('Outcome').apply(lambda x: x['CarInsurance'].value_counts())
    df10 = df10.unstack().fillna(0)
    df10.plot.bar(stacked=True)
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.savefig('outcome.jpg', bbox_inches='tight')
    plt.clf()
    
    '''
    Several interesting points can be made by inspecting the figures.
    Those married or with house/car loan are less likely to buy insurance.
    Also some days and months seem to be particularly good or bad for selling inurance,
    although we do not have balanced samples from each category.
    '''
    
    os.chdir(working_dir)



'''
The dataset is generally in good condition. Most features do not require any processing
as the data types are consistent (both numerical and categorical features)
with the exception of CallStart and CallEnd 
'''

# Analyze and process train and test data together

data_raw = train_data_raw.copy().append(test_data_raw)
data_raw.index = data_raw['Id'] - 1

'''
CallStart and CallEnd cannot be used as features as they stand, e.g. treating them
as categorical data by using only the call hour. Instead what seems to be important
is the call duration that can be easily inferred from the difference of the two.
Next I create the new feature CallDuration which is the difference of 
end and start times, converted to seconds.
'''

frmt = '%H:%M:%S'
CallDuration = []
for index, row in data_raw.iterrows():
    t1 = datetime.strptime(row['CallStart'],frmt)
    t2 = datetime.strptime(row['CallEnd'],frmt)
    dt = t2-t1
    dt_secs = dt.total_seconds()
    CallDuration.append(dt_secs)
    
data = data_raw.copy()
data['CallDuration'] = CallDuration
data = data.drop(['Id', 'CallStart', 'CallEnd'], axis=1)

# Statistical analysis of dataset
data.describe().to_csv('data_description.csv')
    
# Produce visuals for data analysis
produce_visuals()

# Slice the training data
train_data = data.loc[0:3999,:].copy()

# Removing outliers
'''
Following the data analysis from plotting the dataset we can easily spot a few outliers
for some features. I did not use standard technique, e.g. removing samples that are
more than 3 stds away from the mean as the values for some features are dispersed
far from the mean and that would reduce the already small dataset size.
Instead I visually spotted a couple of outliers in the features below and removed them.
'''
train_data.drop(train_data[train_data['Balance'] > 70000].index, inplace = True) 
train_data.drop(train_data[train_data['PrevAttempts'] > 35].index, inplace = True)

# Handling missing values
print(train_data.isnull().sum())
'''
Missing values are all from categorical data.
I treat them by replacing with most balanced category with respect to the 
prediction variable, i.e. CarInsurance, in order to reduce the model's bias.
This results in slightly better accuracy than replacing with the most frequent category. 
For example, Outcome missing values are at 75% and are replaced by 'other'.
'''
train_data['Job'].fillna('management', inplace=True)
train_data['Education'].fillna('tertiary', inplace=True)
train_data['Communication'].fillna('cellular', inplace=True)
train_data['Outcome'].fillna('other', inplace=True)


'''
Next I handle multi-class categorcal features, by creating dummy (binary) variables
for each possible category. Features like CarLoan are already binary and do not
require this step. Also I could drop one of the dummies for each feature, since
it could be implied from the rest of the dummies, but I kept them all to improve
the interpretability of the model at the cost of having a few more features.
'''
# Get dummy variables for categorical features
job_values = pd.get_dummies(train_data['Job'])
marital_values = pd.get_dummies(train_data['Marital'])
education_values = pd.get_dummies(train_data['Education'])
comm_values = pd.get_dummies(train_data['Communication'])
day_values = pd.get_dummies(train_data['LastContactDay'])
month_values = pd.get_dummies(train_data['LastContactMonth'])
outcome_values = pd.get_dummies(train_data['Outcome'])

# Join dummy variables to the training dataframe and remove the original data 
data_dummies = train_data.join(job_values).join(marital_values).join(education_values).join(comm_values).join(day_values).join(month_values).join(outcome_values)
data_dummies.drop(['Job', 'Marital', 'Education', 'Communication', 'LastContactDay', 'LastContactMonth', 'Outcome'], inplace=True, axis=1)

# print(data_dummies.columns)
# Get new column names
col_names_ordered = [          'Age',       'Default',       'Balance',   'HHInsurance',
             'CarLoan',  'NoOfContacts',    'DaysPassed',  'PrevAttempts',
              'admin.',   'blue-collar',  'CallDuration',
        'entrepreneur',     'housemaid',    'management',       'retired',
       'self-employed',      'services',       'student',    'technician',
          'unemployed',      'divorced',       'married',        'single',
             'primary',     'secondary',      'tertiary',      'cellular',
           'telephone',               1,               2,               3,
                     4,               5,               6,               7,
                     8,               9,              10,              11,
                    12,              13,              14,              15,
                    16,              17,              18,              19,
                    20,              21,              22,              23,
                    24,              25,              26,              27,
                    28,              29,              30,              31,
                 'apr',           'aug',           'dec',           'feb',
                 'jan',           'jul',           'jun',           'mar',
                 'may',           'nov',           'oct',           'sep',
             'failure',         'other',       'success',   'CarInsurance']

# Reorder columns so that we have all input features at first, and the prediction variable at the end
data_dummies = data_dummies[col_names_ordered]
input_data = data_dummies.iloc[:,0:-1]
output_data = data_dummies['CarInsurance']

# Scalarize data
scaler = StandardScaler()
scaler.fit(input_data)
input_scaled = scaler.transform(input_data)

'''
Unfortunatelly the test set is not labeled and evaluating the test accuracy is 
impossible using the provided data.
For that reason I have elected to make a 80-20 split on the training set to 
evaluate the testing accuracy and validate no overfitting occurs.
'''
# Split data to train/test 
input_train,input_test,output_train,output_test = train_test_split(input_scaled,output_data,train_size=0.8,shuffle=True,random_state=1)

# Use 10-fold cross-validation to avoid overfitting
regressor = LogisticRegressionCV(cv=10)
# regressor.fit(input_scaled,output_data)
regressor.fit(input_train,output_train)
# accuracy_train = regressor.score(input_scaled,output_data)
accuracy_train = regressor.score(input_train,output_train)
accuracy_test = regressor.score(input_test,output_test)
pred = regressor.predict(input_test)
report = classification_report(output_test,pred)
print('\n'+30*'~')
print(f'Training accuracy is {accuracy_train*100:.2f}%')
print(f'Testing accuracy is {accuracy_test*100:.2f}%')
print('Classification Report:\n')
print(report)
print(30*'~'+'\n')
'''
Output training accuacy when using the entire training set for training is 82.82%.
When performing random splits on the training set, training and testing accuracies
are about the same. Classification report indicates that customers who don't
end up buying insurance are easier to classify. 
'''

'''
Next I create a summary for the model output, where I document each feature's
coefficient in a descending order to highlight some interesting insights of the features.
Positive coefficients have a positive affect on predicting a customer will buy insurance,
and negative ones do the opposite. Those close to 0 are probably inconsequential 
to the model's predictive capabilities. 
'''
# Summary
intercept = regressor.intercept_
coefs = regressor.coef_

summary_table = pd.DataFrame(columns=['Feature'], data = input_data.columns.values)
summary_table['Coefficient'] = np.transpose(coefs)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept',intercept[0]]
summary_table = summary_table.sort_index()
summary_table = summary_table.sort_values('Coefficient', ascending=False)

print(summary_table)
summary_table.to_csv('summary.csv')