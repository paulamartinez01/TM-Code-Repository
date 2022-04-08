# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:03:12 2021

@author: Pau
"""

# PREPARING THE FILES
import pandas as pd
_2013 = pd.read_csv('C:/Users/Pau/OneDrive - University of the Philippines/Year 4/IE 198/Week 2/Case 1/2013.csv')
_2014 = pd.read_csv('C:/Users/Pau/OneDrive - University of the Philippines/Year 4/IE 198/Week 2/Case 1/2014.csv')
_2015 = pd.read_csv('C:/Users/Pau/OneDrive - University of the Philippines/Year 4/IE 198/Week 2/Case 1/2015.csv')

print(_2013, '\n')
print(_2014, '\n')
print(_2015)

#%% Concatenating the data
files = [_2013, _2014, _2015]
concatenated = pd.concat(files)
print(concatenated)


#%% 1A AVERAGE NUMBER OF CALLS REGARDLESS OF YEAR
# Grouping by MONTH
concat_copy = concatenated.copy()

# Retaining SUBID Column only
concat_copy.drop(['TRANSID', 'TRANS', 'VALIDITY', 'YEAR'], axis = 1, inplace = True)

# Renaming the SUBID to AVERAGE NUMBER OF CALLS PER MONTH
concat_copy.rename(columns = {'SUBID':'AVERAGE NUMBER OF CALLS PER MONTH'}, inplace = True)
#print(concat_copy)

# Grouping the dataframe by month
by_month = concat_copy.groupby('MONTH')

# Getting the AVERAGE number of calls per month
ave_calls = by_month.count()/3
print(ave_calls)
#%% 1B HIGHEST AND LOWEST AVERAGE CALL VOLUME

max_ave = ave_calls.max()
min_ave = ave_calls.min()
print(max_ave, "\n\n", min_ave)

#%% 2A TOTAL NUMBER OF CALLS PER MONTH FROM 2013 TO 2015
concat2_copy = concatenated.copy()
by_year_month = concat2_copy.groupby(['YEAR','MONTH'])
total_by_year = by_year_month.size()
print(total_by_year)
#%% 2B MONTH AND YEAR WITH THE HIGHEST TRAFFIC OF CALLS
max_by_year = total_by_year.max()
print(max_by_year)
#%% 3A INVALID CALLS IN Q1 AND Q4 IN 2013, 2014, 2015
concat3_copy = concatenated.copy()
concat3_copy.drop(['SUBID','TRANSID'], axis=1, inplace=True)
print(concat3_copy)
#%%
# SETTING MONTH AS THE INDEX - TO REMOVE Q2 AND Q3
index_month = concat3_copy.set_index('MONTH', inplace=False)

# DROPPING Q2-Q3
Q2_to_Q3 = [4,5,6,7,8,9]
only_Q1Q4 = index_month.drop(Q2_to_Q3, axis=0, inplace=True)
#%% REMOVING VALID VALUES
index_validity = index_month.set_index('VALIDITY', inplace=False)
sans_valid = index_validity.drop('VALID', axis=0, inplace=False)
#%% DETERMINING INVALID CALLS PER TYPE
by_validity = sans_valid.groupby(['VALIDITY','TRANS'])
print(by_validity.size())

#%% DETERMINING NUMBER OF INVALID CALLS PER MONTH (Q1 AND Q4 ONLY)
# RESETTING THE INDEX OF THE DATAFRAME WITHOUT VALID VALUES
reset_index = sans_valid.reset_index()
#%% GROUPING BY YEAR (Q1 AND Q4 ONLY)
by_year = reset_index.groupby('YEAR')

print(by_year.count()[['VALIDITY']], "\n")
print(by_year.count().sum()[['VALIDITY']])

#%% 4 AVERAGE NUMBER OF CALLS FROM A SUBSCRIBER
# This uses the SUBID and TRANS Column
# Count the number of times that each SUBID called - group by SUBID

concat4_copy = concatenated.copy()

# GROUPING THE DATAFRAME BY SUBID
by_subid = concat4_copy.groupby('SUBID')
#%% GETTING THE STATISTICS ABOUT SUBSCRIBER CALLS
df = by_subid['TRANS']
total_subcalls = df.count()
ave_subcalls = total_subcalls.mean()
std_subcalls = total_subcalls.std()
median_subcalls = total_subcalls.median()
variance_subcalls = total_subcalls.var()
max_subcalls = total_subcalls.max()
min_subcalls = total_subcalls.min()
range_subcalls = max_subcalls - min_subcalls
sum_subcalls = total_subcalls.sum()
count_subcalls = total_subcalls.count()

print("A subscriber called", ave_subcalls, "on average.","\n")
print("Standard deviation:", std_subcalls,"\n")
print("Median:", median_subcalls,"\n")
print("Variance:", variance_subcalls,"\n")
print("Maximum:", max_subcalls,"\n")
print("Minimum:", min_subcalls,"\n")
print("Range:", range_subcalls,"\n")
print("Sum:", sum_subcalls,"\n")
print("Count:", count_subcalls)
#%% TOP 5 TRANSACTIONS AVAILED BY SUBSCRIBERS
concat5_copy = concatenated.copy()
#%% RETAINING SUBID AND TRANS
concat5_copy.drop(['TRANSID','VALIDITY','MONTH','YEAR'], axis=1, inplace=True)

#%%  RENAMING SUBID TO FREQUENCY
concat5_copy.rename(columns = {'SUBID': 'FREQUENCY'}, inplace=True)

#%% GROUPING BY TRANS
by_trans = concat5_copy.groupby('TRANS')
trans_frequency = by_trans.count()
#%% RANKING THE FREQUENCY OF CALLS PER TRANSACTION IN DESCENDING ORDER
rank_frequency = trans_frequency.sort_values(by='FREQUENCY', ascending=False)
top5 = rank_frequency.head(5)
print(top5)
#%% 6A AVERAGE OF REPEAT CALLS FOR ALL SUBSCRIBERS
concat6_copy = concatenated.copy()
print(concat6_copy)
#%%
# GROUP BY SUBID THEN GET THE FREQUENCY OF CALLS PER SUBID
bysubidtrans = concat6_copy.groupby(['SUBID','TRANS']).size().to_frame('COUNT').reset_index()
print(bysubidtrans)
#%%
descending = bysubidtrans.sort_values(by='COUNT', ascending=False)
print(descending)

#%% 
bysubid = descending.groupby('SUBID').count()
#%%
repeat_call = descending[descending['COUNT'] > 1]
repeat_call = repeat_call.groupby(['SUBID'])
repeat_call = repeat_call.count()
print(repeat_call.sum()/len(bysubid))

#%% 6B AVERAGE NUMBER OF REPEAT CALLS FOR SUBSCRIBERS WITH AT LEAST ONE REPEAT CALL
print(repeat_call.sum()/len(repeat_call))

#%% NUMBER OF REPEAT CALLS PER SUBSCRIBER













