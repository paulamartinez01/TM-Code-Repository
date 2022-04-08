# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:21:32 2021

@author: Pau
"""

import pandas as pd

suicide = pd.read_csv("C:/Users/Pau/OneDrive - University of the Philippines/Year 4/IE 198/Week 3/Case 2/Datasets/suicide.csv")
sui_agesex = suicide.copy()
#%% SUICIDE RATE ON AVERAGE (? CONSULT SIR) 
# RETAINING SUICIDE RATE AND AGE RANGE ONLY
sui_agesex.drop(["country", "year", "suicides_no", "population", "country-year", "HDI for year", "gdp_for_year ($)", "gdp_per_capita ($)", "generation"], axis = 1, inplace = True)


#%% RENAMING THE SUICIDE RATES COLUMN TO AVERAGE SUICIDE RATES
sui_agesex.rename(columns = {"age" : "Age", "sex": "Sex"}, inplace = True)


#%% GETTING THE AVERAGE SUICIDE RATE PER AGE RANGE
by_agesex = sui_agesex.groupby(["Age", "Sex"]).mean()

#%% LONG TO WIDE FORMAT
wide = by_agesex.pivot_table(index="Age", columns = "Sex", values = "suicides/100k pop")


#%% NEW COLUMN FOR THE AVERAGE PER AGE RANGE
wide['avg per age'] = (wide['female'] + wide['male'])/2


#%% SORTING THE VALUES ASCENDINGLY
wide = wide.sort_values(by = "avg per age", ascending = True)


#%% DROPPING THIS COLUMN FOR VISUALIZATION
wide.drop("avg per age", axis=1, inplace = True)

#%% CREATING THE HORIZONTAL STACKED BAR CHART
wide.plot.barh(
    stacked = True,
    color = ["lightgray", "cornflowerblue"],
    title = "Average Suicide Rate per 100,000 Population",
    )
#%% REGRESSION SCATTERPLOT
import seaborn as sns
import pandas as pd
import numpy as np
sui_rategdp = suicide.copy()
#%% REMOVING UNNEEDED COLUMNS
sui_rategdp.drop(['year', 'sex', 'age', 'population', 'country-year', 'HDI for year', 'gdp_for_year ($)', 'generation'], axis = 1, inplace=True)
#%% GETTING THE MEAN VALUES PER COUNTRY
by_country = sui_rategdp.groupby('country').mean().reset_index()
#%% GETTING AN INITIAL PICTURE OF THE RELATIONSHIP BETWEEN VARIABLES
sns.pairplot(by_country) #Getting the histogram and scatterplot of each variable and with each other!
# From the pairplot, we see that there is an inverse relationship between suicide rate/100k population and gdp per capita
# We also see that all the 3 numerical variables are skewed to the right.
# To show these variables in future plots, we can transform them by the square root of the values.
#%% CREATING COLUMNS FOR THE SQRT TRANSFORMATION
by_country['sqrt_gdp_per_cap'] = np.sqrt(by_country['gdp_per_capita ($)'])
by_country['sqrt_suicide#'] = np.sqrt(by_country['suicides_no'])
by_country['sqrt_suicide_rate'] = np.sqrt(by_country['suicides/100k pop'])
#%% CREATING A NEW DF WITH ONLY THE SQRT VALUES, AND REMOVING THESE VALUES FROM BY_COUNTRY
by_country.rename(columns = {"sqrt_gdp_per_cap":"GDP per capita","sqrt_suicide_rate":"Suicide Rate per 100k Population"}, inplace = True)
sqrt_form = by_country.drop(columns = ['suicides/100k pop', 'gdp_per_capita ($)', 'suicides_no'], inplace = False)

#%% CHECKING IF THE DISTRIBUTION IS BETTER
sns.pairplot(sqrt_form)

#%% 
sqrt_form = by_country.drop(columns = ['suicides/100k pop', 'gdp_per_capita ($)', 'suicides_no'], inplace = False)
by_country.drop(columns = ["GDP per capita", "sqrt_suicide#", "Suicide Rate per 100k Population"], inplace = True)


#%% RENAMING THE COLUMNS

#%% CREATING THE REGRESSION SCATTERPLOT
sns.jointplot(x = 'GDP per capita', y = 'Suicide Rate per 100k Population', data = sqrt_form, kind = 'reg')

#%% HEAT MAP
import plotly.graph_objects as go
import plotly
import pandas as pd

suicide = pd.read_csv("C:/Users/Pau/OneDrive - University of the Philippines/Year 4/IE 198/Week 3/Case 2/Datasets/suicide.csv")
df = suicide.copy()

#%% FILE FOR THE CODE OF EACH COUNTRY (TAKEN FROM: https://plotly.com/python/choropleth-maps/)
locations = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

#%% RETAINING COUNTRIES AND THEIR SUICIDE RATE
df.drop(columns = ['year', 'sex', 'age', 'population', 'country-year', 'HDI for year', 'gdp_for_year ($)', 'generation', 'suicides_no', 'gdp_per_capita ($)'], axis = 1, inplace=True)

#%% GROUPING BY COUNTRY, GETTING THE MEAN SUICIDE RATE, THEN RESETTING THE INDEX
group_country = df.groupby('country').mean().reset_index()

#%%
fig = go.Figure(data = go.Choropleth(
    locations = locations['CODE'], 
    z = group_country['suicides/100k pop'].astype(float),  
    text = locations['COUNTRY'], 
    colorscale = 'Blues',
    autocolorscale = False,
    reversescale = False,
    marker_line_color = 'darkgray',
    marker_line_width = 0.5,
    colorbar_title = 'Average Suicide Rate (1987-2014)',
    ))
#%%
fig.update_layout(
    title_text = 'Average Suicide Rate per 100,000 Population (1987-2014)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular',
        countrycolor = 'gray',
        showcountries = True,
        ),
    )

fig.show()
plotly.offline.plot(fig, filename='choropleth.html')

















