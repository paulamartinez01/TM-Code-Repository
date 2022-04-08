# initial analysis
  # row 1 = data #; should be removed
  # 3 binary variables - network, day, type
  # first row first name is a number

# 3.1. Import the Data in R and Fit a Linear Regression Model on the Full Dataset. 
tvdata = read.csv("tvdataset.csv")
rownames(tvdata) = tvdata[,2]
tvdata = tvdata[,c(3:13)]
initialfit <- lm(revenue ~ ., data=tvdata)
summary(initialfit)


# plotting the residual fit
par(mfrow = c(2,2), mar = c(2,2,2,2))
plot(initialfit)


###### 3.2.1 Outliers
# leverage points
plot(hatvalues(initialfit))
abline(h=11/69, col="red") # many potential outliers based on the leverage point

# influential points
plot(rstudent(initialfit))
abline(h=3, col="red")

# Identifying the top 4 shows that are potential outliers
head(sort(abs(rstudent(initialfit)), decreasing=TRUE), 4)

##### 3.2.2 Does the full model satisfy the non-constant variance assumption of regression? 
### check the RVF plot
plot(initialfit)

##### 3.3.3 Reduce the original dataset by removing the identified outliers. 
          
# remove the outliers
newtvdata = tvdata[-c(4,11,28,42),]
newtvdata

newtvdata[c(4,11,28,42),]

# new linear model fit using the new data
newfit = lm(revenue ~ ., data=newtvdata)

### 3.3.1. Which of the variables are significant in the final iteration?
# do Stepwise Regression on the reduced dataset.
stepwise = step(newfit, scope=list(lower = ~1, upper = ~network + day +
                                     length + viewers + d1849rating +
                                     facebooklikes + facebooktalkingabout +
                                     twitter + age + type, direction = "both", trace = 1))

 
# reduced model
reducedfit <- lm(revenue ~ network + day + viewers + d1849rating +
                   facebooklikes + facebooktalkingabout + 
                   twitter + type, data=newtvdata)

summary(reducedfit)

##### 3.4. Standardize the reduced dataset with significant variables. 
# create dummy variables for network, day, and type
dumtvdata = fastDummies::dummy_cols(newtvdata, select_columns = c("network", "day", "type"), remove_first_dummy = TRUE, remove_selected_columns = TRUE)
dumtvdata
knitr::kable(dumtvdata)

# standardizing the model
options(scipen=100)
sdumtvdata = data.frame(scale(dumtvdata, center = TRUE, scale = TRUE))

sfinalfit = lm(revenue ~ ., data=sdumtvdata)
summary(sfinalfit)




