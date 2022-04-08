# check for imbalanced data
churntrain = read.csv("ChurnTrain.csv", stringsAsFactors = TRUE)
table(churntrain$Churn)
prop.table(table(churntrain$Churn))


##### random undersampling
n_true <- 410
new_frac_true <- 0.50
new_n_total2 <- n_true/new_frac_true

undersampling_result <- ovun.sample(Churn ~ ., data=churntrain,
                                   method = "under",
                                   N = new_n_total2,
                                   seed = 1)
undersampled_churntrain <- undersampling_result$data

table(undersampled_churntrain$Churn)

# modeling adaboost random forest
library(RWeka)
adaboostForest3 = AdaBoostM1(Churn ~ ., data=undersampled_churntrain,
                             control = Weka_control(W=list("weka/classifiers/trees/RandomForest")))

summary(adaboostForest3)
prop.table(table(undersampled_churntrain$Churn))

# predicting undersampled churn test data
churntest2 = read.csv("ChurnTest.csv")
churntest2$Predictions = predict(adaboostForest3,churntest2)
churntest2


write.csv(churntest2, file="predictions.csv")


