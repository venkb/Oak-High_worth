#boosted tree model for oakhurst high worth prediction

#load caTools package to split the dataset into test and validation set
library(caTools)
#load mice package to impute the missing values
library(mice)
#load gbm package to build decision trees
library(gbm)
#load caret package to perform k-fold cross validation
library(caret)

#load data
setwd('C:\\Venkat\\Villanova\\Semester 4\\MSA 8220 - Analytical Methods for Data Mining\\Final Project\\Model_R')
set1 = read.csv('cali1.csv')
set2 = read.csv('cali2_SOLN.csv')
dataset = rbind(set1,set2)

#change variable names
colnames(dataset) = c('Lot_Size', 'Num_of_Bedrooms', 'Unit_Type',
                      'Electric_Cost','Internet_Access','Fiber_Optic',
                      'Heatfuel_Type','Num_of_Rooms','Water_Cost',
                      'Built_Year','Ownership_Status','Mortgage_Equity_Status',
                      'Fam_Type_Emp_Status','Household_Language',
                      'Fam_Type_Household_Status','Children_Age',
                      'Residence_Duration','Family_Size','Num_of_Children',
                      'Under_18','Over_60','Over_65','Num_of_Workers',
                      'Household_Work_Exp','Household_Work_Status',
                      'Num_of_Vehicles','Mobile_Broadband','High_Worth')

#data preparation and analysis
str(dataset)

#change the modeling type of variables
dataset$Lot_Size = factor(dataset$Lot_Size)
dataset$Num_of_Bedrooms = as.numeric(dataset$Num_of_Bedrooms)
dataset$Unit_Type = factor(dataset$Unit_Type)
dataset$Electric_Cost = as.numeric(dataset$Electric_Cost)
dataset$Internet_Access = factor(dataset$Internet_Access)
dataset$Fiber_Optic = factor(dataset$Fiber_Optic)
dataset$Heatfuel_Type = factor(dataset$Heatfuel_Type)
dataset$Num_of_Rooms = as.numeric(dataset$Num_of_Rooms)
dataset$Water_Cost = as.numeric(dataset$Water_Cost)
dataset$Built_Year = factor(dataset$Built_Year)
dataset$Ownership_Status = factor(dataset$Ownership_Status)
dataset$Mortgage_Equity_Status = factor(dataset$Mortgage_Equity_Status)
dataset$Fam_Type_Emp_Status = factor(dataset$Fam_Type_Emp_Status)
dataset$Household_Language = factor(dataset$Household_Language)
dataset$Fam_Type_Household_Status = factor(dataset$Fam_Type_Household_Status)
dataset$Children_Age = factor(dataset$Children_Age)
dataset$Residence_Duration = factor(dataset$Residence_Duration)
dataset$Family_Size = as.numeric(dataset$Family_Size)
dataset$Num_of_Children = as.numeric(dataset$Num_of_Children)
dataset$Under_18 = factor(dataset$Under_18)
dataset$Over_60 = factor(dataset$Over_60)
dataset$Over_65 = factor(dataset$Over_65)
dataset$Num_of_Workers = as.numeric(dataset$Num_of_Workers)
dataset$Household_Work_Exp = factor(dataset$Household_Work_Exp)
dataset$Household_Work_Status = factor(dataset$Household_Work_Status)
dataset$Num_of_Vehicles = as.numeric(dataset$Num_of_Vehicles)
dataset$Mobile_Broadband = factor(dataset$Mobile_Broadband)
#for gbm the response variable should be coded as vector and not as factor
dataset$High_Worth = as.vector(dataset$High_Worth)

#impute missing data
impute_dataset = mice(data = dataset,
                      m = 1,
                      maxit = 3,
                      seed = 5)

#complete the dataset with imputed values
imputed_dataset = complete(impute_dataset,1)

#create the training and test set
set.seed(5)
split = sample.split(Y = imputed_dataset$High_Worth,
                     SplitRatio = 0.75)
training_set = subset(imputed_dataset, split == TRUE)
test_set = subset(imputed_dataset, split == FALSE)

#build the decision tree classifier using training set
boostedtree_classifier = gbm(formula = High_Worth ~ .,
                                       data = training_set,
                                       distribution = "bernoulli",
                                       n.trees = 500,
                                       interaction.depth = 3)

#predict the test set
y_prob = predict.gbm(boostedtree_classifier,
                 newdata = test_set[,-28],
                 n.trees = 500,
                 type = 'response')
y_pred = ifelse(y_prob >= 0.29, 1, 0)

#build the confusion matrix
cm = table(test_set[,28], y_pred)
cm

#determine the misclassification rate
misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
misclassification_rate

#validate the model using k-fold cross validation
set.seed(5)
folds = createFolds(y = imputed_dataset$High_Worth,
                    k = 10)
accuracy = lapply(folds, function(x){
        training_fold = imputed_dataset[-x,]
        test_fold = imputed_dataset[x,]
        folds_boostedtree_classifier = gbm(formula = High_Worth ~ .,
                                     data = training_fold,
                                     distribution = "bernoulli",
                                     n.trees = 500,
                                     interaction.depth = 3)
        folds_y_prob = predict.gbm(folds_boostedtree_classifier,
                             newdata = test_fold[,-28],
                             n.trees = 500,
                             type = 'response')
        folds_y_pred = ifelse(folds_y_prob >= 0.29, 1, 0)
        folds_cm = table(test_fold[,28], folds_y_pred)
        folds_accuracy = (folds_cm[1,1] + folds_cm[2,2])/(folds_cm[1,1] + folds_cm[1,2] + folds_cm[2,1] + folds_cm[2,2])
        return(folds_accuracy)
})

mean_accuracy = mean(as.numeric(accuracy))
mean_accuracy

#boosted tree classifier returned a mean accuracy of 0.8840995 over 10 folds