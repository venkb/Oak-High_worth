#random forest/bootstrap forest model for oakhurst high worth prediction

#load caTools package to split the dataset into test and validation set
library(caTools)
#load mice package to impute the missing values
library(mice)
#load randomForest package to build decision trees
library(randomForest)
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
dataset$High_Worth = factor(dataset$High_Worth)

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
randomforest_classifier = randomForest(formula = High_Worth ~ .,
                                data = training_set,
                                ntree = 100,
                                mtry = 5,
                                nodesize = 20)

#predict the test set
y_pred = predict(randomforest_classifier,
                 newdata = test_set[,-28])

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
        folds_randomforest_classifier = randomForest(formula = High_Worth ~ .,
                                               data = training_fold,
                                               ntree = 100,
                                               mtry = 5,
                                               nodesize = 20)
        folds_y_pred = predict(randomforest_classifier,
                               newdata = test_fold[,-28],
                               type = 'class')
        folds_cm = table(test_fold[,28], folds_y_pred)
        folds_accuracy = (folds_cm[1,1] + folds_cm[2,2])/(folds_cm[1,1] + folds_cm[1,2] + folds_cm[2,1] + folds_cm[2,2])
        return(folds_accuracy)
})

mean_accuracy = mean(as.numeric(accuracy))
mean_accuracy

#build full model - final randomforest classifier
randomforest_classifier_final = randomForest(formula = High_Worth ~ .,
                                       data = imputed_dataset,
                                       ntree = 100,
                                       mtry = 5,
                                       nodesize = 20)

#predict a new unseen test set
unseen_test_set = read.csv('cali3.csv')
unseen_test_set = unseen_test_set[,-28]

#take the unseen test set through all the data prep that the training set went through
#change variable names
colnames(unseen_test_set) = c('Lot_Size', 'Num_of_Bedrooms', 'Unit_Type',
                      'Electric_Cost','Internet_Access','Fiber_Optic',
                      'Heatfuel_Type','Num_of_Rooms','Water_Cost',
                      'Built_Year','Ownership_Status','Mortgage_Equity_Status',
                      'Fam_Type_Emp_Status','Household_Language',
                      'Fam_Type_Household_Status','Children_Age',
                      'Residence_Duration','Family_Size','Num_of_Children',
                      'Under_18','Over_60','Over_65','Num_of_Workers',
                      'Household_Work_Exp','Household_Work_Status',
                      'Num_of_Vehicles','Mobile_Broadband')

#data preparation and analysis
str(dataset)

#change the modeling type of variables
unseen_test_set$Lot_Size = factor(unseen_test_set$Lot_Size)
unseen_test_set$Num_of_Bedrooms = as.numeric(unseen_test_set$Num_of_Bedrooms)
unseen_test_set$Unit_Type = factor(unseen_test_set$Unit_Type)
unseen_test_set$Electric_Cost = as.numeric(unseen_test_set$Electric_Cost)
unseen_test_set$Internet_Access = factor(unseen_test_set$Internet_Access)
unseen_test_set$Fiber_Optic = factor(unseen_test_set$Fiber_Optic)
unseen_test_set$Heatfuel_Type = factor(unseen_test_set$Heatfuel_Type,
                                       levels = c(1,2,3,4,5,6,7,8,9),
                                       labels = c(1,2,3,4,5,6,7,8,9))
unseen_test_set$Num_of_Rooms = as.numeric(unseen_test_set$Num_of_Rooms)
unseen_test_set$Water_Cost = as.numeric(unseen_test_set$Water_Cost)
unseen_test_set$Built_Year = factor(unseen_test_set$Built_Year)
unseen_test_set$Ownership_Status = factor(unseen_test_set$Ownership_Status)
unseen_test_set$Mortgage_Equity_Status = factor(unseen_test_set$Mortgage_Equity_Status)
unseen_test_set$Fam_Type_Emp_Status = factor(unseen_test_set$Fam_Type_Emp_Status)
unseen_test_set$Household_Language = factor(unseen_test_set$Household_Language)
unseen_test_set$Fam_Type_Household_Status = factor(unseen_test_set$Fam_Type_Household_Status)
unseen_test_set$Children_Age = factor(unseen_test_set$Children_Age)
unseen_test_set$Residence_Duration = factor(unseen_test_set$Residence_Duration)
unseen_test_set$Family_Size = as.numeric(unseen_test_set$Family_Size)
unseen_test_set$Num_of_Children = as.numeric(unseen_test_set$Num_of_Children)
unseen_test_set$Under_18 = factor(unseen_test_set$Under_18)
unseen_test_set$Over_60 = factor(unseen_test_set$Over_60)
unseen_test_set$Over_65 = factor(unseen_test_set$Over_65)
unseen_test_set$Num_of_Workers = as.numeric(unseen_test_set$Num_of_Workers)
unseen_test_set$Household_Work_Exp = factor(unseen_test_set$Household_Work_Exp)
unseen_test_set$Household_Work_Status = factor(unseen_test_set$Household_Work_Status)
unseen_test_set$Num_of_Vehicles = as.numeric(unseen_test_set$Num_of_Vehicles)
unseen_test_set$Mobile_Broadband = factor(unseen_test_set$Mobile_Broadband)

#impute missing data
impute_unseen_test_set = mice(data = unseen_test_set,
                      m = 1,
                      maxit = 3,
                      seed = 5)

#complete the dataset with imputed values
imputed_unseen_test_set = complete(impute_unseen_test_set,1)

#predict the unseen test set with the final randomforest classifier
unseen_test_set_y_pred = predict(randomforest_classifier_final,
                                 newdata = imputed_unseen_test_set)
oakhurst_randomforest_output = cbind(unseen_test_set,unseen_test_set_y_pred)
names(oakhurst_randomforest_output)[28] = "Prediction"
write.csv(x = oakhurst_randomforest_output,
          file = 'oakhurst_randomforest_output.csv',
          row.names = FALSE)