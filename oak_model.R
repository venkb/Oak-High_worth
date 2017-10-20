# oakhurst high-worth prediction
library(caTools) #to create training and test validation set
library(caret) #to arrive at hyperparameter values & perform k fold cross validation
library(ggplot2) #to visualize data
library(mice) #for imputing missing values
library(MASS) #for linear discriminant analysis
library(rpart) #for building decision tree
library(randomForest) #for building random forest
library(gbm) #for boosted tree
library(xgboost) #for extreme gradient boosting
library(h2o) #to build neural networks
h2o.init(nthreads = -1)

#load data
training_set = read.csv('cali1.csv')
test_set = read.csv('cali2_SOLN.csv')

#change variable names
colnames(training_set) = c('Lot_Size', 'Num_of_Bedrooms', 'Unit_Type',
                      'Electric_Cost','Internet_Access','Fiber_Optic',
                      'Heatfuel_Type','Num_of_Rooms','Water_Cost',
                      'Built_Year','Ownership_Status','Mortgage_Equity_Status',
                      'Fam_Type_Emp_Status','Household_Language',
                      'Fam_Type_Household_Status','Children_Age',
                      'Residence_Duration','Family_Size','Num_of_Children',
                      'Under_18','Over_60','Over_65','Num_of_Workers',
                      'Household_Work_Exp','Household_Work_Status',
                      'Num_of_Vehicles','Mobile_Broadband','High_Worth')

colnames(test_set) = c('Lot_Size', 'Num_of_Bedrooms', 'Unit_Type',
                      'Electric_Cost','Internet_Access','Fiber_Optic',
                      'Heatfuel_Type','Num_of_Rooms','Water_Cost',
                      'Built_Year','Ownership_Status','Mortgage_Equity_Status',
                      'Fam_Type_Emp_Status','Household_Language',
                      'Fam_Type_Household_Status','Children_Age',
                      'Residence_Duration','Family_Size','Num_of_Children',
                      'Under_18','Over_60','Over_65','Num_of_Workers',
                      'Household_Work_Exp','Household_Work_Status',
                      'Num_of_Vehicles','Mobile_Broadband','High_Worth')

#change variable modeling type
training_set$Lot_Size = factor(training_set$Lot_Size)
training_set$Num_of_Bedrooms = as.numeric(training_set$Num_of_Bedrooms)
training_set$Unit_Type = factor(training_set$Unit_Type)
training_set$Electric_Cost = as.numeric(training_set$Electric_Cost)
training_set$Internet_Access = factor(training_set$Internet_Access)
training_set$Fiber_Optic = factor(training_set$Fiber_Optic)
training_set$Heatfuel_Type = factor(training_set$Heatfuel_Type)
training_set$Num_of_Rooms = as.numeric(training_set$Num_of_Rooms)
training_set$Water_Cost = as.numeric(training_set$Water_Cost)
training_set$Built_Year = factor(training_set$Built_Year)
training_set$Ownership_Status = factor(training_set$Ownership_Status)
training_set$Mortgage_Equity_Status = factor(training_set$Mortgage_Equity_Status)
training_set$Fam_Type_Emp_Status = factor(training_set$Fam_Type_Emp_Status)
training_set$Household_Language = factor(training_set$Household_Language)
training_set$Fam_Type_Household_Status = factor(training_set$Fam_Type_Household_Status)
training_set$Children_Age = factor(training_set$Children_Age)
training_set$Residence_Duration = factor(training_set$Residence_Duration)
training_set$Family_Size = as.numeric(training_set$Family_Size)
training_set$Num_of_Children = as.numeric(training_set$Num_of_Children)
training_set$Under_18 = factor(training_set$Under_18)
training_set$Over_60 = factor(training_set$Over_60)
training_set$Over_65 = factor(training_set$Over_65)
training_set$Num_of_Workers = as.numeric(training_set$Num_of_Workers)
training_set$Household_Work_Exp = factor(training_set$Household_Work_Exp)
training_set$Household_Work_Status = factor(training_set$Household_Work_Status)
training_set$Num_of_Vehicles = as.numeric(training_set$Num_of_Vehicles)
training_set$Mobile_Broadband = factor(training_set$Mobile_Broadband)
training_set$High_Worth = factor(training_set$High_Worth)

#change the modeling type of test_set
test_set$Lot_Size = factor(test_set$Lot_Size)
test_set$Num_of_Bedrooms = as.numeric(test_set$Num_of_Bedrooms)
test_set$Unit_Type = factor(test_set$Unit_Type)
test_set$Electric_Cost = as.numeric(test_set$Electric_Cost)
test_set$Internet_Access = factor(test_set$Internet_Access)
test_set$Fiber_Optic = factor(test_set$Fiber_Optic)
test_set$Heatfuel_Type = factor(test_set$Heatfuel_Type)
test_set$Num_of_Rooms = as.numeric(test_set$Num_of_Rooms)
test_set$Water_Cost = as.numeric(test_set$Water_Cost)
test_set$Built_Year = factor(test_set$Built_Year)
test_set$Ownership_Status = factor(test_set$Ownership_Status)
test_set$Mortgage_Equity_Status = factor(test_set$Mortgage_Equity_Status)
test_set$Fam_Type_Emp_Status = factor(test_set$Fam_Type_Emp_Status)
test_set$Household_Language = factor(test_set$Household_Language)
test_set$Fam_Type_Household_Status = factor(test_set$Fam_Type_Household_Status)
test_set$Children_Age = factor(test_set$Children_Age)
test_set$Residence_Duration = factor(test_set$Residence_Duration)
test_set$Family_Size = as.numeric(test_set$Family_Size)
test_set$Num_of_Children = as.numeric(test_set$Num_of_Children)
test_set$Under_18 = factor(test_set$Under_18)
test_set$Over_60 = factor(test_set$Over_60)
test_set$Over_65 = factor(test_set$Over_65)
test_set$Num_of_Workers = as.numeric(test_set$Num_of_Workers)
test_set$Household_Work_Exp = factor(test_set$Household_Work_Exp)
test_set$Household_Work_Status = factor(test_set$Household_Work_Status)
test_set$Num_of_Vehicles = as.numeric(test_set$Num_of_Vehicles)
test_set$Mobile_Broadband = factor(test_set$Mobile_Broadband)
test_set$High_Worth = factor(test_set$High_Worth)

#impute missing values in training set
summary(training_set)
md.pattern(training_set)
impute_training_set = mice(training_set,m=1, maxit=5)
imputed_training_set = complete(impute_training_set,1)

#impute missing values in test set
impute_test_set = mice(test_set, m=1, maxit = 5)
imputed_test_set = complete(impute_test_set,1)

#-------------------------------------------------------------------
#build logistic regression classifier using backward selection
logreg_classifier = glm(High_Worth ~ .,
                        data = imputed_training_set,
                        family = 'binomial')
summary(logreg_classifier)
#full model returns the least test misclassification rate

# logreg_classifier = glm(High_Worth ~ .- Lot_Size -Unit_Type -Children_Age -Under_18 -Over_60,
#                         data = imputed_training_set,
#                         family = 'binomial')

#perform grid search derive the hyper parameter values
logreg_hyper_parameters = train(form = High_Worth ~ .,
                                data = imputed_training_set,
                                method = "glm")
logreg_hyper_parameters$bestTune
#there isnt any hyper parameter to tune for logistic regression

# #logistic regression model selection using subsets
# logreg_classifier_subsets = leaps::regsubsets(High_Worth ~ .,
#                                                           data = imputed_training_set,
#                                                           nbest = 2,
#                                                           method = "backward")
# logreg_classifier_subsets_summary = summary(logisticregression_classifier_subsets)
# which.max(logreg_classifier_subsets_summary$adjr2)
# which.min(logreg_classifier_subsets_summary$bic)
# which.min(logreg_classifier_subsets_summary$cp)
# coef(logreg_classifier_subsets,15)
# 
# logreg_classifier = glm(High_Worth ~ Num_of_Rooms + Household_Language +
#                                 Fam_Type_Household_Status + Num_of_Workers +
#                                 Electric_Cost + Mortgage_Equity_Status,
#                         data = imputed_training_set,
#                         family = 'binomial')
# summary(logreg_classifier)

#predict test set data
logreg_y_prob = predict(logreg_classifier, 
                 newdata = imputed_test_set[,-28],
                 type = 'response')
logreg_y_pred = ifelse(y_prob > 0.5, 1, 0)

#build the confusion matrix
cm = table(imputed_test_set$High_Worth, logreg_y_pred)
logreg_misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
logreg_misclassification_rate
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#build decision tree classifier  
decisiontree_classifier = rpart(High_Worth ~ .,
                                data = imputed_training_set)
plot(decisiontree_classifier)
text(decisiontree_classifier)

#perform grid search to derive the hyper parameter values
decisiontree_hyper_parameters = train(form = High_Worth ~ .,
                                data = imputed_training_set,
                                method = "rpart")
decisiontree_hyper_parameters$bestTune
#the grid search returns a hyper parameter value for cp = 0.002916953

#rebuild the decision tree with arrived hyper parameters value
decisiontree_classifier = rpart(High_Worth ~ .,
                                data = imputed_training_set,
                                control = rpart.control(cp = 0.002916953))
plot(decisiontree_classifier)
text(decisiontree_classifier)

#predict the test set
decisiontree_y_pred = predict(decisiontree_classifier,
                 newdata = imputed_test_set[,-28],
                 type = 'class')

#build the confusion matrix
cm = table(imputed_test_set$High_Worth, decisiontree_y_pred)
decisiontree_misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
decisiontree_misclassification_rate
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#build random forest (bootstrap forest) classifier
randomforest_classifier = randomForest(High_Worth ~ .,
                                       data = imputed_training_set,
                                       ntree = 500,
                                       ntry = 6,
                                       nodesize = 10)

# #perform grid search to derive the hyper parameter values
# randomforest_hyper_parameters = train(form = High_Worth ~ .,
#                                       data = imputed_training_set,
#                                       method = "rf")
# randomforest_hyper_parameters$bestTune
# #grid search was time consuming and hence had to abort

#predict the test set
randomforest_y_pred = predict(randomforest_classifier,
                              newdata = imputed_test_set[,-28])

#build the confusion matrix
cm = table(imputed_test_set$High_Worth, randomforest_y_pred)
randomforest_misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
randomforest_misclassification_rate
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#building boosted tree classifier
#gbm requires the output to be a vector and not factor
gbm_imputed_training_set = imputed_training_set
gbm_imputed_training_set$High_Worth = as.vector(gbm_imputed_training_set$High_Worth)

boostedtree_classifier = gbm(High_Worth ~ .,
                             data = gbm_imputed_training_set,
                             distribution = 'bernoulli',
                             n.trees = 100,
                             interaction.depth = 3)

#predict the test set
boostedtree_y_prod = predict.gbm(boostedtree_classifier,
                                 newdata = imputed_test_set[,-28],
                                 n.trees = 100,
                                 type = 'response')

boostedtree_y_pred = ifelse(boostedtree_y_prod > 0.5, 1, 0)

#build the confusion matrix
class(boostedtree_y_pred)
cm = table(imputed_test_set$High_Worth, boostedtree_y_pred)
boostedtree_misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
boostedtree_misclassification_rate

#-------------------------------------------------------------------

#-------------------------------------------------------------------
#to build neural network classifier
neuralnetwork_classifier = h2o.deeplearning(y = 'High_Worth',
                                            training_frame = as.h2o(imputed_training_set),
                                            activation = 'Tanh',
                                            hidden = c(25,25),
                                            epochs = 100,
                                            train_samples_per_iteration = -2)

neuralnetwork_y_pred = h2o.predict(neuralnetwork_classifier,
                                 newdata = as.h2o(imputed_test_set[,-28]))

neuralnetwork_y_pred = as.data.frame(neuralnetwork_y_pred)

#build the confusion matrix
cm = table(imputed_test_set$High_Worth, neuralnetwork_y_pred$predict)
neuralnetwork_misclassification_rate = (cm[2,1] + cm[1,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
neuralnetwork_misclassification_rate
h2o.shutdown()
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#build extreme gradient boosting classifier
xgboost_classifier = xgboost(data = as.matrix(imputed_training_set[,-28]),
                             label = imputed_training_set$High_Worth,
                             nrounds = 10)
