mydata = read.csv("diabetes_prediction_dataset.csv")
str(mydata)
# Check number of rows and columns
dim(mydata)

# order: 0 1 2 3 4 5
order <- c("No Info", "never", "not current", "former", "current", "ever")
# Perform ordinal encoding
mydata[,5] <- match(mydata[,5], order)
# Perform binary encoding for "gender", Female = 1, Male = 0
mydata[,1] <- as.numeric(mydata[,1] == "Female")
# convert factor to numeric (if deal with character, always convert it into numeric)
for(i in 1:9) {
  mydata[, i] <- as.numeric(as.character(mydata[, i]))
}

# Change Y values to 1's and 0's and factor (if 1 is 1, 0 is 0)
mydata$diabetes <- ifelse(mydata$diabetes == "1", 1, 0)
mydata$diabetes <- factor(mydata$diabetes, levels = c(0, 1))

# Prep Training and Test data.
library(caret)
'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.
set.seed(100)

#split the data in 70% training, 30% testing
trainDataIndex <- createDataPartition(mydata$diabetes, p=0.7, list = F)
trainData <- mydata[trainDataIndex, ]
testData <- mydata[-trainDataIndex, ]

# Class distribution of train data
table(trainData$diabetes)

# Build Logistic Model
mydata1 <- glm(diabetes ~ ., family = "binomial", data=trainData)

pred <- predict(mydata1, newdata = testData, type = "response") #response to compute prediction probabilities

# Recode factors (cutoff 0.5 as probabilities)
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- testData$diabetes

# Accuracy
mean(y_pred == y_act) 

############### Remove unnecessary variable (4)  #################
# Build Logistic Model
mydata2.reduced <- glm(diabetes ~ age + bmi + HbA1c_level + blood_glucose_level,
                       family = "binomial", data=trainData)

pred <- predict(mydata2.reduced, newdata = testData, type = "response") #response to compute prediction probabilities

# Recode factors (cutoff 0.5 as probabilities)
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- testData$diabetes

# Accuracy
mean(y_pred == y_act)  

############### Remove unnecessary variable (2) - gender and smoking_history #################
# Build Logistic Model
mydata.reduced <- glm(diabetes ~ age + bmi + heart_disease + hypertension + HbA1c_level + blood_glucose_level,
                family = "binomial", data=trainData)

pred <- predict(mydata.reduced, newdata = testData, type = "response") #response to compute prediction probabilities

# Recode factors (cutoff 0.5 as probabilities)
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- testData$diabetes

# Accuracy
mean(y_pred == y_act) 

anova(mydata1, mydata.reduced, test="Chisq") 

library(InformationValue)

#define threshold value
optCutOff <- optimalCutoff(testData$diabetes, pred) 
#classification error
misClassError(testData$diabetes, pred, threshold = optCutOff)
#plot AUC
plotROC(testData$diabetes, pred)
sensitivity(testData$diabetes, pred, threshold = optCutOff)
confusionMatrix(testData$diabetes, pred, threshold = optCutOff)

# after decide which data, drop the unnecessary
mydata = subset(mydata, select = -c(gender, smoking_history)) 
str(mydata)
# Plot Predicted data and original data points
plot(diabetes ~ age, data=testData)
plot(diabetes ~ hypertension, data=testData)
plot(diabetes ~ heart_disease, data=testData)
plot(diabetes ~ bmi, data=testData)
plot(diabetes ~ HbA1c_level, data=testData)
plot(diabetes ~ blood_glucose_level, data=testData)

x <- mydata[ , 1:6]
y <- mydata[ , 7]
#install.packages("ellipse")
library(ellipse)

# Make dependent variable as a factor (categorical)
mydata$diabetes = as.factor(mydata$diabetes)
featurePlot(x=x, y=y, plot="ellipse")

# Split data into training (70%) and validation (30%)
dt = sort(sample(nrow(mydata), nrow(mydata)*.7))
train<-mydata[dt,]
val<-mydata[-dt,] # Check number of rows in training data set
nrow(train)

str(train)
# Decision Tree Model
library(rpart)
mtree <- rpart(diabetes~., data = train, method="class", 
               control = rpart.control(minsplit = 20, minbucket = 7, 
                                       maxdepth = 10, usesurrogate = 2, 
                                       xval =10 ))

mtree

#Plot tree
plot(mtree)
text(mtree)

## pruning the tree
printcp(mtree)
bestcp <- mtree$cptable[which.min(mtree$cptable[,"xerror"]),"CP"]

# Prune the tree using the best cp.
pruned <- prune(mtree, cp = bestcp)
#install.packages("rattle")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
# Plot pruned tree
prp(pruned, faclen = 0, cex = 0.8, extra = 1)
