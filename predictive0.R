# install.packages('AppliedPredictiveModeling')
library(AppliedPredictiveModeling)
data("segmentationOriginal")
#k <- data("segmentationOriginal")
# View(k)
# train <- segmentationOriginal[segmentationOriginal[,"Case"]=='Train',]
train <- subset(segmentationOriginal,Case=='Train')
apropos("confusion")
segData <- subset(segmentationOriginal,Case == "Train")
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
segData <- segData[,-(1:3)]
# we are going to eliminate status columns
statusColNum <- grep("Status",names(segData))
statusColNum
segData <- segData[,-statusColNum]
# let's calculate the skewness
# install.packages('e1071')
library(e1071)
# for one predictor
skewness(segData$AngleCh1)
skewValues <- apply(segData,2,skewness)
head(skewValues,8)
# we are going to transform our predictors since some are skewed
# install.packages('caret')
library(caret)
ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)
ch1AreaTrans

# original data
head(segData$AreaCh1)

# transformed data
predict(ch1AreaTrans,head(segData$AreaCh1))

# let's calculate PCA
pcaObject <- prcomp(segData,center = TRUE,scale. = TRUE)

#let's find the varience accounted by the first 3 principal components
percentVariance <- pcaObject$sdev^2/sum(pcaObject$sdev^2)*100
percentVariance[1:3]

# let's view the stored pca
head(pcaObject$x[,1:5])

# let's view the predictors weights (loadings) in the firs three pr comp
head(pcaObject$rotation[,1:3])

# let's transform our data using preProcess method
trans <- preProcess(segData,
                    method = c("BoxCox","center","scale","pca"))
trans

#let's transform the variables in trans properly
transformed <- predict(trans,segData)
head(transformed[,1:3])

# filter near zero variance predictors
nearZeroVar(segData)

# filter between predictor correlation
correlation <- cor(segData)
dim(correlation)
correlation[1:4,1:4]
library(corrplot)
dev.new(width=30, height=30, unit="in")
corrplot(correlation,order="hclust",tl.cex = 0.25)
#filtering based off correlation
highCorr <- findCorrelation(correlation,cutoff = .75) #yields number of predictor to cut
length(highCorr)
head(highCorr)
filteredSegData <-segData[,-highCorr] 

# page 56


# creating dummy variables
data("cars")
nrow(cars)
set.seed(42)
index <- sample(1:nrow(cars),402)
carSubset <- cars[index,]
head(carSubset)
################################

#Overfitting and model tuning

# install.packages("MASS")
# install.packages("Design")
# install.packages("ipred")

# Data splitting

library(AppliedPredictiveModeling)
# data("logisticCreditPredictions")
# data("schedulingData")
# data("twoClassData")
str(predictors)
str(classes)

# lets create a stratified random sample with createdatapartition
# install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
set.seed(1)
trainingRows <- createDataPartition(classes,
                                    p = .80,
                                    list = FALSE)
head(trainingRows)
trainingPredictors <- predictors[trainingRows,]
trainingClasses <- classes[trainingRows]
testPredictors <- predictors[-trainingRows,]
testClasses <- classes[-trainingRows]
str(trainingPredictors)
str(testPredictors)
maxdissrows <- maxDissim(trainingPredictors,predictors,length(testClasses))
head(maxdissrows)
length(maxdissrows)
#############################################################

# Resampling

set.seed(1)
repeatedSplits <- createDataPartition(trainingClasses,
                                      p=.80,
                                      times = 3)
str(repeatedSplits)

set.seed(1)
cvSplits <- createFolds(trainingClasses,k=10,
                        returnTrain = TRUE)
str(cvSplits)
fold1 <- cvSplits[[1]]
cvPredictors1 <- predictors[fold1,]
cvClasses1 <- classes[fold1]
nrow(trainingPredictors)
str(trainingPredictors)
nrow(cvPredictors1)
nrow(predictors)
################################################################

#Basic model building

#KNN3
trainPredictors <- as.matrix(trainingPredictors)
knnFit <- knn3(x=trainPredictors, y = trainingClasses,k=5)
knnFit
testPredictions <- predict(knnFit,newdata = testPredictors,
                           type = 'class')
head(testPredictions)
str(testPredictions)
###################################################################

#Determination of tuning parameters

data("GermanCredit")
# scriptLocation()
GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)]
GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

## Split the data into training (80%) and test sets (20%)
set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]
inTrain
str(GermanCreditTrain)
set.seed(1056)
svmFit <- train(Class~.,
                data=GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center","scale"),
                tuneLength = 10,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 5,
                                         classProbs = TRUE))
svmFit
plot(svmFit,scales = list(x=list(log=2)))
predictedClass <- predict(svmFit,GermanCreditTest)
str(predictedClass)
predictedProb <- predict(svmFit,newdata = GermanCreditTest,
                         type = "prob")
head(predictedProb)

#############################################################

#between model comparisons
##########################################################
library(caret)
logReg <- train(Class~.,
                data = GermanCreditTrain,
                method = "glm",
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 5))
logReg
##################################################
#Let's compare the two models (SVM,Logistic)
##################################################
resamp <- resamples(list(SVM = svmFit,Logistic = logReg))
summary(resamp)
xyplot(resamp)
modelDifferences <- diff(resamp)
summary(modelDifferences)
####################################################
######################################################

#REGRESSION MODELS
#####################################################

#Variance-bias trade off

observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4,0.62, 0.99, 
              -0.18, 0.32, 0.34, -0.30, 0.04, -0.87, 0.55, -1.30, -1.15, 0.20)
predicted <- c(0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43,
               0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42,
               -0.25, -0.64, -1.26, -0.07)
residualValues <- observed - predicted
summary(residualValues)
#let's visualize the relationships
extRange <- extendrange(c(observed,predicted))
par(mfrow=c(1,2))
plot(observed,predicted,
     ylim = extRange,
     xlim = extRange)
abline(0,1,col="red",lty=2)
plot(predicted,residualValues,
     ylab = "residuals")
abline(h=0,col="blue",lty=2)

#let's calculate the r2 and different correlations
require(caret)
R2(predicted,observed)
cor(predicted,observed)
RMSE(predicted,observed)
cor(predicted,observed,method = "spearman") #rank correlation
