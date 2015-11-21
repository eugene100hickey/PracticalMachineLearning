library(caret)
library(rJava)
library(extraTrees)
library(gplots)

set.seed(27183)

setwd("C:/Users/Eugene/Desktop/Coursera/08 Practical Machine Learning/Project")
train.data = read.csv2("pml-training.csv", sep=",")
test.data = read.csv2("pml-testing.csv", sep=",")

#remove empty columns from test data
test.data = test.data[,colSums(is.na(test.data))<nrow(test.data)]
#remove problem_id column and initial boring columns from test data that just counts from 1:20
test.data = test.data[,-c((1:7),60)]

#give train data same columns as test data, but with classe column (160) added
train.data = train.data[,c(which(names(train.data) %in% names(test.data)), 160)]


#splitting into a training and validation sets
intrain = createDataPartition(y=train.data$classe, p=0.7, list=F)
training=train.data[intrain,]
valid = train.data[-intrain,]

valid1=valid

#training.x = sapply(training, FUN=as.numeric)
#testing = sapply(testing, as.numeric)
#predXT= extraTrees(x=training.x, y = training[,53], na.action = "zero")

#predXT.results = predict(predXT, testing)

#mytab = table(predXT.results, testing1$classe)

#heatmap.2(x=mytab, dendrogram="none", cellnote=mytab, notecex=2, notecol="black",trace='none')


#weeding out columns with low variability
train.x = training[,-53]
train.x = sapply(train.x, FUN=as.numeric)
means = apply(train.x, MARGIN=2, FUN=mean)
sds = apply(train.x, MARGIN=2, FUN=sd)
plot(abs(sds/means))
train.xt = train.x[,(abs(sds/means)>2)]
train.xt = as.data.frame(train.xt)
train.xt = cbind(train.xt, training$classe)
names(train.xt)[length(train.xt)] = "classe"


#fit random forest to the reduced set, using cross validation with k=3
modfit.rf = train(classe~.,data=train.xt, 
               method="rf", 
               trControl = trainControl(method = "cv", number = 3), 
               allowParallel=T)

#validating model, gives accuracy of 0.897
valid1 = valid1[c(which(names(valid1) %in% names(train.xt)))]
pred.rf = predict(modfit.rf, valid1)
confusionMatrix(valid1$classe, pred.rf)

#fit random forest10 to the reduced set, using cross validation with k=10
modfit.rf10 = train(classe~.,data=train.xt, 
                  method="rf", 
                  trControl = trainControl(method = "cv", number = 10), 
                  allowParallel=T)

#validating model, gives accuracy of 0.94
pred.rf10 = predict(modfit.rf10, valid1)
confusionMatrix(valid1$classe, pred.rf10)

#fit random forest10 to the reduced set, using cross validation with k=100
modfit.rf100 = train(classe~.,data=train.xt, 
                    method="rf", 
                    trControl = trainControl(method = "cv", number = 100), 
                    allowParallel=T)

#validating model, gives accuracy of 0.94
pred.rf100 = predict(modfit.rf100, valid1)
confusionMatrix(valid1$classe, pred.rf100)

#fit random forest to the reduced set, using boost632
modfit.rf632 = train(classe~.,data=train.xt,
                     method="rf",
                     trControl = trainControl(method = "boot632", number = 3), 
                     allowParallel=T)

#validating rf632 model, gives accuracy of 0.900
pred.rf = predict(modfit.rf632, valid1)
cm = confusionMatrix(valid1$classe, pred.rf)

#gbm model using cv
modfit.gbm = train(classe ~ ., data=train.xt, method = "gbm",
                   trControl = trainControl(method="repeatedcv",number=5, repeats=1),
                   verbose = FALSE,
                   allowParallel=T)

#validating gbm model, gives accuracy of 0.79
pred.gbm = predict(modfit.gbm, valid1)
cm = confusionMatrix(valid1$classe, pred.gbm)


#featureplot for reduced parameter set
featurePlot(x = abs(train.xt[,c(1,2,3)]), y=as.numeric(train.xt$classe), plot="pairs")
