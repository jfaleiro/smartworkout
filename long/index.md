# Smartbells: When Your Dumbbell Outsmarts Your Personal Trainer
J Faleiro  
May 5, 2015  



# Introduction

People regularly quantify _*how much*_ of a particular activity they do, but they rarely quantify _*how well*_ that same activity is performed. 

This analysis predicts how well people exercise based on data produced by accelarators attached to their belt, forearm, arm, and dumbell. 

The quality in which people exercise is given by the "classe" variable in the training set. The experimentation and previous analysis was performed by the [HAR](http://groupware.les.inf.puc-rio.br/har) laboratory, and was previously detailed in [this paper][1].

On this work we describe alternatives for data analysis based on different techniques for pre-processing, feature selections, and model building.

By the end of analysis we use the best performing model to predict the exercise quality of 20 different cases.

# Pre-Processing



For this analysis a number of libraries are required. If you intend to replicate this report make sure you use the same list of dependencies:


```r
pacman::p_load(knitr, caret, e1071, mlbench, ggplot2, ISLR, Hmisc, gridExtra, RANN, 
               AppliedPredictiveModeling, corrplot, randomForest, gbm, splines, parallel,
               plyr, dplyr, leaps, MASS, pROC, C50)
```

## Downloading

The first step is downloading the `pml-training.csv` dataset, available at the address:


```r
url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
```

Initially we download a sample of the dataset to get a glimpse of the internal structure:


```r
temporaryFile <- 'pml-training.csv'
download.file(url, destfile=temporaryFile, method="curl")
```


```r
trainingRaw = read.csv(temporaryFile)
```


```r
dim(trainingRaw)
```

```
## [1] 19622   160
```

This is a fairly extensive dataset. There are a number of data points that cannot be processed the way they are, specifically the representation of `NA` values are not regular:


```r
str(trainingRaw$kurtosis_roll_dumbbell)
```

```
##  Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
str(trainingRaw$kurtosis_yaw_dumbbell)
```

```
##  Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
```

We define a number of constants to make sure we consider reasonable `NA` values, filter out non-relevant columns for the regression, and set a `shrinkage` factor. This factor is useful mostly to lower processing times during iterations of model development, where it is helpful to work with a partial set of rows.


```r
na.strings <- c('','.','NA','#DIV/0!')
grepExpr <- paste(c('.*_dumbbell', '.*_arm', '.*_belt', '.*_forearm', 'classe'), collapse='|')
shrinkage <- 1.0
```


```r
trainingRaw <- read.csv(temporaryFile, na.strings=na.strings)
```

It has the same dimensions, `na.strings` should not interfere with either the number of columns or rows:


```r
dim(trainingRaw)
```

```
## [1] 19622   160
```

And `NA` values are better represented:


```r
str(trainingRaw$kurtosis_roll_dumbbell)
```

```
##  num [1:19622] NA NA NA NA NA NA NA NA NA NA ...
```

```r
str(trainingRaw$kurtosis_yaw_dumbbell)
```

```
##  logi [1:19622] NA NA NA NA NA NA ...
```

We will shrink the original dataset by a factor, to ease processing times. On the final model, `shrinkage` should be equal to one.


```r
trainingRaw <- trainingRaw[,grep(grepExpr, names(trainingRaw))]
if (shrinkage < 1.0) {
    set.seed(123)
    index <- createDataPartition(trainingRaw$classe, p=shrinkage, list=FALSE)
    trainingRaw <- trainingRaw[index,]
}
```

What should have reduced the number of rows, but not columns.


```r
dim(trainingRaw)
```

```
## [1] 19622   153
```

## Filtering Low Variance Features

We decided to take out of the model predictors with a low variance, this is a common technique for situations when we have literally dozens of regressors and using a slow tool like `R`:


```r
set.seed(123)
nzv <- nearZeroVar(trainingRaw)
trainingNotNzv <- trainingRaw[,-nzv]
```

We changed the number of columns, but not the number of rows:


```r
dim(trainingRaw)
```

```
## [1] 19622   153
```

```r
dim(trainingNotNzv)
```

```
## [1] 19622   118
```

We removed **35** low variace columns from the original dataset, specifically these columns: 


```r
setdiff(colnames(trainingRaw), colnames(trainingNotNzv))
```

```
##  [1] "kurtosis_yaw_belt"      "skewness_yaw_belt"     
##  [3] "amplitude_yaw_belt"     "avg_roll_arm"          
##  [5] "stddev_roll_arm"        "var_roll_arm"          
##  [7] "avg_pitch_arm"          "stddev_pitch_arm"      
##  [9] "var_pitch_arm"          "avg_yaw_arm"           
## [11] "stddev_yaw_arm"         "var_yaw_arm"           
## [13] "max_roll_arm"           "min_roll_arm"          
## [15] "min_pitch_arm"          "amplitude_roll_arm"    
## [17] "amplitude_pitch_arm"    "kurtosis_yaw_dumbbell" 
## [19] "skewness_yaw_dumbbell"  "amplitude_yaw_dumbbell"
## [21] "kurtosis_yaw_forearm"   "skewness_yaw_forearm"  
## [23] "max_roll_forearm"       "min_roll_forearm"      
## [25] "amplitude_roll_forearm" "amplitude_yaw_forearm" 
## [27] "avg_roll_forearm"       "stddev_roll_forearm"   
## [29] "var_roll_forearm"       "avg_pitch_forearm"     
## [31] "stddev_pitch_forearm"   "var_pitch_forearm"     
## [33] "avg_yaw_forearm"        "stddev_yaw_forearm"    
## [35] "var_yaw_forearm"
```

## Imputing Missing Values

Now we will deal with `NA` values on the resulting dataset:


```r
sum(is.na(trainingNotNzv))
```

```
## [1] 1250007
```

We have a lot of `NA` values we have to take care of before proceeding on to data analysis, for that we will leverage the troublesome k nearest neighbors mean in `caret` to fill `NA` values:


```r
pp <- preProcess(trainingNotNzv, method=c('knnImpute'))
trainingImputed <- predict(pp, trainingNotNzv[,1:ncol(trainingNotNzv)-1])
trainingImputed$classe <- trainingNotNzv$classe
```

We did not change dimensions of the data set:


```r
dim(trainingImputed)
```

```
## [1] 19622   118
```

But we did change the number of `NA`, what was the intent:


```r
sum(is.na(trainingImputed))
```

```
## [1] 0
```

## Partitioning

We create training and testing partitions by leveraging `createDataPartition` in `caret` following a recommended factor of 75/25% respectivelly:


```r
inTraining <- createDataPartition(trainingImputed$classe, p=.75, list=FALSE)
training <- trainingImputed[inTraining,]
testing <- trainingImputed[-inTraining,]
```


```r
dim(training)
```

```
## [1] 14718   118
```


```r
dim(testing)
```

```
## [1] 4904  118
```

I balanced the alternative of pre-processing done before or after partitioning and decided by the former, mostly for the sake of simplicity. 

This is an interesting consideration that should be better left for separate experimentations.

# Data Analysis

Now we proceed to data analysis _per se_. The analysis is driven mostly by the chosen technique for feature selection. Multiple alternatives are explained in each of the subsequent topics. 

## No Feature Selection

What we call _"no feature selecion"_ is the adoption of a model that does not provide [implicit feature selection][2] in caret.

### Linear Discriminant Analysis


```r
set.seed(123)
fit <- train(classe ~ ., data=training, method='lda', importance=TRUE)
```


```r
set.seed(123)
ldaPred <- predict(fit, testing)
```


```r
cm <- confusionMatrix(testing$classe, ldaPred)
```


```r
cm$overall['Accuracy']
```

```
## Accuracy 
## 0.754894
```


```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1205   38   73   70    9
##          B  107  612  151   36   43
##          C   64   75  656   36   24
##          D   42   24  123  597   18
##          E   23   98   85   63  632
```


```r
importance <- varImp(fit, scale=FALSE)
```


```r
importance
```

```
## ROC curve variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 117)
## 
##                           A      B      C      D      E
## amplitude_pitch_belt 0.8144 0.6967 0.7692 0.7085 0.8144
## pitch_forearm        0.8055 0.6938 0.7129 0.8055 0.7079
## var_pitch_belt       0.7939 0.7057 0.7816 0.7031 0.7939
## stddev_pitch_belt    0.7879 0.6982 0.7649 0.6934 0.7879
## var_roll_belt        0.7696 0.6586 0.7475 0.6546 0.7696
## stddev_roll_belt     0.7674 0.6610 0.7380 0.6386 0.7674
## roll_dumbbell        0.6665 0.6975 0.7633 0.7633 0.6813
## avg_roll_dumbbell    0.6895 0.7178 0.7534 0.7534 0.6699
## accel_forearm_x      0.7459 0.6513 0.6945 0.7459 0.6415
## var_total_accel_belt 0.7453 0.6530 0.7299 0.6452 0.7453
## magnet_arm_x         0.7418 0.6686 0.6633 0.7418 0.7021
## magnet_arm_y         0.7375 0.6258 0.6614 0.7375 0.7085
## accel_arm_x          0.7250 0.6615 0.6430 0.7250 0.6891
## pitch_dumbbell       0.6680 0.7204 0.7204 0.6893 0.6416
## magnet_forearm_x     0.7164 0.6534 0.6187 0.7164 0.6350
## max_roll_dumbbell    0.6789 0.7150 0.7150 0.7102 0.6923
## avg_pitch_dumbbell   0.6761 0.6844 0.7137 0.7137 0.6360
## skewness_yaw_arm     0.7117 0.6618 0.6834 0.6935 0.7117
## var_yaw_belt         0.7077 0.6422 0.6663 0.6390 0.7077
## magnet_belt_y        0.7037 0.6821 0.6856 0.6872 0.7037
```

```r
plot(importance, top=15)
```

![](index_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

As expected, the accuracy of models that provide no feature selection for a wide dataset is comparativelly low due to natural overfitting.

## Implicit Feature Selection

Some models in `caret` provide [implicit feature selection][2] automatically. We should expect a better performance of these models.

### Decision Trees


```r
set.seed(123)
fit <- train(classe ~ ., data=training, method='C5.0Tree', importance=TRUE, prox=TRUE, allowParallel=TRUE,
             trControl=trainControl(method='cv', number=3))
```


```r
set.seed(123)
c50treePred <- predict(fit, testing)
```


```r
cm <- confusionMatrix(testing$classe, c50treePred)
```


```r
cm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9586052
```


```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1366   19    3    4    3
##          B   14  900   23    8    4
##          C    3   28  802   22    0
##          D    7    9   20  755   13
##          E    2   13    4    4  878
```


```r
importance <- varImp(fit, scale=FALSE)
```


```r
importance
```

```
## C5.0Tree variable importance
## 
##   only 20 most important variables shown (out of 117)
## 
##                   Overall
## roll_belt          100.00
## pitch_forearm       91.64
## gyros_belt_z        79.30
## yaw_belt            77.01
## var_accel_arm       72.99
## accel_dumbbell_x    69.15
## pitch_belt          66.99
## gyros_belt_y        63.91
## yaw_arm             63.28
## gyros_dumbbell_y    56.40
## magnet_dumbbell_z   54.91
## accel_dumbbell_y    34.46
## magnet_arm_z        32.41
## accel_dumbbell_z    31.62
## pitch_arm           26.46
## roll_forearm        23.16
## gyros_arm_x         19.20
## avg_roll_dumbbell   19.19
## magnet_belt_z       18.92
## gyros_forearm_y     18.34
```

```r
plot(importance, top=15)
```

![](index_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

### Random Forests


```r
set.seed(123)
rfFit <- randomForest(classe ~ ., data=training, importance=TRUE)
```


```r
set.seed(123)
rfPred <- predict(rfFit, testing)
```


```r
rfCm <- confusionMatrix(testing$classe, rfPred)
```


```r
rfCm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9887847
```


```r
rfCm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    4  940    5    0    0
##          C    0   15  834    6    0
##          D    2    0   18  782    2
##          E    0    0    1    2  898
```


```r
varImpPlot(rfFit, type=2, n.var=15)
```

![](index_files/figure-html/unnamed-chunk-23-1.png)<!-- -->

### Random Forest (caret)

Random forests wrapped by `caret` using `method='rf` never complete. The wrapper is extremelly opaque and the implementation is unstable. We leave the code here for a reference but we never managed to complete a computation even after 5 hours.


```r
set.seed(123)
fit <- train(classe ~ ., data=training, method='rf', importance=TRUE, prox=TRUE, allowParallel=TRUE,
             trControl=trainControl(method='cv', number=2))
```


```r
set.seed(123)
pred <- predict(fit, testing)
```


```r
cm <- confusionMatrix(testing$classe, pred)
```

## Feature Selection by Correlation Threshold

A different way to select features is to remove highly correlated features and apply a model on the resulting set of predictors. Despite the simplicity, the performance of this technique was surprisingly high.

We need a function to perform fittings for different levels of threshold:


```r
correlationThreshold <- function(dfTraining, dfTesting, threshold, method='lda') {
    dfTrainingNumeric <- dfTraining[,1:ncol(dfTraining)-1]
    dfCorr <- cor(dfTrainingNumeric)
    if (threshold < 1.0) {
        highlyCorrColumns <- findCorrelation(dfCorr, threshold)
        dfFiltered <- dfTrainingNumeric[,-highlyCorrColumns]
    } else {
        dfFiltered <- dfTrainingNumeric
    }
    dfFilteredCorr <- cor(dfFiltered)
    dfFiltered$classe <- dfTraining$classe
    mod <- randomForest(classe ~ ., data=dfFiltered)
    pred <- predict(mod, dfTesting)
    cm <- confusionMatrix(dfTesting$classe, pred)
    list(corr=dfFilteredCorr, cm=cm)
}
```

How does the 1.0 correlation threshold, , i.e. all features, look like?


```r
ct10 <- correlationThreshold(training, testing, 1.0)
```


```r
corrplot(ct10$corr, order='hclust', tl.pos='n')
```

![](index_files/figure-html/unnamed-chunk-24-1.png)<!-- -->

Where the overall accuracy is:


```r
ct10$cm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9893964
```

In theory, running a model against a heatmap that is mostly blank will perform better, like this one:


```r
ctLower <- correlationThreshold(training, testing, 0.2)
```


```r
corrplot(ctLower$corr, order='hclust', method='number', tl.pos='n')
```

![](index_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

And the overall accuracy is:


```r
ctLower$cm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9125204
```

What is lower than the accuracy of the same model when no threshold filtering. 

Let's check how accuracies compare based on a varying threshold:


```r
thresholds <- seq(0.1, 0.9, by=0.1)
set.seed(123)
correlations <- lapply(thresholds, function(threshold) {
    correlationThreshold(training, testing, threshold)
})
```


```r
d <- data.frame(threshold=thresholds, 
                accuracy=sapply(correlations, function(x) {x$cm$overall['Accuracy']}),
                kappa=sapply(correlations, function(x) {x$cm$overall['Kappa']}))
ggplot(d, aes(x=threshold,y=accuracy,col=kappa)) +
    geom_point()
```

![](index_files/figure-html/unnamed-chunk-28-1.png)<!-- -->

We can see that accuracies increase to a point, and then we see the effects of overfitting bringing accuracies down.

The highest accuracy with this method is given by:


```r
max(sapply(correlations, function(x) {x$cm$overall['Accuracy']}))
```

```
## [1] 0.9893964
```

## Automatic Feature Selection

We will base our automatic feature selection on `leaps`, the wrappers on `caret` never completed and/or provided opaque error messages, would take too much time to investigate the cause.

We need a function to extract best variables at level `nv` based on regression `reg`:


```r
featureSearchColumns <- function(reg, nv) {
    df <- as.data.frame(reg$outmat)
    df <- as.data.frame(t(df))
    colnames(df) <- sapply(1:ncol(df), function(c) {paste('v',c,sep='')})
    var <- paste('v', nv, sep='')
    columns <- rownames(df[df[var] == '*',])
    columns
}
```

We will only try sequential methods.

### Backward Selection

We start with a full set, and go backwards removing features, keeping track of error on each iteration:


```r
regBackwards <- regsubsets(classe ~ ., data=training, nvmax=50, method='backward')
```

```
## Reordering variables and trying again:
```


```r
b.sum <- summary(regBackwards)
plot(b.sum$bic, type='l', xlab='# features', ylab='bic', main='BIC score by feature inclusion')
```

![](index_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

We can see BIC score is lower nearby 27 features, so let's retrieve the features related to that point:


```r
columns <- featureSearchColumns(b.sum, 27)
columns
```

```
##  [1] "roll_belt"               "pitch_belt"             
##  [3] "yaw_belt"                "stddev_roll_belt"       
##  [5] "var_roll_belt"           "accel_belt_y"           
##  [7] "magnet_belt_y"           "magnet_belt_z"          
##  [9] "gyros_arm_x"             "accel_arm_z"            
## [11] "magnet_arm_y"            "skewness_roll_arm"      
## [13] "skewness_yaw_arm"        "roll_dumbbell"          
## [15] "yaw_dumbbell"            "max_picth_dumbbell"     
## [17] "min_pitch_dumbbell"      "amplitude_roll_dumbbell"
## [19] "total_accel_dumbbell"    "accel_dumbbell_x"       
## [21] "magnet_dumbbell_x"       "magnet_dumbbell_y"      
## [23] "magnet_dumbbell_z"       "pitch_forearm"          
## [25] "total_accel_forearm"     "accel_forearm_z"        
## [27] "magnet_forearm_z"
```


```r
set.seed(123)
fitBackward <- randomForest(training$classe ~ ., data=training[,columns])
```


```r
set.seed(123)
predBackward <- predict(fitBackward, testing)
```


```r
cmBw <- confusionMatrix(testing$classe, predBackward)
```


```r
cmBw$overall['Accuracy']
```

```
##  Accuracy 
## 0.9926591
```


```r
cmBw$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    6  936    7    0    0
##          C    0    9  842    4    0
##          D    0    0    7  796    1
##          E    0    0    0    1  900
```


```r
varImpPlot(fitBackward, type=2, n.var=15)
```

![](index_files/figure-html/unnamed-chunk-35-1.png)<!-- -->

### Forward Selection

We start with an empty set, and go forward adding features, keeping track of error on each iteration:


```r
regForward <- regsubsets(classe ~ ., data=training, nvmax=50, method='forward')
```

```
## Reordering variables and trying again:
```


```r
b.sum <- summary(regForward)
plot(b.sum$bic, type='l', xlab='# features', ylab='bic', main='BIC score by feature inclusion')
```

![](index_files/figure-html/unnamed-chunk-36-1.png)<!-- -->

We can see BIC score is lower nearby 33 features, so let's retrieve the features related to that point:


```r
columns <- featureSearchColumns(b.sum, 33)
columns
```

```
##  [1] "roll_belt"               "pitch_belt"             
##  [3] "yaw_belt"                "total_accel_belt"       
##  [5] "amplitude_pitch_belt"    "var_total_accel_belt"   
##  [7] "stddev_roll_belt"        "accel_belt_y"           
##  [9] "accel_belt_z"            "magnet_belt_x"          
## [11] "magnet_belt_y"           "magnet_belt_z"          
## [13] "pitch_arm"               "gyros_arm_x"            
## [15] "accel_arm_z"             "magnet_arm_x"           
## [17] "magnet_arm_y"            "kurtosis_yaw_arm"       
## [19] "roll_dumbbell"           "yaw_dumbbell"           
## [21] "skewness_pitch_dumbbell" "total_accel_dumbbell"   
## [23] "accel_dumbbell_x"        "accel_dumbbell_z"       
## [25] "magnet_dumbbell_x"       "magnet_dumbbell_y"      
## [27] "magnet_dumbbell_z"       "pitch_forearm"          
## [29] "kurtosis_picth_forearm"  "max_picth_forearm"      
## [31] "total_accel_forearm"     "accel_forearm_z"        
## [33] "magnet_forearm_z"
```


```r
set.seed(123)
fitForward <- randomForest(training$classe ~ ., data=training[,columns])
```


```r
set.seed(123)
predForward <- predict(fitForward, testing)
```


```r
cmFw <- confusionMatrix(testing$classe, predForward)
```


```r
cmFw
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    2    0    0    0
##          B    8  932    9    0    0
##          C    0   10  844    1    0
##          D    0    0    7  795    2
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9918          
##                  95% CI : (0.9889, 0.9942)
##     No Information Rate : 0.2857          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9897          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9943   0.9873   0.9814   0.9975   0.9978
## Specificity            0.9994   0.9957   0.9973   0.9978   0.9998
## Pos Pred Value         0.9986   0.9821   0.9871   0.9888   0.9989
## Neg Pred Value         0.9977   0.9970   0.9960   0.9995   0.9995
## Prevalence             0.2857   0.1925   0.1754   0.1625   0.1839
## Detection Rate         0.2841   0.1900   0.1721   0.1621   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9969   0.9915   0.9893   0.9976   0.9988
```


```r
cmFw$overall['Accuracy']
```

```
##  Accuracy 
## 0.9918434
```


```r
cmFw$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    2    0    0    0
##          B    8  932    9    0    0
##          C    0   10  844    1    0
##          D    0    0    7  795    2
##          E    0    0    0    1  900
```


```r
varImpPlot(fitForward, type=2, n.var=15)
```

![](index_files/figure-html/unnamed-chunk-41-1.png)<!-- -->

## Stacked Models

Stacked models are a fitting done on top of previous fittings, with the hope that the stacked model provides a smaller error.


```r
stackedDf <- data.frame(c50treePred, rfPred, ldaPred, classe=testing$classe)
stackedFit <- train(classe ~ ., method='lda', data=stackedDf)
```


```r
stackedPred = predict(stackedFit, testing)
```


```r
cm <- confusionMatrix(testing$classe, stackedPred)
```


```r
cm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9887847
```


```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    4  940    5    0    0
##          C    0   15  834    6    0
##          D    2    0   18  782    2
##          E    0    0    1    2  898
```


```r
importance <- varImp(fit, scale=FALSE)
```


```r
importance
```

```
## C5.0Tree variable importance
## 
##   only 20 most important variables shown (out of 117)
## 
##                   Overall
## roll_belt          100.00
## pitch_forearm       91.64
## gyros_belt_z        79.30
## yaw_belt            77.01
## var_accel_arm       72.99
## accel_dumbbell_x    69.15
## pitch_belt          66.99
## gyros_belt_y        63.91
## yaw_arm             63.28
## gyros_dumbbell_y    56.40
## magnet_dumbbell_z   54.91
## accel_dumbbell_y    34.46
## magnet_arm_z        32.41
## accel_dumbbell_z    31.62
## pitch_arm           26.46
## roll_forearm        23.16
## gyros_arm_x         19.20
## avg_roll_dumbbell   19.19
## magnet_belt_z       18.92
## gyros_forearm_y     18.34
```

```r
plot(importance, top=15)
```

![](index_files/figure-html/unnamed-chunk-45-1.png)<!-- -->

# Predicting Raw Test Dataset


```r
urlTesting <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
temporaryFile <- 'pml-testing.csv'
download.file(urlTesting, destfile=temporaryFile, method="curl")
testingRaw <- read.csv(temporaryFile, na.strings=na.strings) 
```


```r
dim(testing)
```

```
## [1] 4904  118
```

```r
dim(testingRaw)
```

```
## [1]  20 160
```


```r
dim(testingRaw)
```

```
## [1]  20 160
```


```r
setdiff(colnames(testing), colnames(testingRaw))
```

```
## [1] "classe"
```


```r
setdiff(colnames(testingRaw), colnames(testing))
```

```
##  [1] "X"                      "user_name"             
##  [3] "raw_timestamp_part_1"   "raw_timestamp_part_2"  
##  [5] "cvtd_timestamp"         "new_window"            
##  [7] "num_window"             "kurtosis_yaw_belt"     
##  [9] "skewness_yaw_belt"      "amplitude_yaw_belt"    
## [11] "avg_roll_arm"           "stddev_roll_arm"       
## [13] "var_roll_arm"           "avg_pitch_arm"         
## [15] "stddev_pitch_arm"       "var_pitch_arm"         
## [17] "avg_yaw_arm"            "stddev_yaw_arm"        
## [19] "var_yaw_arm"            "max_roll_arm"          
## [21] "min_roll_arm"           "min_pitch_arm"         
## [23] "amplitude_roll_arm"     "amplitude_pitch_arm"   
## [25] "kurtosis_yaw_dumbbell"  "skewness_yaw_dumbbell" 
## [27] "amplitude_yaw_dumbbell" "kurtosis_yaw_forearm"  
## [29] "skewness_yaw_forearm"   "max_roll_forearm"      
## [31] "min_roll_forearm"       "amplitude_roll_forearm"
## [33] "amplitude_yaw_forearm"  "avg_roll_forearm"      
## [35] "stddev_roll_forearm"    "var_roll_forearm"      
## [37] "avg_pitch_forearm"      "stddev_pitch_forearm"  
## [39] "var_pitch_forearm"      "avg_yaw_forearm"       
## [41] "stddev_yaw_forearm"     "var_yaw_forearm"       
## [43] "problem_id"
```

## No Pre-Processing

We will use Random Forests, with an expected prediction rate of:


```r
rfCm$overall['Accuracy']
```

```
##  Accuracy 
## 0.9887847
```

All predicted values are `NA`:


```r
noPreProcessingPred <- predict(rfFit, testingRaw)
```


```r
testingSubset <- testingRaw[,intersect(colnames(testing), colnames(testingRaw))]
#testingSubset <- testingSubset[,colSums(is.na(testingSubset)) < nrow(testingSubset)]
```

A few parameters from training set


```r
trainingSubset <- training[,intersect(colnames(training), colnames(testingSubset))]
```


```r
sum(is.na(trainingSubset))
```

```
## [1] 0
```

```r
mean(trainingSubset[!is.na(trainingSubset)])
```

```
## [1] 0.02952278
```

```r
sd(trainingSubset[!is.na(trainingSubset)])
```

```
## [1] 0.8242742
```

## Standardizing


```r
testingSubset <- testingRaw[,intersect(colnames(testing), colnames(testingRaw))]
#testingSubset <- testingSubset[,colSums(is.na(testingSubset)) < nrow(testingSubset)]
```


```r
sum(is.na(testingSubset))
```

```
## [1] 1300
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 25.81951
```


```r
for (c in colnames(testingSubset)) {
    testingSubset[is.na(testingSubset[,c]),c] <- mean(training[,c])
    testingSubset[,c] <- (testingSubset[,c] - mean(training[,c])) / sd(training[,c])
}
```


```r
sum(is.na(testingSubset))
```

```
## [1] 0
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 11.39844
```

```r
sd(testingSubset[!is.na(testingSubset)])
```

```
## [1] 146.53
```



```r
standardizedPred <- predict(rfFit, testingSubset)
```

## KNN Imputing Based on Training Points, Forward Fitting Model


```r
testingSubset <- testingRaw[,intersect(colnames(testing), colnames(testingRaw))]
```


```r
sum(is.na(testingSubset))
```

```
## [1] 1300
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 25.81951
```


```r
for (c in colnames(testingSubset)) {
    testingSubset[is.na(testingSubset[,c]),c] <- mean(training[,c])
}
```


```r
pp <- preProcess(training, method=c('knnImpute'))
testingSubset <- predict(pp, testingSubset)
```


```r
sum(is.na(testingSubset))
```

```
## [1] 0
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 11.39844
```

```r
sd(testingSubset[!is.na(testingSubset)])
```

```
## [1] 146.53
```


```r
knnOnTrainingFitForwardPred <- predict(fitForward, testingSubset)
```

## KNN Imputing Based on Testing Points


## KNN Imputing Based on Training Points


```r
testingSubset <- testingRaw[,intersect(colnames(testing), colnames(testingRaw))]
```


```r
sum(is.na(testingSubset))
```

```
## [1] 1300
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 25.81951
```


```r
for (c in colnames(testingSubset)) {
    testingSubset[is.na(testingSubset[,c]),c] <- mean(training[,c])
}
```


```r
pp <- preProcess(training, method=c('knnImpute'))
testingSubset <- predict(pp, testingSubset)
```


```r
sum(is.na(testingSubset))
```

```
## [1] 0
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 11.39844
```

```r
sd(testingSubset[!is.na(testingSubset)])
```

```
## [1] 146.53
```


```r
knnOnTrainingPred <- predict(rfFit, testingSubset)
```

## KNN Imputing Based on Testing Points


```r
testingSubset <- testingRaw[,intersect(colnames(testing), colnames(testingRaw))]
```


```r
sum(is.na(testingSubset))
```

```
## [1] 1300
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] 25.81951
```


```r
for (c in colnames(testingSubset)) {
    testingSubset[is.na(testingSubset[,c]),c] <- mean(training[,c])
}
```


```r
pp <- preProcess(testingSubset, method=c('knnImpute'))
testingSubset <- predict(pp, testingSubset)
```


```r
sum(is.na(testingSubset))
```

```
## [1] 0
```

```r
mean(testingSubset[!is.na(testingSubset)])
```

```
## [1] -1.171031e-17
```

```r
sd(testingSubset[!is.na(testingSubset)])
```

```
## [1] 0.6499252
```


```r
knnOnTestingPred <- predict(rfFit, testingSubset)
```

# Final Observations



The idea behind `caret` is cool, but at this point it seems it has a long way to go to a full blown ME library. There is plenty of documentation but it is mostly short and examples cannot be readily reproduced. Runtime support could be better, error messages are opaque and to little help. You have to constant rely on google searches that are often to no end and a time sink.

Most of the algorithms could not be used on a reasonably large dataset like this one. I had to use the underlying API most of the times.

# References

We used a number of references for this analysis, specifically:

* [1] "Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6."

[1]: http://groupware.les.inf.puc-rio.br/public/papers/2012.Ugulino.WearableComputing.HAR.Classifier.RIBBON.pdf 
[2]: http://topepo.github.io/caret/Implicit_Feature_Selection.html
