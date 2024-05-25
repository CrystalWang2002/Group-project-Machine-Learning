rm(list=ls())

RMSE <- function(pred, truth){ 
  return(sqrt(mean((truth - pred)^2)))
} 

library(readr)
library(caret)
library(pls)
library(randomForest)
library(MASS)
library(ipred)
library(dplyr)
library(zoo)

#load data, covert to ts data
setwd("D:/nus/ECA5304/dm测试/code")
ukdata = read.csv("UK.csv")
ukdata$Date = as.Date(ukdata$Date, format="%Y-%m-%d")
data = zoo(ukdata[,-1], order.by = ukdata$Date)
data = as.data.frame(data)
data = data[,-c(7:9,11:17)]
names(data) <- gsub(" ", "_", names(data))
data = data[c("UNEMP_RATE", setdiff(names(data), "UNEMP_RATE"))]

save(data,file = "rawdata.csv")

####Modelling: only base on sentiment####
data1 = data[,c(1:7)]

################################fixed window############################
data1lag = data1
lag=embed(data1lag$UNEMP_RATE,5)
data1lag <- cbind(data1lag[(5:nrow(data1lag)), ], lag1 = lag[, 2], lag2 = lag[, 3], lag3 = lag[, 4], lag4 = lag[, 5])

ntrain=148 
tr <- 1:ntrain
test <- (ntrain+1):nrow(data1lag) 

##Bagging-fixed window

bagging_model <- randomForest(UNEMP_RATE ~ ., data = data1lag[tr,], ntree = 5000, mtr=6)
print(bagging_model)

bagging_pred <- predict(bagging_model, newdata = data1lag[test,])
bagging_rmse_senti_fixed = RMSE(bagging_pred, data1lag[test,]$UNEMP_RATE)
print(bagging_rmse_senti_fixed)

##Random Forest-fixed window
set.seed(5304)

rf_model <- randomForest(UNEMP_RATE ~ ., data = data1lag[tr,], ntree = 5000, mtr=2)
print(rf_model)

rf_pred <- predict(rf_model, newdata = data1lag[test,])
rf_rmse_senti_fixed = RMSE(rf_pred, data1lag[test,]$UNEMP_RATE)
print(rf_rmse_senti_fixed)






################################rolling window##########################
nprev=140

source("func-bagging.R")
set.seed(5304)
#index for dependent variable = 1
bagging1c1=bagging.rolling.window(data1,nprev,1,1)
bagging3c1=bagging.rolling.window(data1,nprev,1,3)
bagging6c1=bagging.rolling.window(data1,nprev,1,6)
bagging12c1=bagging.rolling.window(data1,nprev,1,12)

#See the RMSE:
bagging1.rmse1=bagging1c1$errors[1]
bagging1.rmse3=bagging3c1$errors[1] 
bagging1.rmse6=bagging6c1$errors[1] 
bagging1.rmse12=bagging12c1$errors[1]


print(bagging1.rmse1)
print(bagging1.rmse3)
print(bagging1.rmse6)
print(bagging1.rmse12)




##Random Forest-rolling window
set.seed(5304)
source("func-rf.R")

#index for dependent variable = 1
rf1c1=rf.rolling.window(data1,nprev,1,1)
rf3c1=rf.rolling.window(data1,nprev,1,3)
rf6c1=rf.rolling.window(data1,nprev,1,6)
rf12c1=rf.rolling.window(data1,nprev,1,12) 

#See the RMSE:
rf1.rmse1=rf1c1$errors[1]
rf1.rmse3=rf3c1$errors[1]
rf1.rmse6=rf6c1$errors[1]
rf1.rmse12=rf12c1$errors[1]


print(rf1.rmse1)
print(rf1.rmse3)
print(rf1.rmse6)
print(rf1.rmse12)



####Modelling: only base on macro var####
data2 = data[,-c(2:7)]

################################fixed window############################
data2lag = data2
data2lag <- cbind(data2lag[(5:nrow(data2lag)), ], lag1 = lag[, 2], lag2 = lag[, 3], lag3 = lag[, 4], lag4 = lag[, 5])

##Bagging-fixed window
set.seed(5304) 

bagging_model <- randomForest(UNEMP_RATE~ ., data = data2lag[tr,], ntree = 5000, mtr=ncol(data2lag)-1)
print(bagging_model)

bagging_pred <- predict(bagging_model, newdata = data2lag[test,])
bagging_rmse_macro_fixed = RMSE(bagging_pred, data2lag[test,]$UNEMP_RATE)
print(bagging_rmse_macro_fixed)




##Random Forest-fixed window
set.seed(5304)

rf_model <- randomForest(UNEMP_RATE ~ ., data = data2lag[tr,], ntree = 5000, mtr=round((ncol(data2lag)-1)/3))
print(rf_model)

rf_pred <- predict(rf_model, newdata = data2lag[test,])
rf_rmse_macro_fixed = RMSE(rf_pred, data2lag[test,]$UNEMP_RATE)
print(rf_rmse_macro_fixed)




################################rolling window##########################

##Bagging-rolling window

#index for dependent variable = 1
bagging1c2=bagging.rolling.window(data2,nprev,1,1)
bagging3c2=bagging.rolling.window(data2,nprev,1,3)
bagging6c2=bagging.rolling.window(data2,nprev,1,6)
bagging12c2=bagging.rolling.window(data2,nprev,1,12)

#See the RMSE:
bagging2.rmse1=bagging1c2$errors[1]
bagging2.rmse3=bagging3c2$errors[1]
bagging2.rmse6=bagging6c2$errors[1] 
bagging2.rmse12=bagging12c2$errors[1] 

print(bagging2.rmse1)
print(bagging2.rmse3)
print(bagging2.rmse6)
print(bagging2.rmse12)



##Random Forest-rolling window

#index for dependent variable = 1
rf1c2=rf.rolling.window(data2,nprev,1,1)
rf3c2=rf.rolling.window(data2,nprev,1,3)
rf6c2=rf.rolling.window(data2,nprev,1,6)
rf12c2=rf.rolling.window(data2,nprev,1,12) 

#See the RMSE:
rf2.rmse1=rf1c2$errors[1]
rf2.rmse3=rf3c2$errors[1]
rf2.rmse6=rf6c2$errors[1] 
rf2.rmse12=rf12c2$errors[1] 

print(rf2.rmse1)
print(rf2.rmse3)
print(rf2.rmse6)
print(rf2.rmse12)


####Modelling: combine sentiment and other control var####
data3 = data

################################fixed window############################
data3lag = data3
data3lag <- cbind(data3lag[(5:nrow(data3lag)), ], lag1 = lag[, 2], lag2 = lag[, 3], lag3 = lag[, 4], lag4 = lag[, 5])


##Bagging-fixed window
set.seed(5304) 

bagging_model <- randomForest(UNEMP_RATE ~ ., data = data3lag[tr,], ntree = 5000, mtr=ncol(data3lag)-1)
print(bagging_model)

bagging_pred <- predict(bagging_model, newdata = data3lag[test,])
bagging_rmse_all_fixed = RMSE(bagging_pred, data3lag[test,]$UNEMP_RATE)
print(bagging_rmse_all_fixed)




##Random Forest-fixed window
set.seed(5304)

rf_model <- randomForest(UNEMP_RATE ~ ., data = data3lag[tr,], ntree = 5000, mtr=round((ncol(data3lag)-1)/3))
print(rf_model)

rf_pred <- predict(rf_model, newdata = data3lag[test,])
rf_rmse_all_fixed = RMSE(rf_pred, data3lag[test,]$UNEMP_RATE)
print(rf_rmse_all_fixed)






################################rolling window##########################

##Bagging-rolling window

#index for dependent variable = 1
bagging1c3=bagging.rolling.window(data3,nprev,1,1)
bagging3c3=bagging.rolling.window(data3,nprev,1,3)
bagging6c3=bagging.rolling.window(data3,nprev,1,6)
bagging12c3=bagging.rolling.window(data3,nprev,1,12)

#See the RMSE:
bagging3.rmse1=bagging1c3$errors[1]
bagging3.rmse3=bagging3c3$errors[1]
bagging3.rmse6=bagging6c3$errors[1]
bagging3.rmse12=bagging12c3$errors[1] 

print(bagging3.rmse1)
print(bagging3.rmse3)
print(bagging3.rmse6)
print(bagging3.rmse12)


##Random Forest-rolling window

#index for dependent variable = 1
rf1c3=rf.rolling.window(data3,nprev,1,1)
rf3c3=rf.rolling.window(data3,nprev,1,3)
rf6c3=rf.rolling.window(data3,nprev,1,6)
rf12c3=rf.rolling.window(data3,nprev,1,12) 

#See the RMSE:
rf3.rmse1=rf1c3$errors[1]
rf3.rmse3=rf3c3$errors[1]
rf3.rmse6=rf6c3$errors[1] 
rf3.rmse12=rf12c3$errors[1] 

print(rf3.rmse1)
print(rf3.rmse3)
print(rf3.rmse6)
print(rf3.rmse12)