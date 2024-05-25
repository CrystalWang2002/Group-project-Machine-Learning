rm(list=ls())


RMSE <- function(pred, truth){ 
  return(sqrt(mean((truth - pred)^2)))
} 


library(githubinstall)
library(HDeconometrics)
library(sandwich) #library to estimate variance for DM test regression using NeweyWest()
library(readr)
library(caret)
library(pls)
library(randomForest)
library(MASS)
library(ipred)
library(dplyr)
library(zoo)


setwd("/Users/caimingyi/Desktop/project_04/data")
ukdata = read.csv("UK.csv")
ukdata$Date = as.Date(ukdata$Date, format="%Y-%m-%d")
data = zoo(ukdata[,-1], order.by = ukdata$Date)
data = as.data.frame(data)
data = data[,-c(7:9,11:17)]
names(data) <- gsub(" ", "_", names(data))
data = data[c("UNEMP_RATE", setdiff(names(data), "UNEMP_RATE"))]
rm(ukdata)
data <- as.matrix(data)
Y=data[,1] 
#number of out-of-sample observations (test window)
nprev=140
oos_Y=tail(Y,nprev) #auxiliary:get the out-of-sample true values (last 180 obs. using tail())


set.seed(5304)


########################################################
###Basic Results
#########################################################
#AR(BIC)
source("func-ar.R")
ar1c=ar.rolling.window(data,nprev,1,1,"bic") 
ar3c=ar.rolling.window(data,nprev,1,3,"bic") 
ar6c=ar.rolling.window(data,nprev,1,6,"bic")
ar12c=ar.rolling.window(data,nprev,1,12,"bic")
#AR forecasts RMSE:
ar.rmse1.c=ar1c$errors[1]
ar.rmse3.c=ar3c$errors[1]
ar.rmse6.c=ar6c$errors[1]
ar.rmse12.c=ar12c$errors[1]




#Random Forest
##Random Forest-rolling window
source("func-rf.R")
#index for dependent variable = 1
rf1c3=rf.rolling.window(data,nprev,1,1)
rf3c3=rf.rolling.window(data,nprev,1,3)
rf6c3=rf.rolling.window(data,nprev,1,6)
rf12c3=rf.rolling.window(data,nprev,1,12) 
#See the RMSE:
rf3.rmse1=rf1c3$errors[1]
rf3.rmse3=rf3c3$errors[1]
rf3.rmse6=rf6c3$errors[1] 
rf3.rmse12=rf12c3$errors[1] 



#ElNET(BIC)
source("func-lasso.R")
alpha=0.5
elasticnet1c=lasso.rolling.window(data,nprev,1,1,alpha,IC="bic")
elasticnet3c=lasso.rolling.window(data,nprev,1,3,alpha,IC="bic")
elasticnet6c=lasso.rolling.window(data,nprev,1,6,alpha,IC="bic")
elasticnet12c=lasso.rolling.window(data,nprev,1,12,alpha,IC="bic")
#See the RMSE:
elnet_bic_all.rmse1=elasticnet1c$errors[1]
elnet_bic_all.rmse3=elasticnet3c$errors[1]
elnet_bic_all.rmse6=elasticnet6c$errors[1]
elnet_bic_all.rmse12=elasticnet12c$errors[1]









########################################################
###Diebold-Mariano (DM) tests
#########################################################

#####################################################
#Random Forest vs. AR(BIC) benchmark
#####################################################

#Compute squared loss for different horizons (RF)
lrf1c=(oos_Y-rf1c3$pred)^2
lrf3c=(oos_Y-rf3c3$pred)^2
lrf6c=(oos_Y-rf6c3$pred)^2
lrf12c=(oos_Y-rf12c3$pred)^2

#Compute squared loss for different horizons (AR)
lar1c=(oos_Y-ar1c$pred)^2
lar3c=(oos_Y-ar3c$pred)^2
lar6c=(oos_Y-ar6c$pred)^2
lar12c=(oos_Y-ar12c$pred)^2

#Compute loss differentials (d_t) for different horizons (AR-RF)
darrf1=lar1c-lrf1c
darrf3=lar3c-lrf3c
darrf6=lar6c-lrf6c
darrf12=lar12c-lrf12c

#Create ts object containing loss differentials
dtarrf.ts=ts(cbind(darrf1,darrf3,darrf6,darrf12), start=c(2010,8), end=c(2021,12), freq=12)
colnames(dtarrf.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtarrf.ts, main="Loss differential AR-RF",cex.axis=1.8)




#DM regressions

#Regress d_t (AR-RF) for 1-step forecasts on a constant - get estimate of mean(d_t)
dmarrf1=lm(darrf1~1) #regression
acf(dmarrf1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
#using the (nprev)^(1/3) rule of thumb for lag choice:
#(nprev)^(1/3) = 140^(1/3) = 5.192494 -choose 5
dmarrf1$coefficients/sqrt(NeweyWest(dmarrf1,lag=5)) #form the DM t-statistic


#3-step forecast test
dmarrf3=lm(darrf3~1)
acf(dmarrf3$residuals)
dmarrf3$coefficients/sqrt(NeweyWest(dmarrf3,lag=5))

#6-step forecast test
dmarrf6=lm(darrf6~1)
acf(dmarrf6$residuals)
dmarrf6$coefficients/sqrt(NeweyWest(dmarrf6,lag=5))

#12-step forecast test
dmarrf12=lm(darrf12~1)
acf(dmarrf12$residuals)
dmarrf12$coefficients/sqrt(NeweyWest(dmarrf12,lag=5))









#####################################################
#Set up for the tests of ELNET(BIC) vs. Random Forest
#####################################################

#Compute squared loss for different horizons (ELNET(BIC))
lel1c=(oos_Y-elasticnet1c$pred)^2
lel3c=(oos_Y-elasticnet1c$pred)^2
lel6c=(oos_Y-elasticnet1c$pred)^2
lel12c=(oos_Y-elasticnet1c$pred)^2

#Compute loss differentials (d_t) for different horizons (PLASSO-RF)
delrf1=lel1c-lrf1c
delrf3=lel3c-lrf3c
delrf6=lel6c-lrf6c
delrf12=lel12c-lrf12c

#Create ts object containing loss differentials
dtelrf.ts=ts(cbind(delrf1,delrf3,delrf6,delrf12), start=c(2010,8), end=c(2021,12), freq=12)
#Plot them to examine stationarity:
colnames(dtelrf.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
plot.ts(dtelrf.ts, main="Loss differential ElNET(BIC)-RF",cex.axis=1.8)





#DM regressions

#ELNET(BIC)-RF

#1-step forecast test
dmelrf1=lm(delrf1~1)
acf(dmelrf1$residuals)
dmelrf1$coefficients/sqrt(NeweyWest(dmelrf1,lag=5))

#3-step forecast test
dmelrf3=lm(delrf3~1)
acf(dmelrf3$residuals)
dmelrf3$coefficients/sqrt(NeweyWest(dmelrf3,lag=5))

#6-step forecast test
dmelrf6=lm(delrf6~1)
acf(dmelrf6$residuals)
dmelrf6$coefficients/sqrt(NeweyWest(dmelrf6,lag=5))

#12-step forecast test
dmelrf12=lm(delrf12~1)
acf(dmelrf12$residuals)
dmelrf12$coefficients/sqrt(NeweyWest(dmelrf12,lag=5))





