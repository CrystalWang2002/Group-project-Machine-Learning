rm(list=ls())

#Auxiliary function to compute root MSE (same as MSE before, but with square root):
RMSE <- function(pred, truth){ #start and end body of the function by { } - same as a loop 
  return(sqrt(mean((truth - pred)^2)))
} #end function with a return(output) statement. Here we can go straight to return because the object of interest is a simple function of inputs


#install.packages("githubinstall") #this package is needed to install packages from GitHub (a popular code repository)
#install.packages("sandwich")
#githubinstall("HDeconometrics") #install Medeiros et al's package, you will be prompted to say "Yes" to confirm the name of the installed package:
library(githubinstall)
library(HDeconometrics)
library(sandwich) #library to estimate variance for DM test regression using NeweyWest()




#Set working directory and load data:
setwd("/Users/caimingyi/Desktop/project_04/data")
df=read.csv("UK.csv")
df = df[c("UNEMP_RATE", setdiff(names(df), "UNEMP_RATE"))]
df = df[c("Date", setdiff(names(df), "Date"))]

#get the y variable - Unemployment Rate
Y=df[,2] 
#number of out-of-sample observations (test window)
nprev=140
oos_Y=tail(Y,nprev) #auxiliary:get the out-of-sample true values (last 180 obs. using tail())


set.seed(5304)

########################################################################
#Random Walk
########################################################################
#create lags
rwtemp=embed(Y,13)
#Simple RW forecast:
rw1c=tail(rwtemp[,2],nprev)
rw3c=tail(rwtemp[,4],nprev)
rw6c=tail(rwtemp[,7],nprev)
rw12c=tail(rwtemp[,13],nprev)
#Collect RMSE's for randomw walk:
rw.rmse1=RMSE(oos_Y,rw1c)
print(rw.rmse1)
rw.rmse3=RMSE(oos_Y,rw3c)
print(rw.rmse3)
rw.rmse6=RMSE(oos_Y,rw6c)
print(rw.rmse6)
rw.rmse12=RMSE(oos_Y,rw12c)
print(rw.rmse12)






########################################################################
#AR
########################################################################
source("func-ar.R")
#bic
ar1c=ar.rolling.window(df,nprev,2,1,"bic") #1-step AR forecast
ar3c=ar.rolling.window(df,nprev,2,3,"bic") #3-step AR forecast
ar6c=ar.rolling.window(df,nprev,2,6,"bic") #6-step AR forecast
ar12c=ar.rolling.window(df,nprev,2,12,"bic") #12-step AR forecast


#ar-4lags
ar1f=ar.rolling.window(df,nprev,2,1,"fixed") #1-step AR forecast
ar3f=ar.rolling.window(df,nprev,2,3,"fixed") #3-step AR forecast
ar6f=ar.rolling.window(df,nprev,2,6,"fixed") #6-step AR forecast
ar12f=ar.rolling.window(df,nprev,2,12,"fixed") #12-step AR forecast





#AR forecasts RMSE:
#bic
ar.rmse1.c=ar1c$errors[1]
print(ar.rmse1.c)
ar.rmse3.c=ar3c$errors[1]
print(ar.rmse3.c)
ar.rmse6.c=ar6c$errors[1]
print(ar.rmse6.c)
ar.rmse12.c=ar12c$errors[1]
print(ar.rmse12.c)

#ar-4lags
ar.rmse1.f=ar1f$errors[1]
print(ar.rmse1.f)
ar.rmse3.f=ar3f$errors[1]
print(ar.rmse3.f)
ar.rmse6.f=ar6f$errors[1]
print(ar.rmse6.f)
ar.rmse12.f=ar12f$errors[1]
print(ar.rmse12.f)




#Benchmark forecast graphics:

#Plot benchmark coefficients
#Here I use plot.ts(), which plots time series objects that are dated

#Syntax: ts(object, start=startdate, end=enddate, freq=frequency (periods per year))
arcoef.ts=ts(ar1f$coef, start=c(1998,1), end=c(2021,12), freq=12)
colnames(arcoef.ts)=c("Constant","Phi1","Phi2","Phi3","Phi4") #name columns to distinguish plots

#Plot all the coefficients over time (plot.ts() same as plot, but for tme series objects):
plot.ts(arcoef.ts, main="AR regression coefficients", cex.axis=0.5)

#Similarly, I create ts objects out of 1-step and 12-step benchmark forecasts
bench1.ts=ts(cbind(rw1c,ar1f$pred,ar1c$pred,oos_Y), start=c(1998,1), end=c(2021,12), freq=12)
colnames(bench1.ts)=c("RW","AR(4)","AR(BIC)","True Value")

bench3.ts=ts(cbind(rw3c,ar3f$pred,ar3c$pred,oos_Y), start=c(1998,1), end=c(2021,12), freq=12)
colnames(bench3.ts)=c("RW","AR(4)","AR(BIC)","True Value")

bench6.ts=ts(cbind(rw6c,ar6f$pred,ar6c$pred,oos_Y), start=c(1998,1), end=c(2021,12), freq=12)
colnames(bench6.ts)=c("RW","AR(4)","AR(BIC)","True Value")

bench12.ts=ts(cbind(rw12c,ar12f$pred,ar12c$pred,oos_Y), start=c(1998,1), end=c(2021,12), freq=12)
colnames(bench12.ts)=c("RW","AR(4)","AR(BIC)","True Value")

#Plot 1-step forecasts:
plot.ts(bench1.ts[,1], main="1-step Benchmark forecasts", cex.axis=0.5, lwd=0.8, col="blue", ylab="Unemployment Rate")
points(bench1.ts[,2], type="l", col="red",lwd=0.8)
points(bench1.ts[,3], type="l", col="green",lwd=0.8)
points(bench1.ts[,4], type="l", col="black",lwd=0.8)
legend("bottomleft",legend=c("RW","AR(4)","AR(BIC)","Unemployment Rate"), lty=1, col=c("blue", "red","green","black"), lwd=0.8, cex=0.75)

#Plot 3-step forecasts:
plot.ts(bench3.ts[,1], main="3-step Benchmark forecasts", cex.axis=0.5, lwd=0.8, col="blue", ylab="Unemployment Rate")
points(bench3.ts[,2], type="l", col="red",lwd=0.8)
points(bench1.ts[,3], type="l", col="green",lwd=0.8)
points(bench3.ts[,4], type="l", col="black",lwd=0.8)
legend("bottomleft",legend=c("RW","AR(4)","AR(BIC)","Unemployment Rate"), lty=1, col=c("blue", "red", "green","black"), lwd=0.8, cex=0.75)

#Plot 6-step forecasts:
plot.ts(bench6.ts[,1], main="6-step Benchmark forecasts", cex.axis=0.5, lwd=0.8, col="blue", ylab="Unemployment Rate")
points(bench6.ts[,2], type="l", col="red",lwd=0.8)
points(bench1.ts[,3], type="l", col="green",lwd=0.8)
points(bench6.ts[,4], type="l", col="black",lwd=0.8)
legend("bottomleft",legend=c("RW","AR(4)","AR(BIC)","Unemployment Rate"), lty=1, col=c("blue", "red", "green","black"), lwd=0.8, cex=0.75)

#Plot 12-step forecasts:
plot.ts(bench12.ts[,1], main="12-step Benchmark forecasts", cex.axis=0.5, lwd=0.8, col="blue", ylab="Unemployment Rate")
points(bench12.ts[,2], type="l", col="red",lwd=0.8)
points(bench1.ts[,3], type="l", col="green",lwd=0.8)
points(bench12.ts[,4], type="l", col="black",lwd=0.8)
legend("bottomleft",legend=c("RW","AR(4)","AR(BIC)","Unemployment Rate"), lty=1, col=c("blue", "red", "green","black"), lwd=0.8, cex=0.75)


















