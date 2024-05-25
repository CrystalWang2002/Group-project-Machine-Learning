rm(list=ls())
RMSE <- function(pred, truth){
  return(sqrt(mean((truth - pred)^2)))
}

library(HDeconometrics)
library(sandwich)
library(randomForest)


setwd("/Users/caimingyi/Desktop/project_04/data")
df=read.csv("UK.csv")


uk_data <- read.csv("UK.csv")
uk_data <- uk_data[,-1]
uk_data <- subset(uk_data, select = -c(EMP, EMP_PART, EMP_TEMP, UNEMP_DURA_6mth, `UNEMP_DURA_6.12mth`, `UNEMP_DURA_12mth.`, `UNEMP_DURA_24mth.`, EMP_RATE, EMP_ACT, EMP_ACT_RATE))
y <- uk_data$UNEMP_RATE
X <- as.matrix(uk_data[, -which(names(uk_data) == "UNEMP_RATE")])

all <- cbind(y, X)
#extract sentiment variables
senti = all[,1:7]
out_senti = all[, -c(2:7)]

nprev=140

set.seed(5304)


#all

#1.1.1_lasso_BIC
source("func-lasso.R")
alpha=1
lasso1c=lasso.rolling.window(all,nprev,1,1,alpha,IC="bic")
lasso3c=lasso.rolling.window(all,nprev,1,3,alpha,IC="bic")
lasso6c=lasso.rolling.window(all,nprev,1,6,alpha,IC="bic")
lasso12c=lasso.rolling.window(all,nprev,1,12,alpha,IC="bic")
lasso_bic_all.rmse1=lasso1c$errors[1]
lasso_bic_all.rmse3=lasso3c$errors[1]
lasso_bic_all.rmse6=lasso6c$errors[1]
lasso_bic_all.rmse12=lasso12c$errors[1]
print(lasso_bic_all.rmse1)
print(lasso_bic_all.rmse3)
print(lasso_bic_all.rmse6)
print(lasso_bic_all.rmse12)








#1.1.2_lasso_AIC
lasso1ca=lasso.rolling.window(all,nprev,1,1,alpha,IC="aic")
lasso3ca=lasso.rolling.window(all,nprev,1,3,alpha,IC="aic")
lasso4ca=lasso.rolling.window(all,nprev,1,3,alpha,IC="aic")
lasso6ca=lasso.rolling.window(all,nprev,1,6,alpha,IC="aic")
lasso12ca=lasso.rolling.window(all,nprev,1,12,alpha,IC="aic")
lasso_aic_all.rmse1=lasso1ca$errors[1]
lasso_aic_all.rmse3=lasso3ca$errors[1]
lasso_aic_all.rmse4=lasso3ca$errors[1]
lasso_aic_all.rmse6=lasso6ca$errors[1]
lasso_aic_all.rmse12=lasso12ca$errors[1]
print(lasso_aic_all.rmse1)
print(lasso_aic_all.rmse3)
print(lasso_aic_all.rmse4)
print(lasso_aic_all.rmse6)
print(lasso_aic_all.rmse12)








#1.1.3_lasso_AICc
lasso1caic=lasso.rolling.window(all,nprev,1,1,alpha,IC="aicc")
lasso3caic=lasso.rolling.window(all,nprev,1,3,alpha,IC="aicc")
lasso6caic=lasso.rolling.window(all,nprev,1,6,alpha,IC="aicc")
lasso12caic=lasso.rolling.window(all,nprev,1,12,alpha,IC="aicc")
lasso_ac_all.rmse1=lasso1caic$errors[1]
lasso_ac_all.rmse3=lasso3caic$errors[1]
lasso_ac_all.rmse6=lasso6caic$errors[1]
lasso_ac_all.rmse12=lasso12caic$errors[1]
print(lasso_ac_all.rmse1)
print(lasso_ac_all.rmse3)
print(lasso_ac_all.rmse6)
print(lasso_ac_all.rmse12)







#1.1.4_Post-LASSO_BIC
pols.lasso1c=pols.rolling.window(all,nprev,1,1,lasso1c$coef)
pols.lasso3c=pols.rolling.window(all,nprev,1,3,lasso3c$coef)
pols.lasso6c=pols.rolling.window(all,nprev,1,6,lasso6c$coef)
pols.lasso12c=pols.rolling.window(all,nprev,1,12,lasso12c$coef)
plasso_all.rmse1=pols.lasso1c$errors[1]
plasso_all.rmse3=pols.lasso3c$errors[1]
plasso_all.rmse6=pols.lasso6c$errors[1]
plasso_all.rmse12=pols.lasso12c$errors[1]
print(plasso_all.rmse1)
print(plasso_all.rmse3)
print(plasso_all.rmse6)
print(plasso_all.rmse12)







#1.2.1_elastic net_BIC
alpha=0.5
elasticnet1c=lasso.rolling.window(all,nprev,1,1,alpha,IC="bic")
elasticnet3c=lasso.rolling.window(all,nprev,1,3,alpha,IC="bic")
elasticnet6c=lasso.rolling.window(all,nprev,1,6,alpha,IC="bic")
elasticnet12c=lasso.rolling.window(all,nprev,1,12,alpha,IC="bic")
elnet_bic_all.rmse1=elasticnet1c$errors[1]
elnet_bic_all.rmse3=elasticnet3c$errors[1]
elnet_bic_all.rmse6=elasticnet6c$errors[1]
elnet_bic_all.rmse12=elasticnet12c$errors[1]
print(elnet_bic_all.rmse1)
print(elnet_bic_all.rmse3)
print(elnet_bic_all.rmse6)
print(elnet_bic_all.rmse12)







#1.2.2_elastic net_AIC
elasticnet1ca=lasso.rolling.window(all,nprev,1,1,alpha,IC="aic")
elasticnet3ca=lasso.rolling.window(all,nprev,1,3,alpha,IC="aic")
elasticnet6ca=lasso.rolling.window(all,nprev,1,6,alpha,IC="aic")
elasticnet12ca=lasso.rolling.window(all,nprev,1,12,alpha,IC="aic")
elneta_aic_all.rmse1=elasticnet1ca$errors[1]
elneta_aic_all.rmse3=elasticnet3ca$errors[1]
elneta_aic_all.rmse6=elasticnet6ca$errors[1]
elneta_aic_all.rmse12=elasticnet12ca$errors[1]
print(elneta_aic_all.rmse1)
print(elneta_aic_all.rmse3)
print(elneta_aic_all.rmse6)
print(elneta_aic_all.rmse12)






#1.2.3_elastic net_AICc
elasticnet1caic=lasso.rolling.window(all,nprev,1,1,alpha,IC="aicc")
elasticnet3caic=lasso.rolling.window(all,nprev,1,3,alpha,IC="aicc")
elasticnet6caic=lasso.rolling.window(all,nprev,1,6,alpha,IC="aicc")
elasticnet12caic=lasso.rolling.window(all,nprev,1,12,alpha,IC="aicc")
elnet_ac_all.rmse1=elasticnet1caic$errors[1]
elnet_ac_all.rmse3=elasticnet3caic$errors[1]
elnet_ac_all.rmse6=elasticnet6caic$errors[1]
elnet_ac_all.rmse12=elasticnet12caic$errors[1]
print(elnet_ac_all.rmse1)
print(elnet_ac_all.rmse3)
print(elnet_ac_all.rmse6)
print(elnet_ac_all.rmse12)





#Sparsity analysis over time plots for LASSO and ElNet
#Get nonzero coefficient numbers for different horizons (LASSO(BIC))
c1c=rowSums(lasso1c$coef != 0)
c3c=rowSums(lasso3c$coef != 0)
c6c=rowSums(lasso6c$coef != 0)
c12c=rowSums(lasso12c$coef != 0)

#Create a ts object for the plot
lcoef.ts=ts(cbind(c1c,c3c,c6c,c12c), start=c(2010,4), end=c(2021,12), freq=12)
colnames(lcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window
plot.ts(lcoef.ts, main="Sparsity Analysis for LASSO",cex.axis=1.5)

#Get nonzero coefficient numbers for different horizons (Elastic Net)
ce1c=rowSums(elasticnet1c$coef != 0)
ce3c=rowSums(elasticnet3c$coef != 0)
ce6c=rowSums(elasticnet6c$coef != 0)
ce12c=rowSums(elasticnet12c$coef != 0)

#Create a respective ts object for the plot
elcoef.ts=ts(cbind(ce1c,ce3c,ce6c,ce12c), start=c(2010,8), end=c(2021,12), freq=12)
colnames(elcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window:
plot.ts(elcoef.ts, main="Sparsity Analysis for ElNet",cex.axis=1.5)













#macro

#1.1.1_lasso_BIC
source("func-lasso.R")
alpha=1
lasso1c=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="bic")
lasso3c=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="bic")
lasso6c=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="bic")
lasso12c=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="bic")
lasso_bic_out.rmse1=lasso1c$errors[1]
lasso_bic_out.rmse3=lasso3c$errors[1]
lasso_bic_out.rmse6=lasso6c$errors[1]
lasso_bic_out.rmse12=lasso12c$errors[1]
print(lasso_bic_out.rmse1)
print(lasso_bic_out.rmse3)
print(lasso_bic_out.rmse6)
print(lasso_bic_out.rmse12)





#1.1.2_lasso_AIC
lasso1ca=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="aic")
lasso3ca=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="aic")
lasso6ca=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="aic")
lasso12ca=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="aic")
lasso_aic_out.rmse1=lasso1ca$errors[1]
lasso_aic_out.rmse3=lasso3ca$errors[1]
lasso_aic_out.rmse6=lasso6ca$errors[1]
lasso_aic_out.rmse12=lasso12ca$errors[1]
print(lasso_aic_out.rmse1)
print(lasso_aic_out.rmse3)
print(lasso_aic_out.rmse6)
print(lasso_aic_out.rmse12)





#1.1.3_lasso_AICc
lasso1caic=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="aicc")
lasso3caic=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="aicc")
lasso6caic=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="aicc")
lasso12caic=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="aicc")
lasso_ac_out.rmse1=lasso1caic$errors[1]
lasso_ac_out.rmse3=lasso3caic$errors[1]
lasso_ac_out.rmse6=lasso6caic$errors[1]
lasso_ac_out.rmse12=lasso12caic$errors[1]
print(lasso_ac_out.rmse1)
print(lasso_ac_out.rmse3)
print(lasso_ac_out.rmse6)
print(lasso_ac_out.rmse12)






#1.1.4_Post-LASSO_BIC
pols.lasso1c=pols.rolling.window(out_senti,nprev,1,1,lasso1c$coef)
pols.lasso3c=pols.rolling.window(out_senti,nprev,1,3,lasso3c$coef)
pols.lasso6c=pols.rolling.window(out_senti,nprev,1,6,lasso6c$coef)
pols.lasso12c=pols.rolling.window(out_senti,nprev,1,12,lasso12c$coef)
plasso_out.rmse1=pols.lasso1c$errors[1]
plasso_out.rmse3=pols.lasso3c$errors[1]
plasso_out.rmse6=pols.lasso6c$errors[1]
plasso_out.rmse12=pols.lasso12c$errors[1]
print(plasso_out.rmse1)
print(plasso_out.rmse3)
print(plasso_out.rmse6)
print(plasso_out.rmse12)






#1.2.1_elastic net_BIC
alpha=0.5
elasticnet1c=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="bic")
elasticnet3c=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="bic")
elasticnet6c=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="bic")
elasticnet12c=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="bic")
elnet_bic_out.rmse1=elasticnet1c$errors[1]
elnet_bic_out.rmse3=elasticnet3c$errors[1]
elnet_bic_out.rmse6=elasticnet6c$errors[1]
elnet_bic_out.rmse12=elasticnet12c$errors[1]
print(elnet_bic_out.rmse1)
print(elnet_bic_out.rmse3)
print(elnet_bic_out.rmse6)
print(elnet_bic_out.rmse12)






#1.2.2_elastic net_AIC
elasticnet1ca=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="aic")
elasticnet3ca=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="aic")
elasticnet6ca=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="aic")
elasticnet12ca=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="aic")
elnet_aic_out.rmse1=elasticnet1ca$errors[1]
elnet_aic_out.rmse3=elasticnet3ca$errors[1]
elnet_aic_out.rmse6=elasticnet6ca$errors[1]
elnet_aic_out.rmse12=elasticnet12ca$errors[1]
print(elnet_aic_out.rmse1)
print(elnet_aic_out.rmse3)
print(elnet_aic_out.rmse6)
print(elnet_aic_out.rmse12)





#1.2.3_elastic net_AICc
elasticnet1caic=lasso.rolling.window(out_senti,nprev,1,1,alpha,IC="aicc")
elasticnet3caic=lasso.rolling.window(out_senti,nprev,1,3,alpha,IC="aicc")
elasticnet6caic=lasso.rolling.window(out_senti,nprev,1,6,alpha,IC="aicc")
elasticnet12caic=lasso.rolling.window(out_senti,nprev,1,12,alpha,IC="aicc")
elnet_ac_out.rmse1=elasticnet1caic$errors[1]
elnet_ac_out.rmse3=elasticnet3caic$errors[1]
elnet_ac_out.rmse6=elasticnet6caic$errors[1]
elnet_ac_out.rmse12=elasticnet12caic$errors[1]
print(elnet_ac_out.rmse1)
print(elnet_ac_out.rmse3)
print(elnet_ac_out.rmse6)
print(elnet_ac_out.rmse12)



#Sparsity analysis over time plots for LASSO and ElNet
#Get nonzero coefficient numbers for different horizons (LASSO(BIC))
c1c=rowSums(lasso1c$coef != 0)
c3c=rowSums(lasso3c$coef != 0)
c6c=rowSums(lasso6c$coef != 0)
c12c=rowSums(lasso12c$coef != 0)

#Create a ts object for the plot
lcoef.ts=ts(cbind(c1c,c3c,c6c,c12c), start=c(2010,4), end=c(2021,12), freq=12)
colnames(lcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window
plot.ts(lcoef.ts, main="Sparsity Analysis for LASSO",cex.axis=1.5)

#Get nonzero coefficient numbers for different horizons (Elastic Net)
ce1c=rowSums(elasticnet1c$coef != 0)
ce3c=rowSums(elasticnet3c$coef != 0)
ce6c=rowSums(elasticnet6c$coef != 0)
ce12c=rowSums(elasticnet12c$coef != 0)

#Create a respective ts object for the plot
elcoef.ts=ts(cbind(ce1c,ce3c,ce6c,ce12c), start=c(2010,8), end=c(2021,12), freq=12)
colnames(elcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window:
plot.ts(elcoef.ts, main="Sparsity Analysis for ElNet",cex.axis=1.5)

















#sentiments

#1.1.1_lasso_BIC
alpha=1
lasso1c=lasso.rolling.window(senti,nprev,1,1,alpha,IC="bic")
lasso3c=lasso.rolling.window(senti,nprev,1,3,alpha,IC="bic")
lasso6c=lasso.rolling.window(senti,nprev,1,6,alpha,IC="bic")
lasso12c=lasso.rolling.window(senti,nprev,1,12,alpha,IC="bic")
lasso_bic_senti.rmse1=lasso1c$errors[1]
lasso_bic_senti.rmse3=lasso3c$errors[1]
lasso_bic_senti.rmse6=lasso6c$errors[1]
lasso_bic_senti.rmse12=lasso12c$errors[1]
print(lasso_bic_senti.rmse1)
print(lasso_bic_senti.rmse3)
print(lasso_bic_senti.rmse6)
print(lasso_bic_senti.rmse12)




#1.1.2_lasso_AIC
lasso1ca=lasso.rolling.window(senti,nprev,1,1,alpha,IC="aic")
lasso3ca=lasso.rolling.window(senti,nprev,1,3,alpha,IC="aic")
lasso6ca=lasso.rolling.window(senti,nprev,1,6,alpha,IC="aic")
lasso12ca=lasso.rolling.window(senti,nprev,1,12,alpha,IC="aic")
lasso_aic_senti.rmse1=lasso1ca$errors[1]
lasso_aic_senti.rmse3=lasso3ca$errors[1]
lasso_aic_senti.rmse6=lasso6ca$errors[1]
lasso_aic_senti.rmse12=lasso12ca$errors[1]
print(lasso_aic_senti.rmse1)
print(lasso_aic_senti.rmse3)
print(lasso_aic_senti.rmse6)
print(lasso_aic_senti.rmse12)




#1.1.3_lasso_AICc
lasso1caic=lasso.rolling.window(senti,nprev,1,1,alpha,IC="aicc")
lasso3caic=lasso.rolling.window(senti,nprev,1,3,alpha,IC="aicc")
lasso6caic=lasso.rolling.window(senti,nprev,1,6,alpha,IC="aicc")
lasso12caic=lasso.rolling.window(senti,nprev,1,12,alpha,IC="aicc")
lasso_ac_senti.rmse1=lasso1caic$errors[1]
lasso_ac_senti.rmse3=lasso3caic$errors[1]
lasso_ac_senti.rmse6=lasso6caic$errors[1]
lasso_ac_senti.rmse12=lasso12caic$errors[1]
print(lasso_ac_senti.rmse1)
print(lasso_ac_senti.rmse3)
print(lasso_ac_senti.rmse6)
print(lasso_ac_senti.rmse12)






#1.1.4_Post-LASSO_BIC
pols.lasso1c=pols.rolling.window(senti,nprev,1,1,lasso1c$coef)
pols.lasso3c=pols.rolling.window(senti,nprev,1,3,lasso3c$coef)
pols.lasso6c=pols.rolling.window(senti,nprev,1,6,lasso6c$coef)
pols.lasso12c=pols.rolling.window(senti,nprev,1,12,lasso12c$coef)
plasso_senti.rmse1=pols.lasso1c$errors[1]
plasso_senti.rmse3=pols.lasso3c$errors[1]
plasso_senti.rmse6=pols.lasso6c$errors[1]
plasso_senti.rmse12=pols.lasso12c$errors[1]
print(plasso_senti.rmse1)
print(plasso_senti.rmse3)
print(plasso_senti.rmse6)
print(plasso_senti.rmse12)







#1.2.1_elastic net_BIC
alpha=0.5
elasticnet1c=lasso.rolling.window(senti,nprev,1,1,alpha,IC="bic")
elasticnet3c=lasso.rolling.window(senti,nprev,1,3,alpha,IC="bic")
elasticnet6c=lasso.rolling.window(senti,nprev,1,6,alpha,IC="bic")
elasticnet12c=lasso.rolling.window(senti,nprev,1,12,alpha,IC="bic")
elnet_bic_senti.rmse1=elasticnet1c$errors[1]
elnet_bic_senti.rmse3=elasticnet3c$errors[1]
elnet_bic_senti.rmse6=elasticnet6c$errors[1]
elnet_bic_senti.rmse12=elasticnet12c$errors[1]
print(elnet_bic_senti.rmse1)
print(elnet_bic_senti.rmse3)
print(elnet_bic_senti.rmse6)
print(elnet_bic_senti.rmse12)







#1.2.2_elastic net_AIC
elasticnet1ca=lasso.rolling.window(senti,nprev,1,1,alpha,IC="aic")
elasticnet3ca=lasso.rolling.window(senti,nprev,1,3,alpha,IC="aic")
elasticnet6ca=lasso.rolling.window(senti,nprev,1,6,alpha,IC="aic")
elasticnet12ca=lasso.rolling.window(senti,nprev,1,12,alpha,IC="aic")
elnet_aic_senti.rmse1=elasticnet1ca$errors[1]
elnet_aic_senti.rmse3=elasticnet3ca$errors[1]
elnet_aic_senti.rmse6=elasticnet6ca$errors[1]
elnet_aic_senti.rmse12=elasticnet12ca$errors[1]
print(elnet_aic_senti.rmse1)
print(elnet_aic_senti.rmse3)
print(elnet_aic_senti.rmse6)
print(elnet_aic_senti.rmse12)







#1.2.3_elastic net_AICc
elasticnet1caic=lasso.rolling.window(senti,nprev,1,1,alpha,IC="aicc")
elasticnet3caic=lasso.rolling.window(senti,nprev,1,3,alpha,IC="aicc")
elasticnet6caic=lasso.rolling.window(senti,nprev,1,6,alpha,IC="aicc")
elasticnet12caic=lasso.rolling.window(senti,nprev,1,12,alpha,IC="aicc")
elnet_ac_senti.rmse1=elasticnet1caic$errors[1]
elnet_ac_senti.rmse3=elasticnet3caic$errors[1]
elnet_ac_senti.rmse6=elasticnet6caic$errors[1]
elnet_ac_senti.rmse12=elasticnet12caic$errors[1]
print(elnet_ac_senti.rmse1)
print(elnet_ac_senti.rmse3)
print(elnet_ac_senti.rmse6)
print(elnet_ac_senti.rmse12)




#Sparsity analysis over time plots for LASSO and ElNet
#Get nonzero coefficient numbers for different horizons (LASSO(BIC))
c1c=rowSums(lasso1c$coef != 0)
c3c=rowSums(lasso3c$coef != 0)
c6c=rowSums(lasso6c$coef != 0)
c12c=rowSums(lasso12c$coef != 0)

#Create a ts object for the plot
lcoef.ts=ts(cbind(c1c,c3c,c6c,c12c), start=c(2010,4), end=c(2021,12), freq=12)
colnames(lcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window
plot.ts(lcoef.ts, main="Sparsity Analysis for LASSO",cex.axis=1.5)

#Get nonzero coefficient numbers for different horizons (Elastic Net)
ce1c=rowSums(elasticnet1c$coef != 0)
ce3c=rowSums(elasticnet3c$coef != 0)
ce6c=rowSums(elasticnet6c$coef != 0)
ce12c=rowSums(elasticnet12c$coef != 0)

#Create a respective ts object for the plot
elcoef.ts=ts(cbind(ce1c,ce3c,ce6c,ce12c), start=c(2010,8), end=c(2021,12), freq=12)
colnames(elcoef.ts)=c("1-step","3-step","6-step","12-step")
#Plot numbers of nonzero coefficients across the test window:
plot.ts(elcoef.ts, main="Sparsity Analysis for ElNet",cex.axis=1.5)








