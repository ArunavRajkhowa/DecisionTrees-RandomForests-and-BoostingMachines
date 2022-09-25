

## ------------------------------------------------------------------------
bs=read.csv("/Users/lalitsachan/Dropbox/0.0 Data/bike_sharing_hours.csv",stringsAsFactors = F)

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

## -----------------------------------------------------
library(dplyr)
bs=bs %>% select(-dteday,-casual,-registered)

## ------------------------------------------------------------------------
char_cols=c("season","mnth","hr","holiday","weekday","workingday","weathersit")
for(col in char_cols){
  bs=CreateDummies(bs,col,500)
}

## -----------------------------------------------------
library(gbm)
library(cvTools)

param=list(interaction.depth=c(1:7),
           n.trees=c(50,100,200,500,700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

## ------------------------------------------------------------------------
subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=10
my_params=subset_paras(param,num_trials)
# Note: A good value for num_trials is around 10-20% of total possible 
# combination. It doesnt have to be always 10

## ---- this code might take too long to run--------------------------------------------------------------
myerror=9999999

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]

  k=cvTuning(gbm,cnt~.,
             data =bs,
             tuning =params,
             args = list(distribution="gaussian"),
             folds = cvFolds(nrow(bs), K=10, type = "random"),
             seed =2,
             predictArgs = list(n.trees=params$n.trees)
             )
  score.this=k$cv[,2]

  if(score.this<myerror){
    print(params)
    # uncomment the line above to keep track of progress
    myerror=score.this
    print(myerror)
    # uncomment the line above to keep track of progress
    best_params=params
  }

  print('DONE')
  # uncomment the line above to keep track of progress
}

## ----these are the values from a previous run--------------------------------------------------------------
myerror=52.29379
best_params=data.frame(interaction.depth=6,
                       n.trees=500,
                       shrinkage=0.1,
                       n.minobsnode=10)

## ------------------------------------------------------------------------
myerror

## ------------------------------------------------------------------------
best_params

## ------------------------------------------------------------------------
bs.gbm.final=gbm(cnt~.,data=bs,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "gaussian")

## ----for prediction and submission on test data--------------------------------------------------------------
## test.pred=predict(bs.gbm.final,newdata=bs_test,n.trees = best_params$n.trees)
## write.csv(test.pred,"mysubmission.csv",row.names = F)

## ----message=F-----------------------------------------------------------
ci_train=read.csv("~/Dropbox/0.0 Data/census_income.csv",
                  stringsAsFactors =F)
glimpse(ci_train)

ci_train=ci_train %>% select(-education)
ci_train$Y=as.numeric(ci_train$Y==" >50K")

cat_var=names(ci_train)[sapply(ci_train,is.character)]


for(var in cat_var){
  ci_train=CreateDummies(ci_train,var,500)
}

## For classification no need tk convert to factor type for gbm
## we'll just change distribution to "bernoulli"



## ------------------------------------------------------------------------
param=list(interaction.depth=c(1:7),
           n.trees=c(50,100,200,500,700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

num_trials=10
my_params=subset_paras(param,num_trials)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}
# Note: A good value for num_trials is around 10-20% of total possible 
# combination. It doesnt have to be always 10

## ----this code will take too long to run--------------------------------------------------------------
myauc=0

## Cvtuning
## This code will take couple hours to finish
## Dont execute in the class
for(i in 1:num_trials){
  # print(paste('starting iteration :',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]

  k=cvTuning(gbm,Y~.,
             data =ci_train,
             tuning =params,
             args=list(distribution="bernoulli"),
             folds = cvFolds(nrow(ci_train), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="response",n.trees=params$n.trees)
             )
  score.this=k$cv[,2]

  if(score.this>myauc){
    # print(params)
    # uncomment the line above to keep track of progress
    myauc=score.this
    # print(myauc)
    # uncomment the line above to keep track of progress
    best_params=params
  }

  # print('DONE')
  # uncomment the line above to keep track of progress
}

## ----these values are from a previous run--------------------------------------------------------------
myauc=0.9217784
best_params=data.frame(interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsinnode=5)

## ------------------------------------------------------------------------
myauc

## ------------------------------------------------------------------------
best_params

## ------------------------------------------------------------------------
ci.gbm.final=gbm(Y~.,data=ci_train,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsinnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "bernoulli")

## ----use these for prediciton and submission on test data--------------------------------------------------------------
## test.score=predict(ci.gbm.final,newdata=ci_test,type='response',
##                    n.trees = best_params$n.trees)
## write.csv(test.score,"mysubmission.csv",row.names=F)

## ------------------------------------------------------------------------
mykfolds=function(nobs,nfold=5){
  
  t=cvFolds(nobs,K=nfold,type='random')
  
  folds=list()
  
  for(i in 1:nfold){
    
    test=t$subsets[t$which==i]
    train=t$subsets[t$which!=i]
    
    folds[[i]]=list('train'=train,'test'=test)
  }
  
  return(folds)
}

## ------------------------------------------------------------------------
rg_train=read.csv("~/Dropbox/0.0 Data/rg_train.csv",stringsAsFactors = FALSE)
rg_test=read.csv("~/Dropbox/0.0 Data/rg_test.csv",stringsAsFactors = FALSE)

rg_test$Revenue.Grid=NA

rg_train$data='train'
rg_test$data='test'

rg=rbind(rg_train,rg_test)

rg = rg %>%
  mutate(children=ifelse(children=="Zero",0,substr(children,1,1)),
         children=as.numeric(children))

rg=rg %>%
mutate(a1=as.numeric(substr(age_band,1,2)),
       a2=as.numeric(substr(age_band,4,5)),
      age=ifelse(substr(age_band,1,2)=="71",71,ifelse(age_band=="Unknown",NA,0.5*(a1+a2)))
       ) %>%
  select(-a1,-a2,-age_band)

cat_cols=c("status","occupation","occupation_partner","home_status",
           "family_income","self_employed",
           "self_employed_partner","TVarea","gender","region")

for(cat in cat_cols){
  rg=CreateDummies(rg,cat,50)
}

rg=rg %>%
  select(-post_code,-post_area)

rg$Revenue.Grid=as.numeric(rg$Revenue.Grid==1)

for(col in names(rg)){
  
  if(sum(is.na(rg[,col]))>0 & !(col %in% c("data","Revenue.Grid"))){
    
    rg[is.na(rg[,col]),col]=mean(rg[rg$data=='train',col],na.rm=T)
  }
  
}

rg$Revenue.Grid=as.numeric(rg$Revenue.Grid==1)

rg_train=rg %>% filter(data=='train') %>% select(-data)
rg_test=rg %>% filter(data=='test') %>% select (-data,-Revenue.Grid)

## ------------------------------------------------------------------------
myfolds=mykfolds(nrow(rg_train),10)

## ------------------------------------------------------------------------
rg_train_layer=data.frame(rf_var=numeric(nrow(rg_train)),
                           gbm_var=numeric(nrow(rg_train)))

## -------------this code will take long to run----------------------------------------------
library(randomForest)
for(i in 1:10){
  print(c(i))
  fold=myfolds[[i]]
  
  train_data=rg_train[fold$train,]
  # for this iteration model will be built on this chunk of the data
  test_data=rg_train[fold$test,]
  # predicitons will be made on this chunk which is not being
  # used in the modeling process
  
  print('rf')
  
  rf.fit=randomForest(factor(Revenue.Grid)~.-REF_NO,
                         mtry=10,
                         ntree=500,
                         maxnodes=100,
                         nodesize=10,
                         data=train_data
                         )
  ## these value of parameters have been chosen randomly here
  ## but separately tuned parameter choices will be better
  rf_score=predict(rf.fit,newdata=test_data,type='prob')[,1]
  
  print('gbm')
  gbm.fit=gbm(Revenue.Grid~.-REF_NO,data=train_data,
                 interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsinnode=5,
                 distribution = "bernoulli")
   ## these value of parameters have been chosen randomly here
  ## but separately tuned parameter choices will be better
  gbm_score=predict(gbm.fit,newdata=test_data,
                    n.trees=700,type='response')
  
  rg_train_layer$rf_var[fold$test]=rf_score
  
  rg_train_layer$gbm_var[fold$test]=gbm_score
}

## ------------------------------------------------------------------------
rg_test_layer=data.frame(rf_var=numeric(nrow(rg_test)),
                          gbm_var=numeric(nrow(rg_test)))

full.rf=randomForest(factor(Revenue.Grid)~.-REF_NO,
                         mtry=10,
                         ntree=500,
                         maxnodes=100,
                         nodesize=10,
                         data=rg_train
                         )
full.gbm=gbm(Revenue.Grid~.-REF_NO,data=rg_train,
                 interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsinnode=5,
                 distribution = "bernoulli")
# note that paramater choices are exactly same as earlier

rg_test_layer$rf_var=predict(full.rf,newdata=rg_test,type='prob')[,1]
rg_test_layer$gbm_var=predict(full.gbm,newdata=rg_test,
                               n.trees=700,type='response')

## ------------------------------------------------------------------------
rg_train_layer$Revenue.Grid=rg_train$Revenue.Grid

log.mod=glm(Revenue.Grid~.,data=rg_train_layer,family = "binomial")


## ----use this for prediction and submission on test--------------------------------------------------------------
## test.score=predict(log.mod,newdata=rg_test_layer,type='response')
## write.csv(test.score,"mysubmission.csv",row.names = F)

