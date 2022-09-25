library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)

# dtree for classification

#importing train and test datasets
rg_train=read.csv("D:\\IITK Data Analytics\\R\\DecisionTrees-RandomForests-and-BoostingMachines\\rg_train.csv",stringsAsFactors = FALSE)
rg_test=read.csv("D:\\IITK Data Analytics\\R\\DecisionTrees-RandomForests-and-BoostingMachines\\rg_test.csv",stringsAsFactors = FALSE)

age_band_func=function(x){
  a1=as.numeric(substr(x,1,2))
  a2=as.numeric(substr(x,4,5))
  age=ifelse(substr(x,1,2)=="71",71, #for 71+
             ifelse(x=="Unknown",NA,0.5*(a1+a2)))
  return(age)
  
}

children_func=function(x){
  x=ifelse(x=="Zero",0,substr(x,1,1)) #start and stop index
  x=as.numeric(x)
  
  return(x)
}


fi_func=function(x){
  
  x=gsub("[<,>=]","",x)
  
  temp=data.frame(fi=x)
  
  temp=temp %>% 
    separate(fi,into=c("f1","f2")) %>% 
    mutate(f1=as.numeric(f1),
           f2=as.numeric(f2),
           fi=ifelse(is.na(f1),f2,
                     ifelse(is.na(f2),f1,0.5*(f1+f2)))) %>% 
    select(-f1,-f2)
  
  return(temp$fi)
}

#target column to be treated seperately. Good practice
rg_train$Revenue.Grid=as.factor(as.numeric(rg_train$Revenue.Grid==1))

dp_pipe=recipe(Revenue.Grid~.,data=rg_train) %>% 
  
  update_role(REF_NO,post_code,post_area,new_role = "drop_vars") %>%
  update_role(status,occupation,occupation_partner,home_status,
              self_employed,self_employed_partner,TVarea,
              gender,region,new_role="to_dummies") %>% 
  
  step_rm(has_role("drop_vars")) %>% 
  
  step_mutate_at(age_band,fn=age_band_func) %>% 
  step_mutate_at(family_income,fn=fi_func) %>% 
  step_mutate_at(children,fn=children_func) %>% 
  
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=rg_test)

## dtree paramters tuning for classification

# buildind the model but setting the hyperpaprameters for tuning . we will pass values later
tree_model=decision_tree( 
  cost_complexity = tune(), 
  tree_depth = tune(),
  min_n = tune() #min number of obs in a node
) %>%
  set_engine("rpart") %>% #package name rpart , show_engines("decision_tree")
  set_mode("classification") #regression/classification


folds = vfold_cv(train, v = 5)


tree_grid = grid_regular(cost_complexity(), tree_depth(),   # run each ot these indvidually to get idea of range
                         min_n(), levels = 7) #select 3 best values for each of these

# below code runs your code on parallel cores of your cpu
# makes the process faster
# doParallel::registerDoParallel() 


my_res=tune_grid(
  tree_model,
  Revenue.Grid~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc), #rmse ,mae for regression
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light() #roc higher the better

fold_metrics=collect_metrics(my_res) 

my_res %>% show_best()

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(Revenue.Grid~.,data=train)

# feature importance
final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# plot the tree

rpart.plot(final_tree_fit$fit)

# predictions

train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$Revenue.Grid

# KS plot

rocit = ROCit::rocit(score = train.score, 
              class = real) 

kplot=ROCit::ksplot(rocit,legend=F)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)

## Random Forest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

  rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 3)


my_res=tune_grid(
  rf_model,
  Revenue.Grid~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(Revenue.Grid~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$Revenue.Grid

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)

## partial dependence plots
model_explainer =explain_tidymodels(
  final_rf_fit,
  data = dplyr::select(train, -Revenue.Grid),
  y = as.integer(train$Revenue.Grid),
  verbose = FALSE
)

pdp = model_profile(
  model_explainer,
  variables = "family_income",
  N = 2000,
  groups='children'
)

plot(pdp)



