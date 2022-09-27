library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)

file=read.csv("D:\\IITK Data Analytics\\R\\DecisionTrees-RandomForests-and-BoostingMachines\\paydayloan_collections.csv",stringsAsFactors = FALSE)
vis_dat(file,warn_large_data=F)
glimpse(file)

## Data Preparation--------

#target column to be treated seperately. Good practice
file$payment=as.factor(as.numeric(file$payment=='Success'))

dp_pipe=recipe(payment~.,data=file) %>% 
  
  #update_role(REF_NO,post_code,post_area,new_role = "drop_vars") %>%
  update_role(var1,var2,var9,var10,var11,var13,var17,var19,var23,
              var29,new_role="to_dummies") %>% 
  
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

file=bake(dp_pipe,new_data=NULL)
vis_dat(file,warn_large_data=F)

#splitting into train and test set
set.seed(2)
s=sample(1:nrow(file),0.7*nrow(file))
train=file[s,]
test=file[-s,]


#Decision Tree Model
tree_model=decision_tree( 
  cost_complexity = tune(), 
  tree_depth = tune(),
  min_n = tune() #min number of obs in a node
) %>%
  set_engine("rpart") %>% #package name rpart , show_engines("decision_tree")
  set_mode("classification") #regression/classification


folds = vfold_cv(train, v = 5)


tree_grid = grid_regular(cost_complexity(), tree_depth(),   # run each ot these indvidually to get idea of range
                         min_n(), levels = 3) #select 3 best values for each of these
my_res=tune_grid(
  tree_model,
  payment~.,
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
  fit(payment~.,data=train)


# predictions

train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)
table(test_pred,test$payment)
### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$payment

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit,legend=T)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)
table(test_hard_class,test$payment)







# Random Forest Model
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
  payment~.,
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
  fit(payment~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$payment

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)
table(test_hard_class,test$payment)
