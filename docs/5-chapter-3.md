---
title: 'Chapter 3: Predicting diseases application'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---
# Chapter 3: Predicting diseases application
We are finally ready to carry out the final objective: using penalised regression to predict diseases. 

## Load data
The first task is to load in the data we prepared in Chapter 1.
```{r}
setwd("data")
data = readRDS("colitis-data-c3.RDS")
X = data$X
y = data$y
groups = data$groups
rm(data)
```

We want to test the models out by fitting on training data and evaluating them on test data. Therefore, we will split the data randomly into train/test datasets. The split proportion between the two datasets is a subjective choice, but we want there to be enough data to fit an informative model, but also have enough test data to conduct an extensive validation. Commonly used proportions of train/test are anything from 50/50 to 80/20. In this case, we use approximately 60/40.

```{r}
set.seed(100)
training_ind <- sample(1:nrow(X), 50) 
train_data <- list(X = X[training_ind,], y = y[training_ind])
test_data <- list(X = X[-training_ind,], y = y[-training_ind])
```

## Initial hyperparameters
To fit the various models, we need to define the hyperparameters so that we are consistent across the models, allowing for fairer comparison. The parameters that we need to decide upon are (some of these were explained in Chapter 2):

* `path_length`: this defines how many values of $\lambda$ we will fit the model for. We will set this to $100$ to allow for more models to be fit. Ideally we want this to be as large as possible, to give as many possible models, but we must also think about computational cost. 
* `min_frac`: this sets the value that the final $\lambda$ value is set to, in the sense that $\lambda_{100} = \lambda_{1}\times \text{min frac}$. Making this small means that we will allow denser models to be fit. We set this to $0.01$ to allow denser models to be considered.
* `alpha`: this is only used for SGL (and SGS in the optional questions): this defines the balance between the variable and group penalties. In genetics, we usually encounter large groups, with many noisy variables. As such, we would prefer to not be limited by the group penalties, in which full groups are picked, but we still want to utilize grouping information. As such, we will use $0.99$ (which is a slight deviation from the recommended $0.95$ value discussed in Chapter 2, showing that sometimes a problem requires deviation from the recommended usage of a model).
* `num_iter`: this is the maximum number of iterations the fitting algorithm should fit for, if convergence is not reached. This tends to be set at $5000$, but as the dataset is quite large, we allow the fitting algorithms to run for longer.

We will also use $\ell_2$ standardisation for each model and fit an intercept.

```{r}
path_length = 100
min_frac = 0.01
alpha = 0.99
num_iter = 10000
```

## Fit lasso model
Now that the data is loaded, we can fit a lasso model. Note that this time we have set `family="binomial"` as our response $y$ is binary. 
```{r}
library(glmnet)
lasso_model <- glmnet(
  x = train_data$X,
  y = train_data$y,
  family = "binomial",
  lambda.min.ratio = min_frac,
  maxit = num_iter,
  standardize = TRUE,
  intercept = TRUE
) 
```
We can investigate the fitted values
```{r}
plot(lasso_model)
```
and also see how many variables are entering the model as we decrease $\lambda$:
```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
```
At the most saturated point, we have about $30$ genes in the model.

### Prediction
Before we move onto the next models, let's see how well the lasso performs in prediction. 

First, we use `predict` function on the lasso model.
```{r}
# calculate predictions
lasso_preds <- predict(lasso_model, test_data$X, type = "class")
lasso_preds[1:5,1:5] # a snapshot
```

Next, we need to compare the predictions to the test data and output this into a new dataframe. To do this, we are checking for each $\lambda$ index (each column of the prediction matrix) how many times on average the prediction matches the test data (which tells us the classification accuracy).

```{r}
# compare to test data
lasso_cr <- apply(lasso_preds, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
lasso_df = data.frame(
  model = "Lasso",
  lambda_index = 1:path_length,
  classification_rate = lasso_cr
)
```
We can now view the classification rate as a function of the $\lambda$ index.
```{r}
plot(x = lasso_df$lambda_index, y = lasso_df$classification_rate, type="l", xlab="Lambda index", ylab = "Classification accuracy")
abline(v = which.max(lasso_df$classification_rate), col = "red") # where the maximum is located
```
```{r}
max(lasso_df$classification_rate)
which.max(lasso_df$classification_rate)
apply(lasso_model$beta, 2, function(x) length(which(x!=0)))[which.max(lasso_df$classification_rate)]
```
So the best model appears to be the one at the $\lambda$ index of $29$, achieving a peak classification score of $93.5\%$. Looking back again at the plot showing the number of genes, we see that the best model is not the one using the most amount of genes, but in fact needs only $11$ genes

```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
abline(v = which.max(lasso_df$classification_rate), col = "red")
```

To gain further insight into the best lasso predictive model, we can plot the decision boundaries. Decision boundaries in classification problems show how close each data point was to being classified differently (we are plotting the class probabilities).

```{r}
false_pred = which(!ifelse(lasso_class_prob>=0.5,1,0) == test_data$y)
lasso_class_prob <- predict(lasso_model, test_data$X, type = "response")[,which.max(lasso_df$classification_rate)]
plot(lasso_class_prob,pch=16, ylab = "Class probability",ylim=c(0,1))
abline(h=0.5)
points(test_data$y,col=ifelse(test_data$y=="TRUE","forestgreen","red"),pch=4)
points(x = false_pred, y = lasso_class_prob[false_pred], col = "orange", pch = 16)
```
The coloured crosses represent the true class of the observations. The lasso appears fairly confident on most observations, not choosing too many probabilities close to the decision boundary of $0.5$. The orange points highlight the incorrectly identified observations. The cluster of three on the right are close to the boundary, but the other two on the left are not.

**Q1: what happens when you don't fit an intercept? What about no standardisation?**

**Q2: apply the lasso to the cancer data**

**Q3 (optional): `glmnet` has the elastic net model. Apply it to the colitis data.**

## Group lasso (gLasso)
We move on to fitting the group lasso model. As mentioned in Chapter 2, the `grplasso` package does not perform standardisation properly, so we do this ourselves before fitting. 

```{r}
library(grplasso)
X_gl <- t(t(train_data$X) - apply(train_data$X, 2, mean))
X_gl <- t(t(X_gl) / apply(X_gl, 2, sd))
X_gl <- cbind(1, X_gl)
groups_gl <- c(NA, groups)

lambda_max_group <- lambdamax(X_gl, as.numeric(train_data$y), groups_gl, standardize = FALSE)
lambdas_gl <- exp(seq(
  from = log(lambda_max_group),
  to = log(lambda_max_group * min_frac),
  length.out = path_length
))
glasso_model <- grplasso(
  x = X_gl,
  y = as.numeric(train_data$y),
  index = groups_gl,
  lambda = lambdas_gl,
  standardize = FALSE,
  max.iter = num_iter
)
```

As before, we investigate the solution
```{r}
plot(glasso_model)
```
and also see how many variables are entering the model as we decrease $\lambda$:
```{r}
plot(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
```
At the most saturated point, we have $24$ genes in the model.

### Prediction
We extract the predictions for gLasso
```{r}
glasso_preds <- predict(object = glasso_model, newdata = cbind(1,test_data$X), type = "response")
glasso_preds[1:5,1:5]
```
However, we notice that unlike the `glmnet` function, the predictions are not given in their final binary format, but are instead probabilities, so we need to convert these. This is another example of a package not having all the features that `glmnet` has.
```{r}
glasso_preds = ifelse(glasso_preds >= 0.5, 1, 0)
glasso_cr <- apply(glasso_preds, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
glasso_df = data.frame(
  model = "gLasso",
  lambda_index = 1:path_length,
  classification_rate = glasso_cr
)
```
We now perform the same visualisations as for the lasso.
```{r}
plot(x = glasso_df$lambda_index, y = glasso_df$classification_rate, type="l", xlab="Lambda index", ylab = "Classification accuracy")
abline(v = which.max(glasso_df$classification_rate), col = "red") # where the maximum is located
```
```{r}
max(glasso_df$classification_rate)
length(which(glasso_model$coefficients[,which.max(glasso_df$classification_rate)]!=0))
```
```{r}
plot(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
abline(v = which.max(glasso_df$classification_rate), col = "red")
```
The best model appears to be the one at the $\lambda$ index of $95$, achieving a peak classification score of $89.6\%$ using $23$ genes. This is another example highlighting the downside of applying only groupwise sparsity. By being forced to pick all variables in a group as active, we are using a lot of noise variables to form our prediction, leading to a decrease in the classification accuracy of $4\%$ in comparison to the lasso. The added complexity of applying a group penalty does not yield any benefit over the simpler lasso. We now turn to SGL to see if this can resolve some of these issues.

**Q4: apply the group lasso to the colitis data.**

## Sparse-group lasso (SGL)
The `SGL` package crashes R when applied to the colitis dataset. This appears to be a bug in the package. Instead, we will use the `sgs` package, which fits SGS models. SGS models can be reduced to SGL models by using constant weights (as indicated by the choice of `v_weights` and `w_weights` below).
```{r}
library(sgs)
sgl_model = fit_sgs(
  X = train_data$X,
  y = train_data$y,
  groups = groups,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  alpha = alpha,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  v_weights = rep(1, ncol(train_data$X)),
  w_weights = rep(1, length(unique(groups)))
)
```
Performing the predictiton:
```{r}
sgl_preds = predict(sgl_model, x = test_data$X)
sgl_cr <- apply(sgl_preds$class, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
sgl_df = data.frame(
  model = "SGL",
  lambda_index = 1:path_length,
  classification_rate = sgl_cr
)
max(sgl_df$classification_rate)
which.max(sgl_df$classification_rate)
```
SGL obtains a peak accuracy of $95\%$ at the index of $80$, using $25$ genes:
```{r}
length(sgl_model$selected_var[[which.max(sgl_df$classification_rate)]])
```
We observe that SGL outperforms the lasso and group lasso, highlighting the benefit of applying sparse-group penalisation. The group lasso is limited as it is forced to make every gene within an active group active, leading to a lot of noise being included in the model. On the other hand, the lasso does not utilise grouping information. SGL is able to overcome both of these issues.
**Q5: apply SGL to the colitis data**

## Comparison of models
### Number of non-zero coefficients
```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero", ylim = c(0,40))
lines(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", col = "red")
lines(unlist(lapply(sgl_model$selected_var,length)), type = "l", col = "brown")
```

### Prediction accuracies
```{r}
plot(lasso_df$classification_rate, type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(glasso_df$classification_rate, type="l", col = "red")
lines(sgl_df$classification_rate, type = "l", col = "brown")
```

| Model    | Classification accuracy (%) | Genes used |
|----------|-----------------------------|------------|
| Lasso    | 93.5                        | 11         |
| gLasso   | 89.6                        | 23         |
| SGL      | 94.8                        | 25         |

## SLOPE models (optional)
We covered SLOPE models in the optional section of Chapter 2. Here, we apply them to the genetics dataset to get some insight into whether the adaptive and sorting components lead to an improvement over the lasso. We apply three different SLOPE-based models, which will allow for comparison to their lasso equivalence.

### SLOPE
The SLOPE package has a lot of options that we can configure. They have been configured here to allow for direct comparison to the other models, ensuring that we also use $100$ different $\lambda$ values. The key quantity that we have had to define compared to the previous models is the `q` parameter, which corresponds to the desired FDR level. In this case, due to the large size of the dataset, we set this to be a very low value to ensure that a strong level of penalisation is applied. 

```{r}
library(SLOPE)

slope_model = SLOPE(
  x = train_data$X,
  y = train_data$y,
  q = 1e-4,
  family = "binomial",
  intercept = TRUE,
  scale = "l2",
  center = TRUE,
  alpha = "path",
  lambda = "bh",
  alpha_min_ratio = min_frac,
  max_passes = 10000,
  path_length = path_length,
  tol_dev_ratio = 1000000,
  max_variables = 10000,
  tol_dev_change = 0
)
```
We now use the SLOPE model to form a prediction.
```{r}
slope_preds = predict(slope_model, x = test_data$X, type = "response") 
slope_preds = ifelse(slope_preds >= 0.5, 1, 0)
slope_cr <- apply(slope_preds, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
slope_df = data.frame(
  model = "SLOPE",
  lambda_index = 1:path_length,
  classification_rate = slope_cr
)
max(slope_df$classification_rate)
which.max(slope_df$classification_rate)
```
SLOPE is found to have a peak classification rate of $94.8%$ at index $30$. 

```{r}
max(slope_df$classification_rate)
which.max(slope_df$classification_rate)
sum(slope_model$nonzeros[,,which.max(slope_df$classification_rate)])
```
This model uses $16$ genes.

**Q6 (optional): apply SLOPE to the colitis data**

### Group SLOPE (gSLOPE)
To fit a gSLOPE model, in this seciton we will use the `sgs` package instead of the `grpSLOPE` package, as in Chapter 2. The `sgs` package has a function for fitting gSLOPE models which contains useful features that the `grpSLOPE` package does not have.

The parameters are as for the other models. The `lambda` sequence option is set to `mean`, which is a sequence derived for gSLOPE to control the false-discovery rate, with the corresponding parameter for this sequence set to `1e-4`. We turn the intercept off here, as the `sgs` package has a bug when using logistic models and intercepts, but given the high dimensionality, the intercept is not particularly important anyway. The key feature for the `sgs` package is that it performs screening, which is an approach that massively speeds up model fitting, particularly in settings where we have many features.

```{r}
library(sgs)
gslope_model = fit_gslope(
  X = train_data$X,
  y = train_data$y,
  groups = groups,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-4, 
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE, 
  verbose = TRUE
)
```
Performing the predictiton:
```{r}
gslope_preds = predict(gslope_model, x = test_data$X)
gslope_cr <- apply(gslope_preds$class, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
gslope_df = data.frame(
  model = "gSLOPE",
  lambda_index = 1:path_length,
  classification_rate = gslope_cr
)
max(gslope_df$classification_rate)
which.max(gslope_df$classification_rate)
```
gSLOPE obtains a peak accuracy of $81%$ at the index of 69.
```{r}
length(gslope_model$selected_var[[which.max(gslope_df$classification_rate)]])
```
And this model uses $43$ genes, which is over twice the amount that SLOPE uses. This is another illustration of the downside of applying just a group penalty. Not only is the prediction accuracy lower, but the model also uses more genes. 

**Q7 (optional): apply gSLOPE to the colitis data**

### Sparse-group SLOPE (SGS)
The final model we will test is sparse-group SLOPE (SGS), which applies adaptive penalisation at both the variable and group levels. In theory, this model should apply the strongest amount of penalisation, leading to the most sparse models. We can use the `sgs` package, as discussed in Chapter 2. SGS has three different choices of penalty sequences. Here, we have set `pen_method = 3`, as this is the fastest (computationally) sequence to calculate.

```{r}
library(sgs)
sgs_model = fit_sgs(
  X = train_data$X,
  y = train_data$y,
  groups = groups,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-4,
  vFDR = 1e-4, 
  alpha = alpha,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  pen_method = 3  
)
```
Performing the predictiton:
```{r}
sgs_preds = predict(sgs_model, x = test_data$X)
sgs_cr <- apply(sgs_preds$class, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
sgs_df = data.frame(
  model = "SGS",
  lambda_index = 1:path_length,
  classification_rate = sgs_cr
)
max(sgs_df$classification_rate)
which.max(sgs_df$classification_rate)
```
SGS obtains a peak accuracy of $95%$ at the index of $68$, using $23$ genes:
```{r}
length(sgs_model$selected_var[[which.max(sgs_df$classification_rate)]])
```

**Q8 (optional): can you achieve a higher predictive accuracy with SGS?**
**Q9 (optional): apply SGS to the colitis data**

### Number of non-zero coefficients
```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", col = "red")
lines(lapply(sgl_model$selected_var,length), type = "l", col = "brown")
lines(apply(slope_model$nonzeros,3,sum), type = "l", col = "blue")
lines(lapply(gslope_model$selected_var,length), type = "l", col = "green")
lines(lapply(sgs_model$selected_var,length), type = "l", col = "purple")
legend("topright", legend = c("Lasso", "gLasso", "SGL", "SLOPE", "gSLOPE", "SGS"),
       col = c("black", "red", "brown", "blue", "green", "purple"), lty = 1)
```

### Prediction accuracies
```{r}
plot(lasso_df$classification_rate, type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(glasso_df$classification_rate, type="l", col = "red")
lines(sgl_df$classification_rate, type = "l", col = "brown")
lines(slope_df$classification_rate, type = "l", col = "blue")
lines(gslope_df$classification_rate, type = "l", col = "green")
lines(sgs_df$classification_rate, type = "l", col = "purple")
legend("topright", legend = c("Lasso", "gLasso", "SGL", "SLOPE", "gSLOPE", "SGS"),
       col = c("black", "red", "brown", "blue", "green", "purple"), lty = 1)
```

| Model    | Classification accuracy (%) | Genes used |
|----------|-----------------------------|------------|
| Lasso    | 93.5                        | 11         |
| gLasso   | 89.6                        | 23         |
| SGL      | 94.8                        | 25         |
| SLOPE    | 94.8                        | 16         |
| gSLOPE   | 80.5                        | 43         | 
| SGS      | 94.8                        | 23         |