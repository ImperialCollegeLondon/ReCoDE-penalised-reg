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
We are finally ready to carry out the final objective: using penalised regression to predict diseases. 

## Load data
The first task is to load in the data we prepared in Chapter 1.
```{r}
data = readRDS("colitis-data-c3.RDS")
X = data$X
y = data$y
groups = data$groups
rm(data)
```

We want to test the models out by fitting on training data and evaluating them on a test data. Therefore, we will split the data randomly into train/test datasets.

```{r}
set.seed(100)
training_ind <- sample(1:nrow(X), 50) 
train_data <- list(X = X[training_ind,], y = y[training_ind])
test_data <- list(X = X[-training_ind,], y = y[-training_ind])
```

## Initial hyperparameters
To fit the various models, we need to define the hyperparameters so that we are consistent across the models, allowing for fairer comparison. The parameters that we need to decide upon are:

* `path_length`: this defines how many values of $\lambda$ we will fit the model for. We will set this to $100$ to allow for more models to be fit. Ideally we want this to be as large as possible, to give as many possible models, but we must also think about computational cost. 
* `min_frac`: this sets the value that the final $\lambda$ value is set to, in the sense that $\lambda_{100} = \lambda{1}\times \text{min frac}$. Making this small means that we will allow denser models to be fit. We set this to $0.01$ to allow denser models to be considered.
* alpha: this is only used for SGL (and SGS in the optional questions): this defines the balance between the variable and group penalties. In genetics, we usually encounter large groups, with many noisy variables. As such, we would prefer to not be limited by the group penalties, in which full groups are picked, but we still want to utilize grouping information. As such, we will use $0.99$.
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
```
So the best model appears to be the one at the $\lambda$ index of $29$, achieving a peak classification score of $93.5%$. Looking back again at the plot showing the number of genes, we see that the best model is not the one using the most amount of genes, but in fact needs only $11$ genes

```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
abline(v = which.max(lasso_df$classification_rate), col = "red")
```

**Q1: what happens when you don't fit an intercept? What about no standardisation?**

**Q2: apply the lasso to the colitis data**

## Group lasso
We move on to fitting the group lasso model. As mentioned in Chapter 2, the `grplasso` package does not perform standardisation properly, so we do this ourselves before fitting. 

```{r}
library(grplasso)
X_gl <- t(t(train_data$X) - apply(train_data$X, 2, mean))
X_gl <- t(t(X_gl) / apply(X_gl, 2, sd))
X_gl <- cbind(1, X_gl)
groups_gl <- c(NA, data$groups)

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
max(lasso_df$classification_rate)
```
```{r}
plot(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
abline(v = which.max(glasso_df$classification_rate), col = "red")
```
The best model appears to be the one at the $\lambda$ index of $95$, achieving a peak classification score of $89.6%$. This is another example highlighting the downside of applying only groupwise sparsity. By being forced to pick all variables in a group as active, we are using a lot of noise variables to form our prediction, leading to a decrease in the classification accuracy of $4%$ in comparison to the lasso. The added complexity of applying a group penalty does not yield any benefit over the simpler lasso. We now turn to SGL to see if this can resolve some of these issues.


**Q3: apply the group lasso to the colitis data**

## Sparse-group lasso (SGL)
```{r}
library(SGL)

sgl_model <- SGL(
  list(x=train_data$X,y=train_data$y),
  index = data$groups,
  type = "logit",
  verbose = TRUE,
  nlam = path_length,
  min.frac = min_frac,
  alpha = alpha,
  maxit = num_iter
)
```

## Comparison of models
### Number of non-zero coefficients
```{r}
plot(apply(lasso_model$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(apply(glasso_model$coefficients, 2, function(x) length(which(x!=0))), type="l", col = "red")
```

### Prediction accuracies

## SLOPE models (optional)

## Classification accuracies

| Model    | Classification accuracy (%) | 
|----------|-----------------------------|
| Lasso    | 93.5                        |
| gLasso   | 89.6                        |
| SGL      | 3.6              |
| SLOPE    | 3.6              | 
| gSLOPE   | 2.2              |
| SGS      | 2.8              |