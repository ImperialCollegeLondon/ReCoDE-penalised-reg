---
title: 'Chapter 2 solutions'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---
# Chapter 2 solutions
## Q1: run the same model again but without standardising. What do you observe?
As before, we run
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

library(glmnet)
fit <- glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1)
```
We also run the model without standardising
```{r}
fit_ns <- glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1, standardize = FALSE)
cbind(fit$beta[,20],fit_ns$beta[,20])
```
The solutions are similar but not quite the same. A lot of things happen in the background when standardisation occurs. Without getting into too much detail, standardisation scales $\lambda$, to ensure we get similar solutions, but this scaling is only approximate, hence the difference. Generally, it is recommended to use standardisation (and when using packages as complete as `glmnet`, to let the package handle it).

## Q2: what happens if we do not specify how many $\lambda$'s we want?
If we remove the lambda options:
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

library(glmnet)
fit <- glmnet(x = X, y = y, family = "gaussian")
length(fit$lambda)
```
We see that we are now using 65 $\lambda$ values. The default value for the function is actually 100, but the function will also end the path early if the deviance ratio (`fit$dev.ratio`) is close to 1, which happened here.

## Q3 (optional): the glmnet documentation states that there is a parameter $\alpha$ that we can change. What does this do?**
The `glmnet` function actually runs the elastic net model, which is defined by
$$
\hat{\beta}_\text{elastic net} = \min_\beta \left\{ \frac{1}{2}\left\|y- X \beta\right\|_2^2 + \lambda \alpha\left\| \beta \right\|_1+\lambda 
(1-\alpha)/2\left\| \beta \right\|_2^2 \right\}.
$$
It uses a combination of the lasso and ridge (similar to how SGL combines the lasso and group lasso), balanced through the $\alpha$ parameter. By default, $\alpha$ is set to 1, so that it reduces to the lasso, which is why we have not had to worry about it so far. However, elastic net has been proposed as an extension to the lasso which overcomes many of its issues, so we compare their performances here.
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)

library(glmnet)
alpha_seq = seq(0,1,length.out=20)
preds = data.frame(alpha = alpha_seq, error = rep(0,length(alpha_seq)))
for (i in 1:length(alpha_seq)){
    set.seed(2)
    fit.cv <- cv.glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1, alpha = alpha_seq[i])
    preds$error[i] = mean((y_new-predict(object = fit.cv, newx = X_new, s = "lambda.1se"))^2)
}
plot(preds$alpha, preds$error, type = "b")
```
Clearly the model performs badly for $\alpha = 0$, which is ridge regression. This is not surprising, as ridge regression does not shrink coefficients exactly to zero, so if we take a close look at the coefficients, we notice that it is forced to keep coefficients active
```{r}
set.seed(2)
fit.cv.ridge <- cv.glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1, alpha = 0)
cbind(beta, fit.cv.ridge$glmnet.fit$beta[,20])
```
Looking at the predictions in more detail
```{r}
preds
```
We can see that $\alpha = 0.94736842$ achieves the lowest error (which is similar to the recommended $\alpha$ value for SGL). Comparing the coefficients it is clear to see why this value works (as we are able to make inactive coefficients exactly zero)
```{r}
set.seed(2)
fit.cv.best <- cv.glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1, alpha = 0.94736842)
cbind(beta, fit.cv.ridge$glmnet.fit$beta[,20], fit.cv.best$glmnet.fit$beta[,20])
```

## Q4 (optional): look at the glmnet documentation - which parameters might be interesting to vary?
There are many options to alter in the `glmnet` function. Generally, it is best to leave them as default, as they have been set to sensible values by the authors, unless you have a reason to change them. Some ones of interest are:

- `alpha`: this is discussed in Q3.
- `standardize`: a TRUE/FALSE indicator as to whether the data is standardised. It is good practice to standardise data, so this should be left on. It is not a good idea to standardise the data yourself and then feed this to `glmnet`, because standardisation alters how $\lambda$ is used (it scales $\lambda$ in the backend, see Q1).
- `intercept`: a TRUE/FALSE indicator as to whether an intercept is fit. Again, it is good practice to leave this on, unless you have a strong reason to believe that your regression line goes through the origin (i.e., that your response is centered at 0, which is rare). You also should not center your response yourself, as again various changes occur in the backend if this is set to on. The two options described show that you do not need to do these pre-processing steps yourself, `glmnet` will do it for you.
- `penalty.factor`: this allows you to use adaptive penalty weights, which leads to the *adaptive lasso* (as in SLOPE, see the optional section, although note that this function can not be used for SLOPE, due to the sorting component of SLOPE). 

## Q5: instead of using the predict function, manually code a prediction.
A linear model is defined as
$$ 
y = X\beta + \epsilon
$$
so, to form a prediction, we need to plug in our values of $\beta$ values that we found in the model. We can do that using
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

library(glmnet)
fit <- glmnet(x = X, y = y, family = "gaussian", lambda = 0.5, intercept = FALSE)

preds_1 = predict(object = fit, newx = X)
preds_2 = X%*%fit$beta
cbind(preds_1[1:5], preds_2[1:5])
```

## Q6: compare the predictions of the 1se model against the min model.
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)

library(glmnet)
fit.cv <- cv.glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1)
mean((y_new-predict(object = fit.cv, newx = X_new, s = "lambda.1se"))^2)
mean((y_new-predict(object = fit.cv, newx = X_new, s = "lambda.min"))^2)
```
In this case, the minimum value actually obtains a lower predictive error, but generally it is still recommended to use the 1se model to reduce variance (overfitting).

## Q7: compare the `grplasso` package to `glmnet` to see if standardisation works properly?
To do this, we need to reduce the group lasso to the lasso. We can use singleton groups (each variable in its own group), so that the two models are equivalent. We have set the `grplasso` model up to use the in-built standardisation 
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

library(glmnet)
library(grplasso)
lasso_model <- glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1)
glasso_model <-  grplasso(x = cbind(1,X), y = y, index = c(NA,1:ncol(X)), lambda = lasso_model$lambda, standardize = TRUE, center = TRUE, model = LinReg())
```

Comparing the final $\lambda$ solution, we see that they are not the same, so the built-in standardisation is not working as intended.
```{r}
cbind(lasso_model$beta[,20], glasso_model$coefficients[-1,20])
```

## Q8: set the group indexing so that the signal variables are all in the same group. What do you observe?
Choosing a different grouping, so that the signal variables are in the same group, would lead to the following
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)
groups <- c(rep(1,4),rep(2,4),rep(3,4),rep(4,4),rep(5,4))
groups_gl <- c(NA, groups) # we add NA as the intercept does not belong to a group

# generate lambda sequence
lambda_max_group <- lambdamax(X_gl, y, groups_gl, standardize = FALSE, center = FALSE) # finding lambda max, we can ignore the warning
lambdas_gl <- exp(seq( # calculating the full lambda sequence
  from = log(lambda_max_group),
  to = log(lambda_max_group * 0.1), # the equivalent of lambda.min.ratio in glmnet
  length.out = 20 # how many lambdas we want
))

# fit model
glasso_model <- grplasso(x = X_gl, y = y, index = groups_gl, lambda = lambdas_gl, standardize = FALSE, center = FALSE, model = LinReg())

cbind(beta,glasso_model$coefficients[-1,20])

mean((y_new-predict(object = glasso_model, newdata = cbind(1,X_new))[,20])^2)
```
We now see that we are no longer selecting a lot of zero variables, although surprisingly, this actually makes the prediction error larger. 

## Q9 (optional): can you figure out why we get inflated values?
If we turn standardisation off we get:
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)
library(SGL)
sgl_model <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 20, min.frac = 0.1, alpha = 0.95)
sgl_model_2 <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 20, min.frac = 0.1, alpha = 0.95, standardize = FALSE)

cbind(beta, sgl_model$beta[,20], sgl_model_2$beta[,20])
```
So it appears that standardisation is not properly implemented in the `SGL` package. This highlights the issue of pre-processing when it is done incorrectly.

## Q10: what happens to the predictive score if we allow $\lambda$ to decrease even further?
We can test the predictive score by decreasing $\lambda$ to quite an extreme minimum and allowing for more $\lambda$ values along the path:
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)

library(SGL)
sgl_model <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 200, min.frac = 0.001, alpha = 0.95)

preds = rep(0,200)
for (i in 1:200){
    preds[i] = mean((y_new-predictSGL(x = sgl_model, newX = X_new, lam = i))^2)
}
plot(preds,type="l")
```
We see that after a certain point, decreasing $\lambda$ further does not provide any additional benefit, only adding more model complexity. Generally, we prefer to use the simplest model that is available, without sacrificing accuracy (a concept known as Occam's Razor).

We can try an even smaller value of `min.frac`
```{r}
sgl_model <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 200, min.frac = 0.000001, alpha = 0.95)
mean((y_new-predictSGL(x = sgl_model, newX = X_new, lam = 20))^2)

preds = rep(0,20)
for (i in 1:20){
    preds[i] = mean((y_new-predictSGL(x = sgl_model, newX = X_new, lam = i))^2)
}
plot(preds,type="b")
```
The prediction error continues to decrease but we are adding a lot of variance into the model by overfitting. This is ok for a simple example like this, but when there are more predictors it can become problematic, especially if the divide between signal and noise is less apparant.

## Q11: vary $\alpha$ in the region $[0,1]$. What do you observe?
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)
library(SGL)

alpha_seq = seq(0,1,length.out=20)
preds = data.frame(alpha = alpha_seq, error = rep(0,length(alpha_seq)))
for (i in 1:length(alpha_seq)){
    set.seed(2)
    fit <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 20, min.frac = 0.001, alpha = alpha_seq[i])
    preds$error[i] = mean((y_new-predictSGL(x = fit, newX = X_new, lam = 20))^2)
}
plot(preds$alpha, preds$error, type = "b")
```
We observe the error shrinking as $\alpha$ gets close to 1 (the lasso). This is expected in this scenario, as we did not add a grouping structure to the synthetic data.

## Q12: can you use the `SGL` R package to fit the lasso and group lasso?
To do this, we just set `alpha` to 0 and 1:
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)

library(SGL)
fit_glasso <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 20, min.frac = 0.001, alpha = 0)
fit_lasso <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 20, min.frac = 0.001, alpha = 1)
```
Even though this will give results very similar to the `glmnet` and `grplasso` packages, it is still recommended to use the package specific to the model, as it will be more optimised.