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
**Q7: set the group indexing so that the signal variables are all in the same group. What do you observe?**
Choosing a different grouping, so that the signal variables are in the same group, would lead to the following
```{r}
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

**Q9: what happens to the predictive score if we allow $\lambda$ to decrease even further?**
We can test the predictive score by decreasing $\lambda$ to quite an extreme minimum and allowing for more $\lambda$ values along the path:
```{r}
set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)

sgl_model <- SGL(list(x=X,y=y), groups, type = "linear", nlam = 200, min.frac = 0.001, alpha = 0.95)

preds = rep(0,200)
for (i in 1:200){
    preds[i] = mean((y_new-predictSGL(x = sgl_model, newX = X_new, lam = i))^2)
}
plot(preds,type="l")
```
We see that after a certain point, decreasing $\lambda$ further does not provide any additional benefit, only adding more model complexity. Generally, we prefer to use the simplest model that is available, without sacrificing accuracy (a concept known as Occam's Razor).