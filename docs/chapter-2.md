---
title: 'Chapter 2: Motivation, Penalised regression, and the lasso'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---
This chapter will provide the (genetics) motivation for the problem we are trying to solve. It will then provide mathematical descriptions of the methods that we will use, including penalised regression and the lasso.
R code is provided for implementing the methods to simple examples, which will be expanded to the genetics data in Chapter 3.

The aim of this chapter is to provide the genetics and mathematical background that allows a student to understand the problem we are trying to solve and the methods we will use to solve it. This will help the student to adapt the methods accordingly to our problem and think about how they can be applied to other problems.

## Motivation (optional)
The genetics background for understanding the motivation was described in Chapter 1. 
Here, we give a detailed description of the problem we are trying to solve, from a genetics point of view. The mathematical formulation of the problem is described in the next section.

A key problem in genetics is to identify which genotypes (genetic markers) are associated with phenotypes (trait).
Early approaches of identifying these genotypes used univariate analysis, where each gene was independently tested for association with the outcome.
In these analyses, simple statistical tests, such as the t-test, are used to calculate p-values on the significance of single gene expressions on a phenotype [R1]. 
However, genes sit within groups (called pathways), and so these approaches do not fully utilise the biological information available. 
We might expects genes in the same pathways to behave in similar ways and interact with each other.

Pathway analysis, also known as gene set analysis, attempts to make use of the information about the biological set-up of the genes being investigated, with the hope of this leading to more accurate models. 
As such, the aim of pathway analysis is to identify pathways, which are groups of genes, that have an association with a particular phenotype. 
Popular examples of pathway analysis methods include over-representation analysis (ORA), functional class scoring (FCS) methods, and pathway topology analysis (PTA) [R1].

The problem of selecting relevant pathways can be seen as both an individual variable and group selection problem - we want to find both the relevant pathways and the genes within those pathways.
Methods for variable selection are countless, ranging from simple linear regression to more sophisticated approaches, such as neural networks [R2, R3]. 

In pathway analysis, we use datasets with more genes ($p$) than individuals ($n$) to which they belong. As such, pathway analysis falls under the $p>>n$ umbrella of problems (this is the field of high-dimensional statistics), commonly known as *short, fat data problems*. 
Additionally, as only a small fraction of genes and pathways tend to be disease-associated, we seek methods which encourage sparse solutions [R4, R5].

A particular family of methods which are often used to solve $p>>n$ problems and generate sparse solutions are penalised regression methods, which we discuss next.

## Penalised regression
*Note on notation*: $\beta_i \in \mathbb{R}$ will refer to the coefficient for a variable $i$, whilst $\beta^{(g)} \in \mathbb{R}^{m_g}$ refers to the vector of coefficients for the variables in group $g$ (of which there are $m_g$).

In the traditional linear regression setting, we have response data $y$, of dimension $n$, and input data $X$, of dimension $n\times p$, and a corresponding model given by $y = X\beta + \epsilon$, where $\epsilon \sim N(0,\sigma^2), \sigma^2>0$. 
With the rise of big data, we are increasingly tackling datasets where $p>>n$ (which falls under the umbrella of high-dimensional statistics), including in cyber security, biomedical imaging, and as mentioned above, pathway analysis. 
In such cases, there are insufficient degrees of freedom to estimate the full model. Indeed, the solution to linear regression (ordinary least squares) can not be computed for high-dimensional datasets, as the inverse of $X$ does not exist (which is needed for the solution).

### Lasso
As a solution for dealing with such a situation, [R6] introduced the *least absolute shrinkage and selection operator* (lasso). The lasso performs variable selection by regularisation; that is, it minimises a loss function subject to some constraints. 
Formally, the lasso finds $\beta$ estimates by solving

$$
\hat{\beta}_\text{lasso} = \min_\beta \left\{ \frac{1}{2}\left\|y- X \beta\right\| _2^2 + \lambda \left\| \beta \right\|_1 \right\},
$$
where $\left\| \cdot \right\|_1$  is the $L^1$ norm and $\left\| \cdot \right\|_2$ is the $L^2$ norm. The parameter $\lambda$ defines the amount of sparsity in the fitted model. 
If $\lambda$ is large, very few coefficients will be non-zero and the model will be very sparse. On the other hand, small values of $\lambda$ will lead to a model with many non-zero coefficients, eventually leading to $\lambda = 0$, where we recover the ordinary least squares solution.
We can pick $\lambda$ subjectively ourselves, but the most common approach is to fit models for different values of $\lambda$ and select the best one (the one with the lowest error) using cross-validation.
The approach generates sparse solutions by shrinking some coefficients and setting others to 0 exactly, retaining the desirable properties of subset selection and ridge regression [R6] - this is shown in the figure below.

![Solutions of the lasso (left) and ridge regression (right) for $p=2$. The blue regions are the constraint regions and the red eclipses are the contour lines of the least squared errors function. The solutions are given by where the contours hit the constraint region. This figure is from [R7].](asset/images/ridgevslasso.png)

As a consequence of the diamond shape of the lasso constraint region, if the solution occurs at the corner, the corresponding parameter $\beta_j$ is set exactly to 0 [R7], which is not possible with the ridge constraint region. For $p>2$, the diamond is a rhomboid, which has many edges, and we retain this desirable property of the lasso.

We can implement the lasso in R using the `glmnet` package, which is one of the most widely used packages. To run the lasso, we create some synthetic Gaussian data
```{r}
set.seed(2)
X <- matrix(rnorm(100 * 20), 100, 20)
beta <- c(rep(5,5),rep(0,15))
y <- X%*%beta + rnorm(100)
```

We can run the lasso using the default set-up, specifying only that we want to run the model for 20 values of $\lambda$.
```{r}
library(glmnet)
fit <- glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1)
```

**Q1: run the same model again but without standardising. What do you observe?** 

**Q2: what happens if we do not specify how many lambdas we want?**

**Q3 (optional): the glmnet documentation states that there is a parameter alpha that we can change. What does this do?**

**Q4 (optional): look at the glmnet documentation - which parameters might be interesting to vary?**

The model will be run from the value of $\lambda$ that generates a null model (no active variables) to the final value of $\lambda$ that is a specified proportion of the first value (the value of `lambda.min.ratio`).
We can visualise the path of $\lambda$ values, which take a log-linear shape (by design)
```{r}
plot(fit$lambda,type="b")
```
We can also visualise some of the active coefficients using
```{r}
plot(fit)
```
and we can observe that the coefficients become larger in absolute value as the value of $\lambda$ become smaller.
So we now have 20 lasso models and we want to pick a single one to use for prediction. As mentioned, we can use cross-validation:
```{r}
fit.cv <- cv.glmnet(x = X, y = y, family = "gaussian", nlambda = 20, lambda.min.ratio = 0.1)
print(fit.cv)
```
In the print-out, there are actually two different "minimums" defined. The first is `min`, which is the absolute minimum error obtained by any of the models.
The other is `1se`, which is the simplest model that is within one-standard error of the minimum. This is often used as a more conservative estimate of the best model, as it is less likely to be overfitting the data.
Using the `1se` model, we can compare the obtained coefficients to the true beta values
```{r}
cbind(beta,fit.cv$glmnet.fit$beta[,19])
```
and we find the lasso does very well. It correctly picks out the first five variables as those with a signal and correctly sets the rest to zero. 
We can now use these fitted coefficients to predict y. To fairly test the accuracy of the model, we first need to generate some new data (that uses the same $\beta$ signal)
```{r}
set.seed(3)
X_new <- matrix(rnorm(10 * 20), 10, 20)
y_new <- X_new%*%beta + rnorm(10)
```
and we can predict the new y values using the lasso model
```{r}
cbind(y_new,predict(object = fit.cv, newx = X_new, s = "lambda.1se"))
```
and we find our predictions are not too bad. We can get a prediction error score (using the mean squared error)
```{r}
mean((y_new-predict(object = fit.cv, newx = X_new, s = "lambda.1se"))^2)
```

**Q5: instead of using the predict function, manually code a prediction.**

**Q6: compare the predictions of the 1se model against the min model.**


### Group lasso
We mentioned in the motivation that genes come in groups and that we would like to utilize this grouping information. 
To that end, [R8] adapted the lasso approach to the problem of selecting grouped variables by introducing the *group lasso* (gLasso). 
Let $X^{(1)}, \dots, X^{(G)}$ be non-overlapping groups of variables (all groups are assumed to be non-overlapping), then the solution is given by

$$
\hat{\beta}_\text{gLasso} = \min_{\beta} \left\{ \frac{1}{2}\left\|y-\sum_{g=1}^{G} X^{(g)} \beta^{(g)} \right\|_2^2 + \lambda  \sum_{g=1}^{G} \sqrt{p_g} \left\| \beta^{(g)} \right\|_2 \right\},
$$
where $\sqrt{p_g}$ is the number of variables in group $g$. If the size of each group is 1, then we recover the lasso solution. 
The group lasso creates sparsity at a group level by shrinking whole groups exactly to 0, so that each variable within a group is shrunk to 0 exactly. 
The group lasso has found widespread use in finding significant genetic variants [R9, R10].

There are two main R packages for fitting the group lasso: `grplasso` and `gglasso`. 
We will use the `grplasso` package, as it contains additional features. To use the group lasso, we need grouping indexes for the variables. 
In genetics, natural groupings are found through pathways (which we saw in Chapter 1). If a natural grouping does not exist, algorithms that perform some form of grouping can be used (such as k-means clustering). 
For simplicity, in this case we will manually assign the variables into groups of size 4.
```{r}
groups <- rep(1:5,4)
```
The `grplasso` package does not automatically calculate a path of $\lambda$ values, so we compute those first, starting with the first $\lambda$ value using the `lambdamax` function from the package.
Additionally, the package does not implement an intercept and standardisation in the same way as `glmnet`, so we do this manually. In general, when working with penalised regression models, it is a good idea to fit an intercept and apply standardisation.
```{r}
# standardise data
X_gl <- t(t(X) - apply(X, 2, mean)) # center data
X_gl <- t(t(X_gl) / apply(X_gl, 2, sd)) # scale data
X_gl <- cbind(1, X_gl) # add intercept
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
```
Comparing the model coefficients to the true ones (for the most saturated model, removing the intercept value):
```{r}
cbind(beta,glasso_model$coefficients[-1,20])
```
and using it to predict the output (you will notice we have to do a few things manually, such as adding the intercept to `X_new`, this is because many R packages do not have as complete features as `glmnet`)
```{r}
mean((y_new-predict(object = glasso_model, newdata = cbind(1,X_new))[,20])^2)
```
This case demonstrates a real limitation of the group lasso. In our manual grouping, we placed a signal variable (one that is not zero) in each group, forcing the group lasso to pick each of those groups as non-zero. The group lasso picks whole groups, so that it is in turn forced to make every variable in those groups non-zero.
In the next section, we explore a solution to this issue, which is especially limiting in genetics.

**Q7: set the group indexing so that the signal variables are all in the same group. What do you observe?**

### Sparse-group lasso
Using the group lasso for pathway analysis would require the assumption that all genes in a significant pathway are also significant, not allowing for additional sparsity within a group [R11]. 
So, one may wish to have sparsity at both the group (pathway) and variable (gene) level. 

To fulfil this wish, [R9] introduced the *sparse-group lasso* (SGL), which combines traditional lasso with the group lasso to create models with bi-level sparsity. 
The solution to SGL is given by
$$
\hat{\beta}_\text{SGL}= \min_{\beta} \left\{ \frac{1}{2n}\left\|y-\sum_{g=1}^{G} X^{(g)} \beta^{(g)} \right\| _2^2 + (1-\alpha)\lambda  \sum_{g=1}^{G} \sqrt{p_g} \left\| \beta^{(g)} \right\|_2 + \alpha \lambda \left\| \beta \right\|_1\right\},
$$
where $\alpha\in [0,1]$ controls the level of sparsity between group and variable sparsity. If $\alpha = 1$, we recover the lasso, and $\alpha=0$ recovers the group lasso. 
The figure shows how SGL is a convex combination of these two approaches.

![Contour lines for the group lasso (dotted), lasso (dashed), and sparse-group lasso (solid) for $p=2$ [R13].](assets/images/all_lasso1.png)

In pathway analysis, the proportion of relevant pathways, amongst all pathways, is often very low. Additionally, the proportion of relevant genes within a particular pathway is also low. As such, the SGL model can provide the required level of sparsity at both the pathway and the individual gene level. Indeed, SGL has already been applied to detecting significant genetic variants in [R11]. 

### SLOPE (optional)

## R package links
- [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html)
- [grplasso](https://cran.r-project.org/web/packages/grplasso/index.html)
- [gglasso](https://cran.r-project.org/web/packages/gglasso/index.html)
- [SGL](https://cran.r-project.org/web/packages/SGL/index.html)
- [SLOPE](https://cran.r-project.org/web/packages/SLOPE/index.html)
- [grpSLOPE](https://cran.r-project.org/web/packages/grpSLOPE/index.html)
- [sgs](https://cran.r-project.org/web/packages/sgs/index.html)

## References
- [R1] F. Maleki, K. Ovens, D. J. Hogan, and A. J. Kusalik. Gene Set Analysis: Challenges, Opportunities,
and Future Research. Frontiers in Genetics, 11(June):1-16, 2020. ISSN 16648021. doi: 10.3389/
fgene.2020.00654.
- [R2] G. Heinze, C. Wallisch, and D. Dunkler. Variable selection - A review and recommendations
for the practicing statistician. Biometrical Journal, 60(3):431-449, 2018. ISSN 15214036. doi:
10.1002/bimj.201700067.
- [R3] M. Ye and Y. Sun. Variable selection via penalized neural network: A drop-out-one loss approach.
35th International Conference on Machine Learning, ICML 2018, 13:8922-8931, 2018.
- [R4] C. Yang, X. Wan, Q. Yang, H. Xue, and W. Yu. Identifying main effects and epistatic interactions
from large-scale SNP data via adaptive group Lasso. BMC Bioinformatics, 11(SUPPLL.1):1-11,
2010. ISSN 14712105. doi: 10.1186/1471-2105-11-S1-S18.
- [R5] Y. Guo, C. Wu, M. Guo, Q. Zou, X. Liu, and A. Keinan. Combining sparse group lasso and linear
mixed model improves power to detect genetic variants underlying quantitative traits. Frontiers
in Genetics, 10(APR):1-11, 2019. ISSN 16648021. doi: 10.3389/fgene.2019.00271.
- [R6] R. Tibshirani. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical
Society., 58(1):267-288, 1996.
- [R7] T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning Data Mining,
Inference, and Prediction, Second Edition, volume 103. Springer New York, 2nd ed. 20 edition,
2009. ISBN 9780387848587.
- [R8] M. Yuan and Y. Lin. Model selection and estimation in regression with grouped variables. Journal of
the Royal Statistical Society. Series B: Statistical Methodology, 68(1):49-67, 2006. ISSN 13697412.
doi: 10.1111/j.1467-9868.2005.00532.x.
- [R9] J. Li, Z.Wang, R. Li, and R.Wu. Bayesian group lasso for nonparametric varying-coefficient models
with application to functional genome-wide association studies. Annals of Applied Statistics, 9
(2):640-664, 2015. ISSN 19417330. doi: 10.1214/15-AOAS808.
- [R10] M. Lim and T. Hastie. Learning interactions via hierarchical group-lasso regularization. J Comput
Graph Stat., 24(3):627-654, 2015. doi: 10.1080/10618600.2014.938812.
- [R11] Y. Guo, C. Wu, M. Guo, Q. Zou, X. Liu, and A. Keinan. Combining sparse group lasso and linear
mixed model improves power to detect genetic variants underlying quantitative traits. Frontiers
in Genetics, 10(APR):1-11, 2019. ISSN 16648021. doi: 10.3389/fgene.2019.00271.
- [R12] N. Simon, J. Friedman, T. Hastie, and R. Tibshirani. A sparse-group lasso. Journal of Compu-
tational and Graphical Statistics, 22(2):231-245, 2013. ISSN 10618600. doi: 10.1080/10618600.
2012.681250.
- [R13] J. Friedman, T. Hastie, and R. Tibshirani. A note on the group lasso and a sparse group lasso.
pages 1-8, 2010. URL http://arxiv.org/abs/1001.0736.