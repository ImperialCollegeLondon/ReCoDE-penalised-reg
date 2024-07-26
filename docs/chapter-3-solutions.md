---
title: 'Chapter 3 solutions'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---
**Q1: what happens when you don't fit an intercept? What about no standardisation?**
We first turn off the intercept option.
```{r}
library(glmnet)
lasso_model_no_intercept <- glmnet(
  x = train_data$X,
  y = train_data$y,
  family = "binomial",
  lambda.min.ratio = min_frac,
  maxit = num_iter,
  standardize = TRUE,
  intercept = FALSE
) 
```
and we obtain an error to say that convergence for the 99th lambda value was not reached. This is because applying an intercept also centers the data matrix, which helps in the fitting process. Now, let's try removing standardisation.
```{r}
lasso_model_no_sd <- glmnet(
  x = train_data$X,
  y = train_data$y,
  family = "binomial",
  lambda.min.ratio = min_frac,
  maxit = num_iter,
  standardize = TRUE,
  intercept = FALSE
) 
```
No error this time, so let's see how it performs for prediction.
```{r}
lasso_preds_no_sd <- predict(lasso_model_no_sd, test_data$X, type = "class")
lasso_cr_no_sd <- apply(lasso_preds_no_sd, 2, function(x) mean(x == test_data$y))

# put classification scores into data frame
lasso_df_no_sd = data.frame(
  model = "Lasso no sd",
  lambda_index = 1:path_length,
  classification_rate = lasso_cr_no_sd
)
max(lasso_df_no_sd$classification_rate)
```
We obtain a peak accuracy of $90%$, which is lower than the one obtained without standardising ($94%$). Standardising scales the data matrix, which is important in regression models as it allows for more direct comparison between the genes, and this is a demonstration of how it can lead to better predictive performance.

**Q2: apply the lasso to the cancer data**
First, we need to split the data as for the colitis data. There are only $60$ observations this time, so we will split it 50/50.
```{r}
sewd("data")
data = readRDS("cancer-data-c8.RDS")
X_cancer = data$X
y_cancer = data$y
groups_cancer = data$groups
rm(data)
set.seed(100)
training_ind <- sample(1:nrow(X_cancer), 30) 
train_data_cancer <- list(X = X_cancer[training_ind,], y = y_cancer[training_ind])
test_data_cancer <- list(X = X_cancer[-training_ind,], y = y_cancer[-training_ind])
```
We can proceed as before.
```{r}
lasso_model_cancer <- glmnet(
  x = train_data_cancer$X,
  y = train_data_cancer$y,
  family = "binomial",
  lambda.min.ratio = min_frac,
  maxit = num_iter,
  standardize = TRUE,
  intercept = TRUE
) 
```
The fitted values are visualised using
```{r}
plot(lasso_model_cancer)
```
There appears to be a trend towards negative coefficients, indicating that there are genes present which reduce the probability of cancer. Looking at how the variables enter the model:
```{r}
plot(apply(lasso_model_cancer$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
```
At the most saturated point, we have just below $30$ genes in the model. Now, testing this model on new data. 
```{r}
# calculate predictions
lasso_preds <- predict(lasso_model_cancer, test_data_cancer$X, type = "class")
# compare to test data
lasso_cr <- apply(lasso_preds, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
lasso_df_cancer = data.frame(
  model = "Lasso",
  lambda_index = 1:path_length,
  classification_rate = lasso_cr
)
plot(x = lasso_df_cancer$lambda_index, y = lasso_df_cancer$classification_rate, type="l", xlab="Lambda index", ylab = "Classification accuracy")
abline(v = which.max(lasso_df_cancer$classification_rate), col = "red") # where the maximum is located
max(lasso_df_cancer$classification_rate)
which.max(lasso_df_cancer$classification_rate)
apply(lasso_model_cancer$beta, 2, function(x) length(which(x!=0)))[which.max(lasso_df_cancer$classification_rate)]
```
The predictive performance here is much worse than for the colitis dataset. This is to be expected. The development of cancer follows a more complex genetic landscape, making prediction very challenging. The particularly interesting aspect here is that the model does not actually improve the predictive performance by adding genes. The best performing model is the one with no variables present, showing that the lasso was not able to identify any signal in the dataset. 

**Q3 (optional): `glmnet` has the elastic net model. Apply it to the colitis data.**
We investigated the elastic net in Chapter 2. Here, we apply it to the colitis dataset to see if we can improve upon the lasso performance.
```{r}
alpha_seq = seq(from = 0, to = 1, length.out = 20)
alpha_data = data.frame(alpha_val = alpha_seq, pred_score = rep(0,length(alpha_seq)))
for (alpha_id in 1:length(alpha_seq)){
en_model <- glmnet(
  x = train_data$X,
  y = train_data$y,
  family = "binomial",
  lambda.min.ratio = min_frac,
  maxit = num_iter,
  standardize = TRUE,
  intercept = TRUE,
  alpha = alpha_seq[alpha_id]
) 
en_preds <- predict(en_model, test_data$X, type = "class")
en_cr <- apply(en_preds, 2, function(x) mean(x == test_data$y))
alpha_data$pred_score[alpha_id] = max(en_cr)}
plot(alpha_data, type = "b", xlab = "Alpha", ylab = "Classification accuracy")
```
We see that the predictive performance is not hugely sensitive to the choice of $\alpha$, although it is clear that the elastic net can improve over the lasso by over $2%$, if we choose $\alpha$ to be in the region of $[0.5, 0.7]$. For the lasso, we had a peak of $93.5%$ and for elastic net we have obtained $96.1%$.

**Q4: apply the group lasso to the cancer data**
As before:
```{r}
X_gl <- t(t(train_data_cancer$X) - apply(train_data_cancer$X, 2, mean))
X_gl <- t(t(X_gl) / apply(X_gl, 2, sd))
X_gl <- cbind(1, X_gl)
groups_gl <- c(NA, groups_cancer)

lambda_max_group <- lambdamax(X_gl, as.numeric(train_data_cancer$y), groups_gl, standardize = FALSE)
lambdas_gl <- exp(seq(
  from = log(lambda_max_group),
  to = log(lambda_max_group * min_frac),
  length.out = path_length
))
glasso_model_cancer <- grplasso(
  x = X_gl,
  y = as.numeric(train_data_cancer$y),
  index = groups_gl,
  lambda = lambdas_gl,
  standardize = FALSE,
  max.iter = num_iter
)
```

As before, we can see how many variables are entering the model as we decrease $\lambda$:
```{r}
plot(apply(glasso_model_cancer$coefficients, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
```
At the most saturated point, we have about $40$ genes in the model. Performing the prediction
```{r}
glasso_preds <- predict(object = glasso_model_cancer, newdata = cbind(1,test_data_cancer$X), type = "response")
glasso_preds = ifelse(glasso_preds >= 0.5, 1, 0)
glasso_cr <- apply(glasso_preds, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
glasso_df_cancer = data.frame(
  model = "gLasso",
  lambda_index = 1:path_length,
  classification_rate = glasso_cr
)
plot(x = glasso_df_cancer$lambda_index, y = glasso_df_cancer$classification_rate, type="l", xlab="Lambda index", ylab = "Classification accuracy")
abline(v = which.max(glasso_df_cancer$classification_rate), col = "red") # where the maximum is located
max(glasso_df_cancer$classification_rate)
length(which(glasso_model_cancer$coefficients[,which.max(glasso_df_cancer$classification_rate)]!=0))
```
The group lasso obtains a peak accuracy of $60%$ using $15$ genes. In this case, it outperforms the lasso, showing that the grouping information is needed to extract the signal from the cancer genes.

**Q5: apply SGL to the cancer data**
```{r}
library(sgs)
sgl_model = fit_sgs(
  X = train_data_cancer$X,
  y = train_data_cancer$y,
  groups = groups_cancer,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  alpha = alpha,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  v_weights = rep(1, ncol(train_data_cancer$X)),
  w_weights = rep(1, length(unique(groups_cancer)))
)
```
Performing the predictiton:
```{r}
sgl_preds = predict(sgl_model, x = test_data_cancer$X)
sgl_cr <- apply(sgl_preds$class, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
sgl_df = data.frame(
  model = "SGL",
  lambda_index = 1:path_length,
  classification_rate = sgl_cr
)
max(sgl_df$classification_rate)
which.max(sgl_df$classification_rate)
```
SGL obtains a peak accuracy of $56.7%$ at the index of $1$, using no genes:
```{r}
length(sgl_model$selected_var[[which.max(sgl_df$classification_rate)]])
```

**Q6 (optional): apply SLOPE to the cancer data**
```{r}
slope_model_cancer = SLOPE(
  x = train_data_cancer$X,
  y = train_data_cancer$y,
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

slope_preds = predict(slope_model_cancer, x = test_data_cancer$X, type = "response") 
slope_preds = ifelse(slope_preds >= 0.5, 1, 0)
slope_cr <- apply(slope_preds, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
slope_df_cancer = data.frame(
  model = "SLOPE",
  lambda_index = 1:path_length,
  classification_rate = slope_cr
)
max(slope_df_cancer$classification_rate)
sum(slope_model_cancer$nonzeros[,,which.max(slope_df_cancer$classification_rate)])
```
SLOPE is found to have a peak classification rate of $56.7%$ using $43$ genes. So, the same accuracy as the lasso, but actually using genes. However, the genes were not found to be informative.

**Q7 (optional): apply gSLOPE to the cancer data**
```{r}
gslope_model_cancer = fit_gslope(
  X = train_data_cancer$X,
  y = train_data_cancer$y,
  groups = groups_cancer,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-4, 
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE
)

gslope_preds = predict(gslope_model_cancer, x = test_data_cancer$X)
gslope_cr <- apply(gslope_preds$class, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
gslope_df_cancer = data.frame(
  model = "gSLOPE",
  lambda_index = 1:path_length,
  classification_rate = gslope_cr
)
max(gslope_df_cancer$classification_rate)
length(gslope_model_cancer$selected_var[[which.max(gslope_df_cancer$classification_rate)]])
```
gSLOPE obtains a peak accuracy of $33.3%$ using $417$ genes. 

**Q8 (optional): can you achieve a higher predictive accuracy with SGS?**
SGS has a few hyperparameters to play around with. Feel free to try changing the different hyperparameters and seeing what the result is. Here, I will alter two to give you an insight into how they can change the model performance.

We first alter $\alpha$ to be $0.5$.
```{r}
sgs_model_2 = fit_sgs(
  X = train_data$X,
  y = train_data$y,
  groups = groups,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-4,
  vFDR = 1e-4, 
  alpha = 0.5,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  pen_method = 3  
)

sgs_preds = predict(sgs_model_2, x = test_data$X)
sgs_cr <- apply(sgs_preds$class, 2, function(x) mean(x == test_data$y))
max(sgs_cr)
length(sgs_model_2$selected_var[[which.max(sgs_cr)]])
```
This model obtains a peak accuracy of $88.3%$ using $70$ genes. So, changing $\alpha$ has lead to worse performance. This was to be expected, as by moving $\alpha$ away from $0.99$ we also moved the model closer to a gSLOPE model, which performs worse than SLOPE for the colitis data.

Next, we alter the penalty sequences through the choice of the FDR hyperparameters (`gFDR` and `vFDR`). I will set these to be significantly smaller, in the hope of inducing more sparsity in the model. The aim here is to try to remove as much noise from the model as possible (without removing all of the signal). 
```{r}
sgs_model_3 = fit_sgs(
  X = train_data$X,
  y = train_data$y,
  groups = groups,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-10,
  vFDR = 1e-10, 
  alpha = alpha,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  pen_method = 3  
)

sgs_preds = predict(sgs_model_3, x = test_data$X)
sgs_cr <- apply(sgs_preds$class, 2, function(x) mean(x == test_data$y))
max(sgs_cr)
length(sgs_model_3$selected_var[[which.max(sgs_cr)]])
```
This model obtains a peak accuracy of $97.4%$ using only $9$ genes. Here, we have an example of how the sparse-group models can be powerful when implemented correctly. This is the highest predictive accuracy we have obtained so far.

**Q9 (optional): apply SGS to the cancer data**
```{r}
sgs_model_cancer = fit_sgs(
  X = train_data_cancer$X,
  y = train_data_cancer$y,
  groups = groups_cancer,
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

sgs_preds = predict(sgs_model_cancer, x = test_data_cancer$X)
sgs_cr <- apply(sgs_preds$class, 2, function(x) mean(x == test_data_cancer$y))

# put classification scores into data frame
sgs_df_cancer = data.frame(
  model = "SGS",
  lambda_index = 1:path_length,
  classification_rate = sgs_cr
)
max(sgs_df_cancer$classification_rate)

length(sgs_model$selected_var[[which.max(sgs_df$classification_rate)]])
```
SGS obtains a peak accuracy of $56.6$ using $23$ genes. However, we could try changing the sequence (as we did for Q8), to see if we get better performance
```{r}
sgs_model_cancer_2 = fit_sgs(
  X = train_data_cancer$X,
  y = train_data_cancer$y,
  groups = groups_cancer,
  type = "logistic",
  path_length = path_length,
  min_frac = min_frac,
  gFDR = 1e-10,
  vFDR = 1e-10, 
  alpha = alpha,
  max_iter = num_iter,
  screen = TRUE,
  intercept = FALSE,
  verbose = TRUE,
  pen_method = 3
)

sgs_preds = predict(sgs_model_cancer_2, x = test_data_cancer$X)
sgs_cr <- apply(sgs_preds$class, 2, function(x) mean(x == test_data_cancer$y))
max(sgs_cr)
length(sgs_model$selected_var[[which.max(sgs_cr)]])
```
In this case, this did not help, as we stay at the same accuracy.

## Comparison
We end the section by comparing all of the models on the cancer dataset.

### Number of non-zero coefficients
```{r}
plot(apply(lasso_model_cancer$beta, 2, function(x) length(which(x!=0))), type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(apply(glasso_model_cancer$coefficients, 2, function(x) length(which(x!=0))), type="l", col = "red")
lines(lapply(sgl_model_cancer$selected_var,length), type = "l", col = "brown")
lines(apply(slope_model_cancer$nonzeros,3,sum), type = "l", col = "blue")
lines(lapply(gslope_model_cancer$selected_var,length), type = "l", col = "green")
lines(lapply(sgs_model_cancer$selected_var,length), type = "l", col = "purple")
legend("topright", legend = c("Lasso", "gLasso", "SGL", "SLOPE", "gSLOPE", "SGS"),
       col = c("black", "red", "brown", "blue", "green", "purple"), lty = 1)
```

### Prediction accuracies
```{r}
plot(lasso_df_cancer$classification_rate, type="l", xlab="Lambda index", ylab = "Number non-zero")
lines(glasso_df_cancer$classification_rate, type="l", col = "red")
lines(sgl_df_cancer$classification_rate, type = "l", col = "brown")
lines(slope_df_cancer$classification_rate, type = "l", col = "blue")
lines(gslope_df_cancer$classification_rate, type = "l", col = "green")
lines(sgs_df_cancer$classification_rate, type = "l", col = "purple")
legend("topright", legend = c("Lasso", "gLasso", "SGL", "SLOPE", "gSLOPE", "SGS"),
       col = c("black", "red", "brown", "blue", "green", "purple"), lty = 1)
```

### Prediction accuracies on cancer dataset
| Model    | Classification accuracy (%) | Genes used |
|----------|-----------------------------|------------|
| Lasso    | 56.7                        | 0          |
| gLasso   | 60.0                        | 15         |
| SGL      | 56.7                        | 0          |
| SLOPE    | 56.7                        | 43         |
| gSLOPE   | 33.3                        | 417        | 
| SGS      | 56.7                        | 23         |

