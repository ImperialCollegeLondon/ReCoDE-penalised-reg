---
title: 'Chapter 1 solutions'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---
## Q1: download the genetics dataset with the ID `GDS807`. Do you get the same objects?

This dataset contains genetic information from patients suffering from breast cancer. We extract the data as before
```{r}
raw_data <- getGEO('GDS807')
eset_data <- GDS2eSet(raw_data, do.log2 = TRUE)
```
We notice that an error regarding NaNs is printed. This is due to the log transformation working on negative data, but is not a concern as we will filter this out anyway. 

We can check the objects in the `eset_data` object
```{r}
head(eset_data$sample)
table(eset_data$disease.state)
head(eset_data$description)
```
We have the same objects for the colitis data, but this time we have 32 controls and 28 patients suffering from the disease. 

## Q2: create the data matrix for the dataset `GDS807`.

As before, we run
```{r}
X <- t(exprs(eset_data))
```
We do an initial check for missingness
```{r}
sum(is.na(X))
```
and observe that there is a high amount of missingness. A nice way to profile missingness in a data matrix is to use the `aggr` function from the `VIM` package.
Running the function on the full data matrix will be too computationally intensive, so we run it on a small subset to illustrate the usefulness of the function.
```{r}
library(VIM)
aggr(X[,1000:1020])
```
We observe on the left plot that there is a fair amount of variance in the amount of missing values between the genes. Some genes have no missing values, while others have only missing values.
On the right plot, we can see which combinations of missingness occur most often. The most frequently occuring combination is of ten genes. This plot is very useful for finding patterns of missingness.

We will remove the missing values at the end of this section.

**Q3: run the full pipeline with the dataset `GDS807`.**

We now complete the data pipeline with this dataset.

### Response
Checking the numerical assignment:
```{r}
levels(pData(eset_data)$disease.state)
unique(as.numeric(pData(eset_data)$disease.state))
```
Unlike in the colitis case, here a value of 2 indicates a patient with the disease. So we assign the response as
```{r}
y <- (as.numeric(pData(eset_data)$disease.state) != 2)
```
The encoding of TRUE/FALSE is equivalent to a 0,1 encoding. We could just as easily have assigned it the other way round - it would not change the modelling process.
Our response is now ready. 

### Grouping structure index
As before, we are using the gene set collections here. This time, we use the C8 collection (although you can also use the C3, or any other).
The C8 collection contains cell type signature gene sets. The file is provided in the GitHub repository (`data/gene-sets`).
```{r, results='hide'}
geneset_data <- GSA.read.gmt("data/gene-sets/c8.all.v2023.2.Hs.symbols.gmt")
```

We perform the matching as before. 
```{r}
gene_identifiers <- Table(raw_data)[,2] # these are the gene names in X
num_geneset <- length(geneset_data$genesets) # the number of gene sets

index <- rep(0,length(gene_identifiers)) # these will be the group indexes, which will be found by matching the genes to the gene sets

for(i in 1:num_geneset){ # loop match over each gene set
  matched_index <- match(geneset_data$genesets[[i]],gene_identifiers)
  index[matched_index] <- i
}
```
We check which genes have been matched to a gene set (note that not every gene will have a match, in which case we drop those genes)
```{r}
ind_include <- which(index != 0)
matched_gene_names <- gene_identifiers[ind_include]
X <- X[,ind_include]
dim(X)
```

We now need to reperform the matching with the remaining genes
```{r}
group_index <- rep(0,ncol(X))
for(i in 1:num_geneset){ # iterate over each gene set
  for(j in 1:length(geneset_data$genesets[[i]])){ # iterate over each gene in the gene set
    change.ind <- match(geneset_data$genesets[[i]][j],matched_gene_names) # matching gene to gene set
    if(!is.na(change.ind)){ # if the gene is in the gene set
      if(group_index[change.ind] == 0){ # if the gene has not been assigned a group index
        group_index[change.ind] <- i
      }
    }
  }
}

head(group_index)
```

### Removing missingness
As a final step, we need to remove the missing values. To do this, we need two functions.
The first function calculates the proportion of missingness for each gene (column). 
The second function imputes the missing values with the mean of the non-missing values for each gene.
```{r}
prop.missing <- function(X){ 
  apply(X,2,function(x){mean(is.na(x))})
}

mean.impute <- function(X){
  means <- apply(X,2,function(x){mean(x[which(!is.na(x))])})
  for(i in 1:ncol(X)){
    ind <- which(is.na(X[,i]))
    X[ind,i] <- means[i]
  }
  return(X)
}
```

We now apply these two functions to the data matrix. We will remove all genes with more than 50% missingness. 
For the remaining genes, we will mean impute any missing values. 
```{r}
prop.m <- prop.missing(X)
remove.ind <- which(prop.m > 0.5)
imp.X <- mean.impute(X[,-remove.ind])
X <- imp.X
dim(X)
```

As we have removed variables (and this should be checked regardless), we need to reset the group indexing so that it runs from 1,2,...
Many of the models we will use do not work unless this is done. We reset the indexing by looping over each gene set and assigning the group index to the genes in the gene set.
```{r}
group_index = group_index[-remove.ind]
grp_ids = data.frame(col_id = 1:length(unique(group_index)),grps=unique(group_index))
ordered_group_indx = rep(0,length(group_index))
for (i in 1:length(group_index)){
  ordered_group_indx[i] = which(grp_ids$grps == group_index[i])
}
```

### Save dataset
We can now save the cleaned dataset (we will save it as an RDS file)
```{r}
setwd("data")
data = list()
data$X = X
data$y = y
data$groups = ordered_group_indx
saveRDS(data,"cancer-data-c8.RDS")
```
