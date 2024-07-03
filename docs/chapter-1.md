---
title: 'Chapter 1: Downloading genetics data'
output:
  pdf_document:
    toc: true
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    toc_collapse: true
---

This chapter will give an overview of genetics data. We will cover:
1. Genetics background (optional). In this section a brief background of genetics will be given, to give context to the problem, but it is optional, as relevant domain knowledge will also be inserted throughout the other chapters.
2. Downloading genetics data. In this section, code is presented to readily download genetics data.
3. Processing the data. Step-by-step instructions are given on how the data processing pipeline works. 

The aim of this chapter is to provide useful code that can be easily adapted and expanded upon to download different genetics datasets. In our model process, we are interested in three objects
1. The input data matrix, X. This is the data used to form inferences about the genes and will be gene expression data.
2. The response vector, y. This will contain the disease state of a patient (1 if the patient has the disease and 0 otherwise).
3. The grouping structure indexes. This will contain group indexes for which groups (pathways) the genes belong to. This is only needed for the models which use grouping information (glasso and SGL).

Some questions to consider when reading this chapter are:

- Why do we need the various processing steps? Are these steps needed each time or can we sometimes skip some of them?
- Are there are steps that have been omitted that may be important?
- Is this pipeline applicable beyond genetics data?

## Genetics background (optional)
There are 46 chromosomes within the nucleus of a human cell. Two copies of the 22 autosome chromosomes and the sex chromosomes XX and XY make up these 46. Each chromosome is made up of a deoxyribonucleic acid (DNA) molecule. The DNA is a double-stranded molecule which encodes genetic information about an individual. The figure below shows a visualisation of these concepts.

![An illustration of a chromosome, the DNA, and genes [R1].](assets/images/genes_diagram.png)
	
DNA is made up of bonds between base pairs of nucleotides. There are four types of nucleotides found in DNA: adenine (A), cytosine (C), guanine (G), and thymine (T); shown in the figure below.

![An illustration of nucleotides [R2].](assets/images/nucleotides.jpg)

*Genes* are regions of the DNA which act as instructions to make proteins. They also provide individuals with inherited characteristics. The section with letters in the figure above shows an example gene.
	
The bonds between the nucleotides hold the DNA strand together in the form of a double helix. *Single-nucleotide polymorphisms* (SNPs) are common variations of the DNA observed in individuals; shown in the figure below. They are variations of single nucleotides of the genome. 

A *locus* is a fixed position on a chromosome which contains a gene [R3]. A DNA sequence at a locus can have different forms (called alleles) from one copy of the chromosome to another. An individual's genotype is their combination of alleles at a specific locus. 
	
![An illustration of a SNP [R4].](assets/images/snp.png)

Let us consider the SNP shown in figure above with two alleles: A and G. Hence, an individual has four possible genotypes that can be observed: AA, AG, GA, and GG. The genotypes AA and GG are referred to as the *homozygous genotypes*. AG and GA are the *heterozygous genotypes*. The allele which is observed the least in a sample population is termed the *minor allele*. 

Finally, a *phenotype* is an observable trait in an individual. Examples include eye colour, hair colour, or a trait of a disease. There is strong interest in discovering the relationships between genotype and phenotype, as this can enable the attempted prediction of the risk of a disease occurring in an individual, based on their genetic makeup. SNPs are often tested for associations with phenotypes. The outcome variables we use are (case-control) phenotype data.
	
The genetics data described above is called genotype data, which (as mentioned) is often used in modelling to uncover associations (and indeed could be used in our proposed pipeline). However, the specific disease problem we are tackling uses gene expression data. Gene expression is the process by which information from a gene is used to synthesise a functional gene product, such as proteins or non-coding RNA, which in turn influences a phenotype.
Put more simply, gene expression is the basic process through which a genotype results in a phenotype, or observable trait. By measursing the strength of the gene expressions in an individual (which is what our data matrix X will contain) we can then discover associations between the genes (providing the information for the gene expression) and a disease outcome (our response y).

A final topic of interest is the grouping structure of genes. Genes are often grouped into pathways, which are sets of genes that work together to perform a specific function. This grouping information can be useful in modelling, as it can help to identify which pathways are associated with a disease and can allow a model to utilise grouping information about the genes. Some of the models we will use will use this grouping information.

## Downloading genetics data
First, we load a few useful packages:
```{r}
library(GEOquery)
library(GSA)
library(Biobase)
```
To download the datasets, we can use the helpful `getGEO` function from the `GEOquery` package loaded above. This function is able to download a GEO object (which will contain the data needed for fitting models) from the National Center for Biotechnology Information (NCBI) database [L1]. 
The NCBI database contains many different biomedical and genomic datasets. We are interested in the Gene Expression Omnibus (GEO) datasets, which can be found through the search function, although websites exist which have indexed the datasets [L2]. 
In our example, we want to investigate whether we can predict cases of inflammatory bowel diseases (Ulcerative colitis and Crohn's). Searching the NCBI website for 'Ulcerative colitis and Crohn's disease comparison' yields the dataset [L3]. Using the dataset ID, we can download the dataset using
```{r}
raw_data <- getGEO('GDS1615')
```
This is an object of class `GDS`, so it does not behave as a normal R object would. To make it easier to manipulate the object, we convert it to a `eSet` class (ExpressionSet, which is a popular Biobase object)
```{r}
eset_data <- GDS2eSet(raw_data, do.log2 = TRUE)
```
We also used this opportunity to log transform the data. A Log2 transformation is commonly applied to gene expression data to stabilize the variance across the range of expression values and to make the data more normally distributed, which is a useful property in statistical modelling.

Now we can see that this object behaves much more like a traditional R object, with three elements: `sample`, `disease.state`, `description`. Let's see what these are:
```{r}
head(eset_data$sample)
```
This contains the sample ID for the patients, which we will not be using.
```{r}
table(eset_data$disease.state)
```
As the name suggests, this contains the disease state of the patients. You can see that there are three levels, with 42 control patients and 85 patients suffering from inflammatory bowel disease.
This is our response variable, y, which we will need to encode later to 0 and 1.
```{r}
head(eset_data$description)
```
This contains the description of the samples, which we will also not be using.

Not every genetics dataset downloaded from NCBI will have the same objects.

**Q1: download the genetics dataset with the ID `GDS807`. Do you get the same objects?** 

## Processing pipeline
Next, we need to process the data to prepare it for model fitting. As mentioned in the introduction to this section, we want to extract three objects:
1. The input data matrix, X.
2. The response vector, y.
3. The grouping structure indexes.

### Data matrix
The data matrix is extracted using the `exprs` function from the `Biobase` package, which extracts the gene expression data. Notice that this was not an object we saw from the `eset_data` object. 
This demonstrates how working with genetics data in R is not always straightforward. The data is often stored in complex objects, and it is necessary to understand the structure of the object to extract the data needed for analysis.
```{r}
X <- t(exprs(eset_data))
```
We also transpose the matrix to get it into the traditional regression format (rows x columns), given by 
```{r}
dim(X)
```
where we see that we have 60 samples (rows) and 22575 genes (columns). We can see a snippet of the data
```{r}
X[1:5,1:5]
```
which shows the first 5 samples and the first 5 genes. We need to check for missingness, as penalised regression models do not work with missing data (which we would need to remove).
There are many ways to check for missingness - one great way is to use the `aggr` function from the `VIM` package (see Q2). In our case, we can simply check the total amount of missing values
```{r}
sum(is.na(X))
```
In this case, we are fortunate that the data has no missingness and is already well-processed, but sometimes this is not the case (as can be seen in Q2).

Our data matrix is now ready.

**Q2: create the data matrix for the dataset `GDS807`**

### Response
We already extracted the response above, but we now assign it to a new variable name and encode it to 0 and 1. We can do this simply by making the output numerical (1,2,3).
To see how R is assigning each label to a number, we can check
```{r}
levels(pData(eset_data)$disease.state)
unique(as.numeric(pData(eset_data)$disease.state))
```
We can see that 2 identifies control patients.
Therefore, we can assign 2 as FALSE
```{r}
y <- (as.numeric(pData(eset_data)$disease.state) != 2)
```
The encoding of TRUE/FALSE is equivalent to a 0,1 encoding. Our response is now ready.

### Grouping structure index
This is probably the most challenging part of the processing pipeline, as we need to group the genes into their pathways. 

To group genes into their pathways, we use the major collections of gene sets of the Human Molecular Signatures Database (MSigDB) [L4]. 
There are nine such collections, C1-C8 and H, each containing pathways that perform different tasks. For this dataset, we will use the C3 pathway, which are the regulatory target gene sets (gene sets representing potential targets of regulation by transcription factors or microRNAs).

We can download the gene set (the .gmt file) from the MSigDB [L4] website. The gene sets are updated regularly and an account is needed to download them. 
For those reasons, the C3 .gmt file has been provided in the GitHub repository (`data/gene-sets`). 

We load the file using
```{r, results='hide'}
geneset_data <- GSA.read.gmt("data/gene-sets/c3.all.v2023.2.Hs.symbols.gmt")
```
This contains three elements: 
1. `genesets`: this is a list, where each list entry is the the names of the genes for a particular gene set.
2. `geneset.names`: the names of the gene sets.
3. `geneset.description`: links which give descriptions of each gene set.

We need to match the genes in X to the genes in the gene sets in the `geneset_data` object. 
We now match the genes to the gene sets. To do this, we use the `match` R function. 
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
length(ind_include)
```
We can see that about 12000 genes have been matched, just over half of the genes present in X. We now reconstruct X with only the matched genes

```{r}
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
We are almost done. The final step is to reset the group indexing so that it runs from 1,2,...
Many of the models we will use do not work unless this is done. We reset the indexing by looping over each gene set and assigning the group index to the genes in the gene set.
```{r}
grp_ids = data.frame(col_id = 1:length(unique(group_index)),grps=unique(group_index))
ordered_group_indx = rep(0,length(group_index))
for (i in 1:length(group_index)){
  ordered_group_indx[i] = which(grp_ids$grps == group_index[i])
}
```

### Save dataset
We can now save the cleaned dataset (we will save it as an RDS file)
```{r}
saveRDS(list(X,y,ordered_group_indx),"colitis-data-c3.RDS")
```

**Q3: run the full pipeline with the dataset `GDS807`**

## Links
- [L1] https://www.ncbi.nlm.nih.gov/gds
- [L2] https://cola-gds.github.io/
- [L3] https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS1615
- [L4] https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp

## References
- [R1] N. E. U. Hermansen. Prognostic Studies in Multiple Myeloma with Emphasis on Genetic
Tumor Lesions. PhD thesis, 2016. doi: 10.13140/RG.2.2.29522.96966.
- [R2] G. Betts, P. DeSaix, E. Johnson, J. E. Johnson, O. Korol, D. H. Kruse, B. Poe, J. A.
Wise, and K. A. Young. The Nucleus and DNA Replication, 2012. URL http:
//philschatz.com/anatomy-book/contents/m46073.html.
- [R3] Sano Genetics. SNP of the week, 2019. URL https://medium.com/@sanogenetics/
snp-of-the-week-58e23927c188.