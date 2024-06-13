# Set up
global_dir = '' # add the path for where the recode folder sits

#source("http://www.bioconductor.org/biocLite.R")
#biocLite("GEOquery")
#library(Biobase)
library(GEOquery)
library(GSA)

# Extract the dataset
gds1615 <- getGEO('GDS1615')

# Process the data
eset <- GDS2eSet(gds1615, do.log2 = TRUE)

# Construct the response vector and design matrix
y <- as.numeric(pData(eset)$disease.state) == 2
X <- t(exprs(eset))

# Get gene identifiers
Gene.Identifiers <- Table(gds1615)[,2]

# Set working directory to location of gene sets
setwd(paste0(global_dir,"ReCoDE-penalised-reg/data/gene-sets"))

# Load the gene set data
filename <- "c3.all.v2023.2.Hs.symbols.gmt"
import_data <- GSA.read.gmt(filename)

# Initialize index vector
num_bands <- length(import_data$genesets)
index <- rep(0, length(Gene.Identifiers))

# Map gene identifiers to gene sets
for(i in 1:num_bands) {
  indi <- match(import_data$genesets[[i]], Gene.Identifiers)
  index[indi] <- i
}

# Filter and create final data matrices
ind.include <- which(index != 0)
genenames <- Gene.Identifiers[ind.include]
X <- X[, ind.include]
group_index <- rep(0, ncol(X))

# Assign group indices to the genes
for(i in 1:num_bands) {
  for(j in 1:length(import_data$genesets[[i]])) {
    change.ind <- match(import_data$genesets[[i]][j], genenames)
    if(!is.na(change.ind) && group_index[change.ind] == 0) {
      group_index[change.ind] <- i
    }
  }
}

# Create group ID dataframe and new group indices
grp_ids <- data.frame(col_id = 1:length(unique(group_index)), grps = unique(group_index))
new_group_indx <- sapply(group_index, function(x) which(grp_ids$grps == x))

# Prepare final data
final_data <- list(x = X, y = y)

# Save final data
saveRDS(list(final_data, new_group_indx), "colitis-data.RDS")
