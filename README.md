# Predicting colitis using penalised regression

## Description

The field of predictive medicine is becoming increasingly popular. A key challenge is dealing with the high-dimensionality of genetics, where the number of genes (factors) is far larger than the number of patients (observations), resulting in classical statistical methods breaking down. This has lead to the rise of penalised regression, where a penalty is applied to the factors to induce sparsity, so that a large majority of factors are deemed irrelevant, allowing for statistical inference to occur. One popular penalised approach, covered in most undergraduate and graduate statistics modules, is the lasso. The lasso has gained a large amount of traction over the last 20 years, with many extensions also proposed. 

The lasso has found particular use in genetics, as it is computationally efficient and is able to select relevant genes as being associated with a disease. One particular useful extension is the group lasso, which can apply this penalisation onto groups of variables. As genes are naturally found in groups (pathways), this extension has also found extensive use in genetics. One final approach is the sparse-group lasso, which combines the two. This project shows how to apply these three methods to predicting whether a patient has colitis, an inflammatory bowel disease. 

## Learning Outcomes
Main outcomes:

- Go through an end-to-end analysis of how we can use penalised regression models to predict cases of colitis using gene expression data.
- Gain an insight into how genetics data can be downloaded, cleaned, and prepared for use in analysis – this gives a good insight into how general R data manipulation works.
-	Follow the introduction into predictive modelling by showing how a fitted model can be used to form predictions.
-	Understand how the wide class of penalised regression models work and how they can be implemented in R. This will include mathematical and statistical background to how these methods work, which will touch upon important regression topics that form the foundation for most models used widely in academia and industry.


Optional outcomes:

- The example provided is for predicting colitis data. The questions provided in each section will provide guidance through another example, where breast cancer is predicted. Therefore, the questions will form a comprehensive additional analysis of breast cancer data.
- An additional class of models, SLOPE models, are presented in the optional sections of the chapters. SLOPE models are adaptive versions of the models covered in the main outcomes and provide additional insight into the direction of the current research in this area.
  

| Task       | Time    |
| ---------- | ------- |
| Chapter 1  | 2 hours |
| Chapter 2  | 5 hours |
| Chapter 3  | 4 hours |

## Requirements

### Academic

- Basic understanding of mathematical concepts that underpin penalised regression: linear algebra, matrices, geometry.
- Basic statistics knowledge, including linear regression and model fitting.
- Genetics background is not needed. Basic background information is provided in Chapter 1.
- Familiarity with R programming language.

### System

| Program                  | Version                  |
| ------------------------ | ------------------------ |
| R                        | Any                      |

## Getting Started

The chapters are structured to be worked through sequentially. Each chapter contains optional content that adds extra insight into the problem but is not required to solve to the core problem. 

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
