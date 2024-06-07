<!-- Your Project title, make it sound catchy! -->

# Predicting colitis using penalised regression

<!-- Provide a short description to your project -->

## Timeline
1.	Document genetics data. This step will involve documenting the various genetic data sources, how the data is accessed and the licenses for each dataset.  
2.	Download data. This step will be writing code for the downloading and processing of the genetics data. This will involve several cleaning and checking steps to ensure the data is suitable to be used for model fitting.
3.	Model background. This step is documenting and explaining the basics behind the models that are being used, so will involve basic mathematical and optimisation theory.
4.	Initial model fitting. This will involve applying basic models to the genetics data to get a feel of what works well, how the fitting can be improved, and investigating signs of deficiencies in the models.
5.	Better models. This step involves describing how the models used so far can be improved and options for more advanced models. These models will then be implemented in R.
6.	Prediction: This step will use all the models discussed so far to form predictions on the colitis data and evaluate the accuracy of the predictions.

## Description

The field of predictive medicine is becoming increasingly popular. A key challenge is dealing with the high-dimensionality of genetics, where the number of genes (factors) is far larger than the number of patients (observations), resulting in classical statistical methods breaking down. This has lead to the rise of penalised regression, where a penalty is applied to the factors to induce sparsity, so that a large majority of factors are deemed irrelevant, allowing for statistical inference to occur. One popular penalised approach, covered in most undergraduate and graduate statistics modules, is the lasso. The lasso has gained a large amount of traction over the last 20 years, with many extensions also proposed. 

The lasso has found particular use in genetics, as it is computationally efficient and is able to select relevant genes as being associated with a disease. One particular useful extension is the group lasso, which can apply this penalisation onto groups of variables. As genes are naturally found in groups (pathways), this extension has also found extensive use in genetics. One final approach is the sparse-group lasso, which combines the two. The project would show how to apply these three methods to predicting whether a patient has colitis. 


<!-- What should the students going through your exemplar learn -->

## Learning Outcomes

- Provide an insight into how genetics data can be downloaded, cleaned, and prepared for use in analysis – this gives a good insight into how general R data manipulation works.
-	Give the student an introduction into predictive modelling by showing how a fitted model can be used to form predictions.
-	Demonstrate how the wide class of penalised regression models work and how they can be implemented in R. This will provide mathematical and statistical background to how these methods work, which will touch upon important regression topics that form the foundation for most models the students will see further down their academic paths.


<!-- How long should they spend reading and practising using your Code.
Provide your best estimate -->

| Task       | Time    |
| ---------- | ------- |
| Reading    | 3 hours |
| Practising | 3 hours |

## Requirements

<!--
If your exemplar requires students to have a background knowledge of something
especially this is the place to mention that.

List any resources you would recommend to get the students started.

If there is an existing exemplar in the ReCoDE repositories link to that.
-->

### Academic

<!-- List the system requirements and how to obtain them, that can be as simple
as adding a hyperlink to as detailed as writting step-by-step instructions.
How detailed the instructions should be will vary on a case-by-case basis.

Here are some examples:

- 50 GB of disk space to hold Dataset X
- Anaconda
- Python 3.11 or newer
- Access to the HPC
- PETSc v3.16
- gfortran compiler
- Paraview
-->

### System

<!-- Instructions on how the student should start going through the exemplar.

Structure this section as you see fit but try to be clear, concise and accurate
when writing your instructions.

For example:
Start by watching the introduction video,
then study Jupyter notebooks 1-3 in the `intro` folder
and attempt to complete exercise 1a and 1b.

Once done, start going through through the PDF in the `main` folder.
By the end of it you should be able to solve exercises 2 to 4.

A final exercise can be found in the `final` folder.

Solutions to the above can be found in `solutions`.
-->

## Getting Started

<!-- An overview of the files and folder in the exemplar.
Not all files and directories need to be listed, just the important
sections of your project, like the learning material, the code, the tests, etc.

A good starting point is using the command `tree` in a terminal(Unix),
copying its output and then removing the unimportant parts.

You can use ellipsis (...) to suggest that there are more files or folders
in a tree node.

-->

## Project Structure

```log
.
├── examples
│   ├── ex1
│   └── ex2
├── src
|   ├── file1.py
|   ├── file2.cpp
|   ├── ...
│   └── data
├── app
├── docs
├── main
└── test
```

<!-- Change this to your License. Make sure you have added the file on GitHub -->

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
