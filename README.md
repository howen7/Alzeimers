# Using CNN to Classify Alzheimers
![alt text](/notebooks/report/figures/readmepic1.jpg)

# Table of Contents

<!--ts-->
 * [General Setup Instructions](https://github.com/howen7/Alzeimers#general-setup-instructions)
 * [Context of Project](https://github.com/howen7/Alzeimers#Context)
 * [Definitions](https://github.com/howen7/Alzeimersw#Definitions)
 * [Data](https://github.com/howen7/Alzeimers#Data)
 * [Process](https://github.com/howen7/Alzeimersmodels-used--methodology)
 * [Results](https://github.com/howen7/Alzeimers#Results)
 * [Next Steps](https://github.com/howen7/Alzeimers#Future-Investigations-and-Recommendations)
<!--te-->

```
.
├── README.md     
├── environment.yml
├── notebooks
│   ├── exploratory
│   │   ├── 4_models.ipynb
│   │   ├── EDA.ipynb
│   │   ├── Final_model.ipynb
│   │   ├── FSM.ipynb
│   │   ├── Import_data.ipynb
│   │   ├── Model_new_tvt.ipynb
│   │   ├── Original_exploratory
│   │   │   ├── data_setup.ipynb
│   │   │   ├──firstmodel.ipynb
│   │   │   ├──further_models.ipynb
│   │   │   ├──Models_bad_test.ipynb
│   │   │   ├──More_models.ipynb
│   │   │   └──README.md
│   └── report
│       ├── figures
│       │   ├── acc_loss.png
│       │   ├── acc.png
│       │   ├── auc.png
│       │   ├── Brain_diagram.png
│       │   ├── Channels.png
│       │   ├── class_images.png
│       │   ├── Confusion_matrix_test.png
│       │   ├──Confusion_matrix_val.png
│       │   ├──layers.png
│       │   ├──Lime_correct_preds.png
│       │   ├──Lime_wrong_preds.png
│       │   ├──loss.png
│       │   ├──ModelSummary.png
│       │   └──layers.png
│       └── FinalNotebook.ipynb
├── reports
│   └──SlideDeck.pdf
│   
└── src
    ├── data
    ├── best_weights_mod1.hdf5   
    ├── history_mod1.json
    ├── mymods.py
    └── 
    

```
#### Repo Navigation Links 
 - [Final summary notebook](https://github.com/howen7/Alzeimers/tree/main/notebooks/report/FinalNotebook.ipynb)
 - [Exploratory notebooks folder](https://github.com/howen7/Alzeimers/tree/main/notebooks/exploratory)
 - [src folder](https://github.com/howen7/Alzeimers/tree/main/src)
 - [Presentation.pdf](https://github.com/howen7/Alzeimers/tree/main/reports)
 
# General Setup Instructions 

Ensure that you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/) 

### `alz-env` conda Environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `alz-env` conda environment. To do so, please run the following commands *in your terminal*:
```bash
# create the alz-env conda environment
conda env create -f environment.yml
# activate the housing conda environment
conda activate alz-env
# if needed, make alz-env available to you as a kernel in jupyter
python -m ipykernel install --user --name alz-env --display-name "Python 3 (alz-env)"
```
# Context:

This projects goal was to provide to create a convolutional neural network to identify 4 different stages of alzheimer

# Aims:

This project aims to:<br>

- Investigate what the model is picking up on when its classifying these images<br>
- Classify degree of Alzheimer with high accuracy.<br>
- Create an online version using flask that will allow users to choose a MRI image and return a classifcation of it<br>
    
# Definitions:



# Data:

This project uses dataset from Kaggle found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images).<br>
The data set compromises of 4 stages of alzeimers labeled: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented


# Models used + Methodology:

This project Uses CNN to classify:<br>



    
# Results:
![alt text](/notebooks/report/figures/Model_performance.png)


### Best model: 6 Layer CNN



#### Future Investigations and Recommendations:

