# Using CNN to Classify Alzheimer's
![alt text](/notebooks/report/figures/read_mepic.png)

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
    └──  mymods.py 
    

```
#### Repo Navigation Links 
 - [Final summary notebook](https://github.com/howen7/Alzeimers/tree/main/notebooks/report/FinalNotebook.ipynb)
 - [Exploratory notebooks folder](https://github.com/howen7/Alzeimers/tree/main/notebooks/exploratory)
 - [src folder](https://github.com/howen7/Alzeimers/tree/main/src)
 - [Presentation.pdf](https://github.com/howen7/Alzeimers/tree/main/reports)
 
# General Setup Instructions 

Ensure that you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/).

### `alz-env` conda Environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `alz-env` conda environment. To do so, please run the following commands *in your terminal*:
```bash
# create the alz-env conda environment
conda env create -f environment.yml
# activate the conda environment
conda activate alz-env
# if needed, make alz-env available to you as a kernel in jupyter
python -m ipykernel install --user --name alz-env --display-name "Python 3 (alz-env)"
```
# Context:

The objective of this project was to create a CNN that could predict 4 different classifications of Alzheimer's with an accuracy over 80% and AUC score above 0.9. Alzheimer’s is a progressive disease that destroys memory and other important mental functions. More than 5 million americans have alzheimer's, and one in ten poeple over the age of 65 have it. It can be one of the most heartbraking diseases as you have to watch the memory of loved ones dissapear. As brain scans and treatments for Alzheimers get better its important that there are models that can help MRI technicians and doctors diagnose patients correctly.

![alt text](/notebooks/report/figures/class_images.png)

# Aims:

This project aims to:<br>

- Investigate what features the model is picking up on when its classifying these images<br>
- Classify degree of Alzheimer with high accuracy and AUC.<br>
- Create an online version using flask that will allow users to choose a MRI image and return a classifcation of it<br>
   

# Data:

This project uses dataset from Kaggle found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images).<br>
The data set compromises of 4 stages of alzeimers labeled: Normal brain, Very Mild Demented, Mild Demented, Moderate Demented

There was a significant class imbalance on the Moderate Alzheimer's class which is why the use of AUC score was important. 

Normal: 3200<br>
Very Mild: 2240<br>
Mild: 896<br>
Moderate: 64<br>


### Reproducibility
To set up the data into train, validation, and test folders so you can repoduce my results go to [Import_data notebook](https://github.com/howen7/Alzeimers/tree/main/notebooks/exploratory/Import_data.ipynb) and run all the code.

# Models used + Methodology:

Earlier iterations of models can be found in exploratory notebooks. I tried a variety of models with test size (64,64), (128,128), and finally (178, 208). Using a size of (178,208) took longer to train, but would almost always have better results which is why I choose it. For the batch size I didn’t want too large of a batch size because they have a chance at getting stuck in local minimums and thus leads to poorer generalizations. I didn’t want to use too small of a batch size because the computational time was to significant. On the simpler models in my exploratory notebook I stuck to batch size less than 64, but once I started getting more advance models I used 64 and was able to achieve great results without the cost of extra computational time. For the imbalanced classes I decided to use class weights instead of data augmentation during my training as I was getting better results and not using a ton of extra computational time for data augmentation. The image was rescaled to normalize the pixels between 0 and 1. Several Learning rate

![alt text](/notebooks/report/figures/layers.png)

# Results:
The CNN was able to classify with a very high accuracy and AUC and below are the results. 
![alt text](/notebooks/report/figures/Confusion_matrix_test.png)
![alt text](/notebooks/report/figures/acc.png)
![alt text](/notebooks/report/figures/auc.png)
![alt text](/notebooks/report/figures/Lime_correct_preds.png)

### Best model: 6 Layer CNN

![alt text](/notebooks/report/figures/ModelSummary.png)

### Future Investigations and Recommendations:

It would be interesting to see this done with positron emission tomography (PET). Patients use fluoro-deoxy-d-glucose (FDG) and amyloid tracers such as Pittsburgh Compound-B (PiB) to study the cerebral metabolism. This is a relativly new imaging technique that can provide alzhiemer's pathphysciogical processes and help diagnose stages. If I can eventually find a data set that has enough PET images of the brain I would like to create another CNN. 
