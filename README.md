## Team Caesar – Home Loan Default Risk Prediction 
  
  
### Project Description:
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders. 

Home Credit strives to broaden financial inclusion for the unbanked population by supplying a positive and safe borrowing experience. To make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities. 

Tags: Tabular Data  
  
  
### Quick Links:
- [Project Rules and Description of Projects](https://docs.google.com/document/d/12uzoeAaKx6GcpCnKcnbjeYEB-_U1J5ddhh8_xT6C4c4/edit)
- [Kaggle Link for Dataset](https://www.kaggle.com/competitions/ai511-homeloan-2022)
  
  
### Allocated TA:
**Name -** Mr. Vijay Jaiankar  
**Email address -** vijay.jaisankar@iiitb.ac.in  
  
### Team Members: 
1. Member 1 (Point of Contact):  
   1. **Name -** Dhruv Awasthi
   2. **Roll number -** MS2022010
   3. **Email address -** dhruv.awasthi@iiitb.ac.in

2. Member 2:
   1. **Name -** Ashok Senapati
   2. **Roll number -** MS2022004 
   3. **Email address -** ashok.senapati@iiitb.ac.in
  
  
### Dataset:
- [Dataset link](https://www.kaggle.com/competitions/ai511-homeloan-2022/data)
- Dataset description:  
  - **train_data.csv -** the training set
  - **test_data.csv -** the test set
  - **sample_solution.csv -** a sample submission file in the correct format
  - **columns_description.csv -** supplemental information about the data; read this to know more about the columns in `train_data.csv`.

- Evaluation:
  - Submissions are evaluated on `TARGET` column using the Macro F Score. 
  - For each `SK_ID_CURR` in the test set, you must predict the `TARGET` i.e predict if the customer with that ID will default their loan or not.

### Model - Stacked Architecture:
- We have built a stacked architecture of estimators using:
  - HistGradientBoostingClassifier
  - AdaBoostClassifier
  - GradientBoostingClassifier
  - LogisticRegression (final estimator)
- It is a method for combining the estimators to reduce the biases.
- The predictions of each individual estimator are stacked together and used as input to a final estimator to compute the prediction.
- The final estimator is trained through cross-validation.  
  

![Model - Stacked Architecture](../docs/model_architecture.png)


### Project Checkpoint Presentations:
You can view our presentations for each of the three project checkpoints in `docs/` 
directory that summarises the work we did for each checkpoint.

### Impactful Notebooks:
Checkout the following notebooks in `notebooks/` directory:
1. **2022_11_09_reduce_dataset_size.ipynb**: how to reduce the dataset size for reducing the space complexity and speed up the training.
2. **2022_12_08_master_eda.ipynb**:  notebook for the final preprocessing steps on training data.  

### How to Run Code:
**1. Create a new environment:**  
It is always a great idea to create new environment for a new project, so you  
don't accidentally mess up with other projects that you are working on and 
requires different version of packages. The steps to create and activate 
environment using two most popular tools are:  

**- virtualenv**  
```
virtualenv ai511_homeloan_2022
source ai511_homeloan_2022/bin/activate
```  
To deactivate type:
```
deactivate
```
**- conda**  
```
conda create -n ai511_homeloan_2022
conda activate ai511_homeloan_2022
```
To deactivate type:
```
conda deactivate
```  

**2. Install the packages:**  
To install all the packages required for this project:
```
pip install -r requirements.txt
```  

**3. Run the project:**  
Checkout the branch `eda`:  
```
git checkout eda
```  

To run the entire project, run the following command: 
```
python main.py --train_and_test_model=True
```
  
Once done, you can check the `submission.csv` file that contains the predictions in the`data/dataset/` directory.

### Author:  
- [Dhruv Awasthi](https://www.linkedin.com/in/dhruv-awasthi/)
- [GitHub profile link](https://github.com/DhruvAwasthi)
- **Email:** dhruvawasthicc@gmail.com