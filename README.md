# Weld Quality Prediction

## Project Context

This project is carried out by a team of three students as part of the Machine Learning course (3A, Centrale).  
The goal is to **predict the quality of welds on steel materials**.

Welding quality is a major industrial concern, for example in the production of wind turbine pipelines. Currently, expert knowledge on weld quality is transferred mainly from experienced welder to expert, making the process highly dependent on human expertise. Our project aims to **extract and standardize expert knowledge using data** and to **discover new patterns through data exploration**.

---

## Team Members

| Name | Email |
|------|-------|
| Martin Danieau | martin.danieau@student-cs.fr |
| Jimmy Guertin | jimmy.guertin@student-cs.fr |
| Marceau Guittard | marceau.guittard@student-cs.fr |


**Date:** November 12, 2025  

---

## Data Source

The public dataset is available here:  
[Welddb Dataset](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html)  

We use **Pandas** for data manipulation and preprocessing. Other libraries include **numpy, seaborn, matplotlib, scikit-learn, and XGBoost** for analysis and modeling.  

### Reference
Cool, R.L. (2012). *The Weld Database: Material and Welding Properties for Predictive Modeling*. [PhD Thesis](https://www.phase-trans.msm.cam.ac.uk/2012/Cool_Thesis.pdf)  
Junak, G.; Adamiec, J.; Łyczkowska, K. (2024). Mechanical Properties of P91 Steel (X10CrMoVNb9-1) during Simulated Operation in a Hydrogen-Containing Environment. Materials, 17(17), 4398. [Paper](https://www.mdpi.com/1996-1944/17/17/4398)  
Duan, P.; Liu, Z.; Li, B.; Li, J.; Tao, X. (2020). Study on microstructure and mechanical properties of P92 steel after high-temperature long-term aging at 650 °C. High Temperature Materials and Processes, 39(1), 545-555. [Article](https://doi.org/10.1515/htmp-2020-0087)  
BEBON Steel. (n.d.). 12Cr1MoV steel plate features. [Article](https://www.steel-plate-china.com/news/1000010000001006.html)

---

## Project Objectives

1. **Exploratory Data Analysis (EDA) and Preprocessing**
   - Handle missing values, convert string inequalities (`<x`) to numeric, and normalize variables when necessary.   
   - Organize Python code into separate modules: `preprocessing.py`, `train.py`, `plot.py`.

2. **Feature Selection and Understanding**
   - Identify variables most relevant to weld quality.  
   - Implement targeted imputation strategies:
     - Sulphur and Phosphorus concentrations → IterativeImputer (mean strategy)  
     - Other concentrations → fill with 0  
     - Other numeric features → IterativeImputer (median strategy)  
     - One-hot encoded features → fill missing values with 0

3. **Machine Learning Modeling**
   - Supervised models: `XGBRegressor`, `RandomForestRegressor`, etc.  
   - Robust **cross-validation** to assess model performance.  
   - Semi-supervised approach using pseudo-labeling for unlabeled targets with confidence filtering based on prediction variance.

4. **Performance Comparison**
   - Metrics: RMSE, relative RMSE (% of mean target), correlation coefficient.  
   - Compare strategies for handling inequalities and different regressors.

5. **Conclusions and Recommendations**
   - Identify the most effective approach for weld quality prediction.  
   - Provide practical recommendations for improving weld quality.


