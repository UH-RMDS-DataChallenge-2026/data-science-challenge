**Town A Flu Vaccine Allocation
Group B · UH–RMDS Data Science Challenge 2026**

How should Town A distribute seasonal flu vaccines during a simulated H3N2 outbreak? We estimate each resident’s probability of accepting a vaccine and combine it with their estimated hospitalisation risk to decide who should receive one. The objective is to minimise the town’s expected cost per resident.
Challenge and decision rule
The challenge assumes that every resident becomes infected. Sending a vaccine costs $300. If a resident would accept a vaccine but receives none, the town pays them $300 in cash. Each hospitalisation costs $18,500.

The notebook estimates unvaccinated hospitalisation risk from age and the number of chronic conditions. Its base scenario assumes 23% vaccine effectiveness against hospitalisation and tests alternative assumptions in a sensitivity analysis. Under the cost model, we send a vaccine when the predicted probability of acceptance exceeds the resident’s threshold:
threshold = 300 / (300 + 18,500 × hospitalisation risk × vaccine effectiveness)
Higher estimated hospitalisation risk lowers the acceptance probability needed to justify sending a vaccine.
Data
File	Purpose
dataset_A_training.csv	Labelled survey data. The seasonal_vaccine target represents vaccine acceptance.
dataset_A_testing.csv	Unlabelled respondents for whom the notebook makes allocation decisions.
dataset_A_example_submission.csv	Required column names and respondent order for submission.


The notebook tries to load the CSVs from the challenge repository. It falls back to files in the working directory or prompts for uploads in Colab.
Modelling and evaluation
1. Inspect data quality, missing values, and patterns in vaccine acceptance. Engineer features from demographic information, missingness, and survey responses. Fit imputation, encoding, and any scaling inside model pipelines.
2. Compare logistic regression, decision tree, random forest, extra trees, histogram gradient boosting, XGBoost, and LightGBM. Use randomised search with log loss to tune XGBoost and histogram gradient boosting.
3. Use stratified five-fold cross-validation to compare a fixed 0.5 threshold, a global cost-based threshold, and a resident-specific risk-based threshold. Report mean cost per resident alongside metrics such as ROC AUC, log loss, and accuracy.
4. Compare costs with sending to nobody, sending to everybody, and a perfect-knowledge lower bound. Test alternative risk and effectiveness assumptions. Inspect model behaviour using calibration plots, permutation importance, SHAP, and partial dependence.
Running the notebook
1. Open [GroupB_Data_Science_Challenge_FINAL (1).ipynb](GroupB_Data_Science_Challenge_FINAL%20%281%29.ipynb) in Google Colab and select Runtime → Run all.
2. If prompted, upload the three challenge CSVs using the filenames listed above. The setup cell installs XGBoost, LightGBM, and SHAP.
3. Check Section 8 for model rankings and diagnostics, Section 9 for sensitivity analysis, and Section 10 for the five submission files. Section 12 summarises the estimated savings.
For local use, open the notebook in Jupyter, place the CSVs in the working directory if the remote files are unavailable, and ensure NumPy, pandas, scikit-learn, Matplotlib, and seaborn are installed.
Submission files
The notebook ranks the best model-and-threshold combination from each family by cross-validated cost. It refits the top five distinct model families on the full training set and exports challenge_submission_group_B_order_1.csv through challenge_submission_group_B_order_5.csv, with order 1 representing the lowest estimated cross-validation cost.

Each CSV contains respondent_id and seasonal_vaccine, in the order specified by the example submission. In the submission, seasonal_vaccine is the allocation decision: 1 means send a vaccine and 0 means do not send one.
Interpreting the results
The notebook’s summary describes lower estimated costs for the resident-specific threshold than for a fixed 0.5 cutoff. Run the notebook to reproduce the rankings and exact costs. The estimates depend on the hospitalisation and vaccine effectiveness assumptions. The notebook selects models and tunes hyperparameters before its 80/20 check, so that check uses data already involved in model selection.
