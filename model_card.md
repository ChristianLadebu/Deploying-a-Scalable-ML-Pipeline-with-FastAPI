# Model Card
 
## Model Details
This model is a binary classification model trained to predict whether an individual’s income exceeds $50,000 per year based on demographic and employment-related features from U.S. Census data.

## Intended Use
The model is intended for educational purposes to demonstrate how to build, train, evaluate, and deploy a machine learning pipeline using best practices, including data preprocessing, model evaluation, and fairness assessment.

## Training Data
The training data comes from the U.S. Census Income dataset (also known as the Adult dataset). The dataset contains demographic attributes such as age, education level, occupation, marital status, race, sex, and native country.

The target variable is whether an individual earns more than $50,000 annually.

## Evaluation Data
The evaluation data consists of the 20% holdout test set that was not used during training. This dataset is processed using the same fitted preprocessing pipeline (encoder and label binarizer) as the training data.

Additionally, model performance was evaluated on data slices corresponding to each unique value of the categorical features to assess fairness and performance consistency.

## Metrics
The model was evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

On the test dataset, the model achieved the following performance:

- Precision: **0.7278**
- Recall: **0.6001**
- F1 Score: **0.6578**

Performance metrics were also computed on slices of the data for each categorical feature (e.g., education level, race, sex). These slice-based evaluations are saved in `slice_output.txt` and were used to identify performance disparities across subgroups.

## Ethical Considerations
This model is trained on historical census data, which may reflect existing societal biases related to income, race, gender, and occupation. As a result, the model may reproduce or amplify these biases when making predictions.

Predictions from this model should not be used to make decisions that affect individuals’ livelihoods or opportunities. Care should be taken to assess fairness and bias before any real-world application.

## Caveats and Recommendations
The model’s performance varies across different demographic groups, as shown in the slice-based evaluation results. Some groups have fewer data points, which may result in less reliable performance metrics.

Future improvements could include:
- Using more advanced models
- Applying feature scaling or alternative preprocessing
- Expanding evaluation to include additional metrics

