# CNN-LSTM-stock-trading-model
Long-only trading strategy using CNN-LSTM networks

<b> 1. Problem Statement </b>

Using AMD daily stock price data from the past 6 years (from 2017-01-01 and 2024-03-01) to produce a CNN-LSTM based neural network model that can predict daily up-moves of
the AMD stock.

<b> 2. Target Variable </b>

Since I am building a model to predict up-moves, I am labeling an up-move as 1, and 0 otherwise. To classify a data point as either 1 or 0, I have set up a threshold level. The threshold level that I set up is going to be a dynamic rather than a static threshold. The dynamic threshold level is based on the ATR (Average True Range) value of the stock price. More specifically, it is 0.1 times the ATR value, where I compute the ATR value with a lookback period of 14.

Having a dynamic threshold based on ATR would mean that in periods of high volatility, the threshold level is going to be higher compared to periods of low volatility. The reason why this is desirable is
because,
- In a low volatility environment, if a fixed threshold level is used, there will be very few up-move signals. Hence, one could miss out on trading opportunities. Whereas if a dynamic threshold is used, the threshold level is going to be lower in periods of low volatility. Therefore, meaningful up-moves in a low volatility environment are going to be captured.
- In a high volatility environment, if a fixed threshold is used, there are going to be a lot of up move signals which could lead to some false signals because non-meaningful up-moves (given the environment) are classified as 1 instead of 0.

<b> 3. Feature Engineering </b>

The Python library called Pandas TA has been used to generate the features. The features can be classified into the following classes: Momentum Technical Indicators (TI), volume TI, volatility TI, trend TI, and performance TI.

<b> 4. Exploratory Data Analysis (EDA) </b>

<i> Data Type </i>

All of the features are either floats or integers, so there is no need to perform any transformations, such as one-hot-encoding.

<i> Missing Values </i>

After feature engineering was performed, there were missing values in the dataset. In
order to handle these missing values, the following sequence is performed:
1. Remove the last row of the dataset, as it contains a null value that appears when the target
variable is generated.
2. Perform forward fill to address the missing values located in the middle section of the dataset
3. To address the missing values located in the early portion of the time-series dataset (which is caused by the use of a lookback window when computing the technical indicators), I am removing those data points with missing values.

<i> Class Imbalance </i>

As mentioned before, each data point is classified as either 1 or 0. It turns out that 763 of the data points are classified as 1, and 965 data points are classified as 0. Therefore, we have an imbalanced dataset.

This class imbalance needs to be dealt with so that the model is not biased in favor of the majority class. Therefore, class weights are going to be assigned to the data points in each class, such that the dataset becomes balanced. The class weights are 1.0 and 1.265 for class 0 and class 1, respectively (i.e. 965 * 1 - 763 * 1.265 = 0).

<b> 5. Training, Validation, and Testing Dataset </b>

The original dataset is now going to be split-up into a training, validation, and testing dataset. The first 60% of the original dataset is used as the testing dataset. The next 20% of the dataset is used as the validation dataset, and the remaining dataset is used as the testing dataset.

This actually gives rise to data leakage. The list below discusses the 2 sources of data leakage and how they can be addressed:
- Computing the features requires the use of data points from a few periods in the past. Hence, I need to drop the first few data points at the beginning of the validation and testing dataset,
such that there is no overlap between the 3 datasets.
- Computing the target variable requires the use of a data point from one period in the future.
Hence, I need to drop the last row of the training and validation dataset.

<b> 6. EDA Part 2</b>

<i> Checking for Outliers </i>

To check for the presence of outliers in our features, the Tukey's Fences rule will be used. Note that the main purpose of checking for outliers, in this case, is to select the appropriate scaling method for the features. 

<i> Checking for normality </i>

To check whether a feature is normally distributed or not, the Shapiro test will be used. Similarly, in this case, the main purpose of checking for normality is to select the appropriate scaling method for the features.

<i> Analyze multicollinearity </i>

Multicollinearity exists when there is a strong correlation between the features in a dataset. A measure called VIF will be used to quantify the level of multicollinearity. (Since the features are all generated from the same price and volume data, one would expect a high level of multicollinearity in this dataset.)

<b> 7. Feature Scaling </b>

Each feature is going to be scaled using 1 out of the 3 different types of scaling: Standard Scaling (for normally distributed features), MinMax Scaling (for non-normal features and no presence of extreme anomalies), and Robust Scaling (for non-normal features with the presence of extreme anomalies).

<b> 8. Feature Selection </b>

In performing feature selection, 3 feature selection methodologies are used: Boruta, RFE (Recursive Feature Elimination), and VIF-based feature selection. 3 different combinations of these methodologies are going to be used to generate 3 different sets of features:
- Boruta + VIF-based feature selection (Boruta + VIF):

I start by implementing Boruta to the original dataset. The reason I start with Boruta is
because Boruta feature selection method is based on the feature importance score; one thing to
note about the feature importance score (when dealing with a dataset that contains
multicollinearity) is that, while the model (i.e. Random Forest) has no preference for one feature over the other features it is correlated with, once one of them is selected in the model, the feature importance score of the other correlated features will decrease because the impurity has been removed by the selected feature (“Selecting Good Features – Part III: Random Forests”, 2014). Hence, those other correlated features become less likely to ‘beat’ the best-performing shadow feature (i.e. those features will not be selected by Boruta), and therefore Boruta will indirectly address multicollinearity. But the key word here is “indirectly” (i.e. its main goal is not to address multicollinearity, but instead, to choose the features with strong predictive power), and so the features selected by Boruta could potentially still exhibit multicollinearity. Therefore, I then proceed to check for the presence of multicollinearity in the features selected by Boruta by computing the VIF score, and if multicollinearity is present (i.e. features with VIF > 5), I would address the multicollinearity by using VIF-based feature selection. Note that the reason one would want to address multicollinearity is because the values of those features that exhibit strong multicollinearity can be predicted by the other features. Therefore, dropping those features will not hurt the performance of the trained model and it will also help to reduce training time.

- RFE with 10 selected features + VIF-based feature selection (RFE_10 + VIF):

I start with RFE until only 10 features remain. Similar to Boruta, since RFE feature selection method is also based on the feature importance score, RFE can indirectly address multicollinearity. But note that the main goal of RFE is to select the most important features, and hence the features selected by RFE could still exhibit multicollinearity. Therefore, after RFE is performed, I then proceed to perform VIF-based feature selection.

- RFE with 20 selected features + VIF-based feature selection (RFE_20 + VIF):

The process is similar to the one above, the only difference is that 20 features are selected using the RFE.

Lastly, there is one thing I want to address. In the EDA section, I used the unscaled dataset for the computation of VIF. This is because VIF measures the linear relationship between the features, and so if I scale the dataset, I could potentially 'disrupt' their linear relationship, and therefore I could arrive at a different VIF score. However, the VIF score that is computed in this section (i.e. during the VIF-based feature selection) is going to be based on the scaled dataset. The reason is that the scaled dataset is used when performing Boruta and RFE, hence to maintain consistency in the feature selection process, I also use the scaled dataset for the VIF-based feature selection. It is important that the scaled dataset is used when performing Boruta and RFE because these techniques are based on the feature importance score, in which according to Strobl, Boulesteix, Zeileis, & Hothorn (2007), the feature importance score becomes unreliable if the features are not scaled

<b> 9. Model Building </b>

A CNN-LSTM network is simply a 1-Dimensional CNN layer/s followed by LSTM layer/s.

Random search is used to perform hyper-parameter tuning. Note that we are tuning 3 models, each with a different dataset, where the first one uses the Boruta + VIF features, the second one uses RFE with 10 selected features + VIF, and the third one uses RFE with 20 selected features + VIF.

<b> 10. Training Tuned Model </b>

The tuned models from the previous section are now going to be trained using their respective training datasets. The training will be performed until the 200th epoch (unless it is stopped early by the early-stopping callback) and the loss function that is used is the Binary Cross Entropy function (since this is a binary classification problem). The early-stopping callback is set to monitor the accuracy score (computed on the validation dataset) and to have a patience of 10 (i.e. if the accuracy score does not improve in the next 10 epochs, the training will be stopped early). This is used to prevent the model from overfitting. Moreover, the Adam optimizer is going to be used to optimize the model’s parameters.

<b> 11. Performance Evaluation </b>

After all of the tuned models have been trained, these models' performances are going to be tested by the testing datasets. And then, based on their predictions, their performances are going to be evaluated using metrics such as the ROC-AUC score, accuracy, recall, precision, and F1-score.

<b> 12. Back-Testing The Best Model </b>

Financial metrics such as monthly returns, cumulative returns, Sharpe ratio, beta, alpha, and drawdowns are computed.

<b> 13. Possible Improvements </b>

- Transaction costs were not considered when setting the threshold level for an up-move. Therefore, transaction costs could be incorporated to ensure that the model is trained to only signal for an up-move when the return is large enough to cover the transaction costs.
- The majority of the derived features are lagging indicators. Leading indicators such as the options price of AMD could be incorporated in the features.
- An ensemble model (either a voting or stacking ensemble) could be created to improve the overall performance of the predictions.


<b> Reference </b>

Selecting Good Features – Part III: Random Forests. (2014, December 1). Diving into Data.
http://blog.datadive.net/selecting-good-features-part-iii-random-forests/

Strobl, C., Boulesteix, A., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable
importance measures: Illustrations, sources and a solution. BMC Bioinformatics, 8(25).
doi:10.1186/1471-2105-8-25










