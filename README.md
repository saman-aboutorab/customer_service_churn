# customer_service_churn

In this project, we will build a classification model using the churn_df
dataset for a customer service call data. The features to use will be
\"account_length\" and \"customer_service_calls\". The target,
\"churn\", needs to be a single column with the same number of
observations as the feature data.

We will convert the features and the target variable into NumPy arrays,
create an instance of a KNN classifier, and then fit it to the data.
:::

::: {.cell .markdown}
![ad](vertopal_13aa0871e7df4015bc19eab03b678e69/6a73857a2b84963ad312512f67458dfcc4cfcbea.jpg)
:::

::: {.cell .markdown}
Source:
<https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn>
:::

::: {.cell .code execution_count="22" id="HXIi-o64k6JI"}
``` python
# Import Pandas
import pandas as pd
# Import Numpy
import numpy as np
# Import Matplotlib
import matplotlib.pyplot as plt


# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
# Import the module
from sklearn.model_selection import train_test_split
```
:::

::: {.cell .markdown id="rG2wg3slqZj9"}
## Importing dataset
:::

::: {.cell .code execution_count="2" id="N_gaSQlUm6so"}
``` python
# Read CSV dataset
churn_df = pd.read_csv('datasets/telecom_churn_clean.csv')
```
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="u0_I-NkVnL_M" outputId="e35a8399-fcf7-4dd0-95cc-060b9a50b210"}
``` python
# Print dataset
print(churn_df.head(5))
```

::: {.output .stream .stdout}
       Unnamed: 0  account_length  area_code  international_plan  voice_mail_plan  \
    0           0             128        415                   0                1   
    1           1             107        415                   0                1   
    2           2             137        415                   0                0   
    3           3              84        408                   1                0   
    4           4              75        415                   1                0   

       number_vmail_messages  total_day_minutes  total_day_calls  \
    0                     25              265.1              110   
    1                     26              161.6              123   
    2                      0              243.4              114   
    3                      0              299.4               71   
    4                      0              166.7              113   

       total_day_charge  total_eve_minutes  total_eve_calls  total_eve_charge  \
    0             45.07              197.4               99             16.78   
    1             27.47              195.5              103             16.62   
    2             41.38              121.2              110             10.30   
    3             50.90               61.9               88              5.26   
    4             28.34              148.3              122             12.61   

       total_night_minutes  total_night_calls  total_night_charge  \
    0                244.7                 91               11.01   
    1                254.4                103               11.45   
    2                162.6                104                7.32   
    3                196.9                 89                8.86   
    4                186.9                121                8.41   

       total_intl_minutes  total_intl_calls  total_intl_charge  \
    0                10.0                 3               2.70   
    1                13.7                 3               3.70   
    2                12.2                 5               3.29   
    3                 6.6                 7               1.78   
    4                10.1                 3               2.73   

       customer_service_calls  churn  
    0                       1      0  
    1                       1      0  
    2                       0      0  
    3                       2      0  
    4                       3      0  
:::
:::

::: {.cell .code execution_count="6" id="02O9iHgynPn7"}
``` python
# Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values
```
:::

::: {.cell .markdown id="4JDuOfbhqP8K"}
# k-Nearest Neighbors: Fit
:::

::: {.cell .code execution_count="7" id="oidfXr9kpVWx"}
``` python
# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
```
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":75}" id="y2Z5qJF-pWsN" outputId="ded42e5e-1bdd-4dc1-bcc6-55e1c43329bd"}
``` python
# Fit the classifier to the data
knn.fit(X, y)
```

::: {.output .execute_result execution_count="8"}
```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=6)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=6)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="12" id="W3LIQMjfpj2z"}
``` python
# Create a sample X_test
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
```
:::

::: {.cell .code execution_count="13" id="t58WGprHpYoi"}
``` python
# Predict the labels for the X_new
y_pred = knn.predict(X_new)
```
:::

::: {.cell .markdown id="3M2W1142qLFI"}
# k-Nearest Neighbors: Predict
:::

::: {.cell .code execution_count="14" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="1P_GMMLrpd4f" outputId="cb14813f-50f1-4d83-8da3-de87271d9715"}
``` python
# Print the predictions for X_new
print("Predictions: {}".format(y_pred))
```

::: {.output .stream .stdout}
    Predictions: [0 1 0]
:::
:::

::: {.cell .markdown id="2xul0cwuqFmr"}
## Train/test split + computing accuracy {#traintest-split--computing-accuracy}
:::

::: {.cell .code execution_count="16" id="U31p_TTtp13I"}
``` python
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values
```
:::

::: {.cell .code execution_count="17" id="Sd6xfvhlqE1t"}
``` python
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
```
:::

::: {.cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":75}" id="RJ9DCT3vqvaU" outputId="ceae58f6-2277-4b7b-86d7-20e0f7156c93"}
``` python
# Fit the classifier to the training data
knn.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="18"}
```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="19" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="fCEuIuF_qyhH" outputId="d07d10e0-7448-429d-ed06-17515684f2d7"}
``` python
# Print the accuracy
print(knn.score(X_test, y_test))
```

::: {.output .stream .stdout}
    0.8545727136431784
:::
:::

::: {.cell .markdown id="tdE9jsyGq8G6"}
# Overfitting and underfitting
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="waouzAmCqz4h" outputId="b56c9cda-50c2-462a-8a1e-469afbf87cd7"}
``` python
# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)

	# Fit the model
	knn.fit(X_train, y_train)

	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
```

::: {.output .stream .stdout}
    [ 1  2  3  4  5  6  7  8  9 10 11 12] 
     {1: 1.0, 2: 0.8885971492873218, 3: 0.8994748687171793, 4: 0.8750937734433608, 5: 0.878469617404351, 6: 0.8660915228807202, 7: 0.8705926481620405, 8: 0.8615903975993998, 9: 0.86384096024006, 10: 0.858589647411853, 11: 0.8604651162790697, 12: 0.8574643660915229} 
     {1: 0.7856071964017991, 2: 0.8470764617691154, 3: 0.8320839580209896, 4: 0.856071964017991, 5: 0.8545727136431784, 6: 0.8590704647676162, 7: 0.8605697151424287, 8: 0.8620689655172413, 9: 0.863568215892054, 10: 0.8605697151424287, 11: 0.8605697151424287, 12: 0.8605697151424287}
:::
:::

::: {.cell .markdown id="nhrNDXiBrOSr"}
# Visualizing model complexity
:::

::: {.cell .code execution_count="23" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":472}" id="at9Z-XJxrGee" outputId="71dd7b75-b9f0-445c-c046-7622af868b51"}
``` python
# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_13aa0871e7df4015bc19eab03b678e69/f01b5c25f44ea6bdc3f93b64bf46617616eb253a.png)
:::
:::

::: {.cell .markdown id="7mH-eVovr8Xh"}
# Conclusion

As could be seen in the chart above, training accuracy decreases and
test accuracy increases as the number of neighbors gets larger. For the
test set, accuracy peaks with 7 neighbors, suggesting it is the optimal
value for our model.
:::

::: {.cell .code id="44FUx8NQrRUv"}
``` python
```
:::
