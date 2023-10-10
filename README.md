# Digit-Datasets-Classifications
Using scikitlearn, applied four different classifiers to improve on the classification accuracy using sampling.
After importing the necessary libraries, I used the digit datasets. Once its loaded, I converted the digit datasets to data frame. I created a function the getDataFrame() to do the data frame conversion. Then, assigning the dataframe function to a variable to access iloc function and select the index of the ones being sliced using lambda x. So the data and target are separated and assigning them to X and y. The k fold cross validation is then assigned to an object. After that, since we are using four classifiers, created an array of the x training data, y training data, x test data, y test data and accuracy and appended them. Using the for loop with the k fold cross validation will give us high accuracy because of the best training and testing split, then took those training and testing split assigned to arrays to get the maximum accuracy. At this point, I believe that it is the finest or best split from the results of K fold technique we can get.

Output results

![image](https://github.com/evanjephi/Digit-Datasets-Classifications/assets/73504127/7f7f429d-0549-44ae-82c8-28d034684d01)

