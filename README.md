# What Drives the Price of a Car?

For the detailed steps I used in this analysis, please refer to [Jupyter Notebook](https://github.com/losflo/ml-what-drives-the-price-of-a-car/main/prompt_II.ipynb)

Greetings `client`,

I've concluded the analysis you requested on *What drives the price of a car?*. Below is a summary of my findings. If you'd like to know the steps I took to get there they will be in the section called `Details`.

# Summary
The two features I found to be the most significant in the price of a car were as follows: `odometer` and `drive`. My recommendation for inventory tuning is cars with a `price` between 2500 and 9500 with an `odometer` reading between 80000 and 150000 and have a `drive` of `fwd`.

# Details

#### Data Preparation:
`id`, `VIN`, and `region`, `model` were all fields I dropped from the dataset. The first two are unique identifiers which would not help in any model as there would be no relation between unique identifiers, and `region` and `model` were inconsistent and sloppy data. The number of unique values for `region`, `model` were too great to use in the final dataset. One Hot Encoding these fields would've increased the complexity of the entire model 1000X. 

Once those were dropped, I needed to clean Cylinders. Dropping the 'other' category (as this was less than 1% of the dataset), and converting the rest to their integer equivalents with string replacement. I then dropped all None/Null/NA values as there were too many null values in the dataset.

I noticed some very large and very small values in the `price` and `odometer` fields. To mitigate this I found the IQR of each field to create an lower and upper bound which I then filtered out of the dataset.

#### Modeling:
When searching for models to select important features I tested 3 models. Sequential Feature Selection with Ridge as the estimator, Ridge as a standalone feature selection, and LASSO as a standalone feature selection. 

Ridge performed the worst, guessing that `manufacturer` was the biggest indicator. Using `permutation_importance` which tests features independent of each other to see if the model stands up, `manufacturer` only scored a 0.000232. So we can toss this model out.

LASSO performed slightly better, estimating that `odometer` and `fuel` were the two biggest indicators. After checking the values of the `fuel` field, I noticed that `fuel:gas` was a large majority of the dataset which would skew results. I revisted this algorithm dropping the `fuel` field as well. It came back with `odometer` and `manufacturer:porche`. Using `permutation_importance` `odometer` scored very well while `manufacturer:porche` did not. We can drop `manufacturer:porche` and take `odometer` as one of the important features.

Sequential Feature Selection with Ridge performed the highest. It found that `odometer` and `drive` were the highest indicators of price. `permutation_importance` resulted in odometer: 0.444054, fuel: 0.156742. I used the results of this model to draw the plot shown above. Then drew a box around the darkest part which would indicate what is the most popular.