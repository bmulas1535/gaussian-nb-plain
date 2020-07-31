# Gaussian Naive Bayes Classifier

## Method

The GaussianNB Classifier is designed to predict the class for a given observation as a probability of that observation being made if derived from either class. The class with the highest probability is chosen as the prediction. Probabilty is calculated using Bayesian Probability, wherein the "Prior" is the class occurance as observed in the training data. $$ P(A|B) = \frac{P(B|A) * P(A)}{P(B)} $$

In this way, the probability of observing x given that it came from any of the available classes is calculated, assuming a normal class distribution. The probabilities are compared, and the class with the highest is chosen as the prediction. 

The probability density function calculates the probability of making an observation from any class by finding the z-score distance of x from a given class, using a method derived from the Gaussian distribution equation:
$$
p(y) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})^2}
$$

The class conditional probability is equal to the product of the probability of observing any feature: $ P(Y | X) = P(x_1 | Y) * P(x_2 | Y) * . . . * P(x_n | Y) * P(Y) $

Now, the denominator of the Bayesian Inference is notably missing. This is due to the fact that after considering the adding the conditional probabilities for all evidence both satisfying and not satisfying our hypothesis, the probabilities sum to 100%, so the denominator is 1.

## Assumptions

This model is called "Naive" because it makes some very basic assumptions.
1. A normal class distribution
2. All features are independent

This is usually not exactly the case in most real world applications, but in practice the model still performs reasonably well regardless. If the model does _not_ perform as well as you might expect, these are the first things to consider checking.

## Usage

The implementation is fairly straightforward. Import the classifier, instantiate it, and fit to your model data. The classifier object takes an $ X $ feature matrix, and a $ y $ target vector as inputs. The expected datatype is an _ndarray_ for $ X $, and _1darray_ for $ y $.

```python
>>> from nb import NaiveBayes
>>> 
>>> # We have some hypothetical X features and a y target
>>> nb_model = NaiveBayes()
>>> nb_model.fit(X_train, y_train) # X feature matrix and y target vector
>>> nb_model.score(X_test, y_test)
0.834
>>>
```
