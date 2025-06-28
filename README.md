# Task-1_DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JYOTSANA BHARDWAJ

*INTERN ID*: CT08DK599

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

*🌸Iris Flower Classification using Decision Tree

📌 Overview:-
This is a simple machine learning project where I implemented a Decision Tree Classifier to classify the species of the famous Iris flower dataset. The Iris dataset is one of the most well-known datasets in machine learning and is often used for beginners to get hands-on with basic ML models. This project helped me understand the entire process of training a supervised ML model. I learned how to load data and visualize the final decision tree.

I completed this project as part of my Internship at CodTech to gain practical experience in basic machine learning with Python and scikit-learn.

💻Technologies Used

Python 3
NumPy
Pandas
Matplotlib
Scikit-learn

🧠 What is the Iris Dataset?

The Iris dataset includes 150 samples of iris flowers from three different species:

Setosa

Versicolor

Virginica

Each flower is described by four features:

1. Sepal length

2. Sepal width

3. Petal length

4. Petal width

The goal of the model is to predict the species of a flower based on these four features.

🧪Steps Performed

1. Import Libraries: I used essential libraries like NumPy and Pandas for data handling and Matplotlib for visualization. I performed the machine learning tasks using scikit-learn.

2. Load Dataset: I loaded the Iris dataset with load_iris() from sklearn, which comes preloaded with features and labels.

3. Data Splitting: I divided the dataset into training (70%) and testing (30%) sets using train_test_split.

4. Model Training: I used the DecisionTreeClassifier from scikit-learn and trained it with the training data.

5. Prediction: I made predictions on the test data with the trained model.

6. Evaluation:

  a. I calculated the accuracy score of the model.

  b. I printed the classification report, which includes precision, recall, and F1-score.

  c. I generated a confusion matrix to see how well the model performed on each class.

7. Visualization: I plotted the Decision Tree with plot_tree(). This helped me understand how the model makes decisions at each node based on the features.

📊 Output Example

Accuracy: Around 95% on the test data.

The tree visualization clearly shows how the data splits based on feature thresholds.

The confusion matrix shows minimal misclassification, indicating that the model performed well.

![Image](https://github.com/user-attachments/assets/d10ccc88-a823-4062-91c1-a4027d0937a6)

📁 How to Run
Make sure you have Python installed along with the required libraries. Then simply run the script:
<pre><code>```bash # Run the Python script python iris_decision_tree.py ```</code></pre>

📬 Contact
If you're a fellow student or beginner in ML and want to discuss or collaborate, feel free to connect!
