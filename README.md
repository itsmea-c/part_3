This is Part 3 of Capstone Project
Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

Trained Decision Tree Classifier with and without constraints. Calculated training and test accuracy for both and compared their gap

1. Decision Tree baseline:
   Trained a Decision Tree Classifier with no constraints and calculated training accuracy and test accuracy
   Decision Tree Training Accuracy: 1.0
   Decision Tree Test Accuracy: 0.79
   This Decision Tree shows signs of overfitting because the training accuracy is high and test accuracy is low so it has memorized the training data instead of learning patterns
   Decision trees fit the training data greedily at each split, without revisiting earlier decisions. With no depth constraints, they continue splitting until leaves are pure which often      leads to memorizing the training data. This makes them highly sensitive to noise and small changes in the dataset. Hence decision trees are described as high-variance models

2. Controlled Decision Tree:
   Trained a second Decision Tree Classifier with constraints and calculated training accuracy and test accuracy again
   Decision Tree Training Accuracy Limited: 0.8775
   Decision Tree Test Accuracy Limited: 0.81

   max_depth: 
   max_depth will limit how deep the tree can grow. With this we can prevent the tree from endlessly splitting until all leaves are pure. This reduces variance but introduces some bias. If max_depth is set too low it may underfit.

   min_samples_split:  
   min_samples_split specifies the minimum number of samples required to split a node. If a node has fewer samples than this threshold, it will not be split further. This prevents the tree from creating branches that respond to noise in very small subsets of the data. It acts as a safeguard against overfitting by ensuring splits are made only when there is enough data to justify them.

   In a Decision Tree Classifier trained with no constraints, the training accuracy is high and test accuracy is low and the gap is 0.21 which means it has memorized the training data instead of learning patterns
   In a Decision Tree Classifier trained with constraints, the training accuracy is reduced and test accuracy is improved and the gap is 0.06 which shows better generalization

3. Gini vs Entropy comparison:
   Gini impurity formula: 1 - Σ pi²
   Entropy formula: -Σ pi log2(pi)

   If Gini is 0 then the node is pure and all samples inside that node belong to a single class. It has no impurity and such nodes are terminal leaves in the tree. They don’t split further because there is no gain in purity to be achieved.

   Note:

    1. Python script is saved as part_3.py and placed in the repository
    2. Cleaned data is saved as cleaned_data.csv and placed in the repository
