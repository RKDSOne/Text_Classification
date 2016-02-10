Classification
--------------
Classification is the task of choosing the correct class label for a given input. In basic classification tasks, each input is considered in isolation from all other inputs, and the set of labels is defined in advance. Some examples of classification tasks are:

Deciding whether an email is spam or not.
Deciding what the topic of a news article is, from a fixed list of topic areas such as "sports," "technology," and "politics."
Deciding whether a given occurrence of the word bank is used to refer to a river bank, a financial institution, the act of tilting to the side, or the act of depositing something in a financial institution.
The basic classification task has a number of interesting variants. For example, in multi-class classification, each instance may be assigned multiple labels; in open-class classification, the set of labels is not defined in advance; and in sequence classification, a list of inputs are jointly classified.

The dataset has 2873 files divided into three categories. The program implements a multinomial naive Bayes classifier for text classification, preprocessed data is created first and then fed into the pipeline that starts with linear classification followed by sentiment reading and then multinomial naive bayes classification.

Finally, the F1 score of the system is been evaluated.