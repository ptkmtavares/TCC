from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import dataExtractor

# --------------------------------------------
# Specify seed random number generator
# --------------------------------------------
seed = 9
np.random.seed(seed)

# --------------------------------------------
# Load data and targets
# --------------------------------------------
selected_data = ['ham', 'spam', 'phishing']
data, index = dataExtractor.getTrainingTestSet('Dataset/index', selected_data, 1.0)

print('data={size:d}'.format(size=len(data)))
print('index={size:d}'.format(size=len(index)))

# --------------------------------------------
# Splits train and test sets
# --------------------------------------------
P = 0.75  # percentage reserved for training
train_set, test_set, train_labels, test_labels = train_test_split(data, index, train_size=P, random_state=seed, shuffle=True)
print('Train test size={size:d}'.format(size=len(train_labels)))

# 3. Preprocess the data by scaling the features
scaler = StandardScaler()
train_set_scaled = scaler.fit_transform(train_set)
test_set_scaled = scaler.transform(test_set)

# 4. Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(40,), activation='tanh', learning_rate='adaptive', solver='adam', alpha=0.001, max_iter=1000, random_state=seed)
mlp.fit(train_set_scaled, train_labels)

accuracy = 100 * mlp.score(test_set_scaled, test_labels)
print('Accuracy for MLP classifier(%)={accuracy:.2f}'.format(accuracy=accuracy))

pred_labels = mlp.predict(test_set_scaled)
num_classes = len(set(test_labels))

cm = confusion_matrix(test_labels, pred_labels)
print('Confusion matrix for MLP classifier:\n', cm)

# Plot confusion matrix for Naive Bayes classifier
fig, ax = plt.subplots()
cm_with_totals = np.zeros((num_classes + 1, num_classes + 1))
cm_with_totals[:num_classes, :num_classes] = cm
cm_with_totals[num_classes, :num_classes] = np.sum(cm, axis=0)
cm_with_totals[:num_classes, num_classes] = np.sum(cm, axis=1)
cm_with_totals[num_classes, num_classes] = np.trace(cm)
im = ax.imshow(cm_with_totals, cmap='RdYlGn')

# Add count values to each cell
for i in range(num_classes + 1):
    for j in range(num_classes + 1):
        if(i == num_classes or j == num_classes):
                text = ax.text(j, i, "{:.0f}\n{:.2%}".format(cm_with_totals[i, j], cm_with_totals[i, j] / np.sum(cm)),
                           ha="center", va="center", color="white", fontweight='bold')
        else:
            text = ax.text(j, i, "{:.0f}\n{:.2%}".format(cm_with_totals[i, j], cm_with_totals[i, j] / np.sum(cm)),
                           ha="center", va="center", color="black")

# Set axis labels and title
ax.set_xticks(range(num_classes + 1))
ax.set_yticks(range(num_classes + 1))
class_labels = selected_data
class_labels.append('Total')
ax.set_xticklabels(class_labels, rotation=45)
ax.set_yticklabels(class_labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('Confusion matrix for MLP classifier')

# Add colorbar
plt.colorbar(im)

# Display the plot
plt.show()

# Get the feature importance using permutation importance
results = permutation_importance(mlp, test_set_scaled, test_labels, scoring='accuracy', n_repeats=2, n_jobs=-1)
importance = results.importances_mean # type: ignore

fig = plt.figure(figsize=(25, 7))

importances_sorted = sorted(zip(importance, dataExtractor.features), reverse=True)
feature_sorted, importance_sorted = zip(*importances_sorted)

perm_mlp_feature_importances = importance_sorted

print("Top features sorted:")
for x, imp in zip(feature_sorted, importance_sorted):
    print('%s, Score: %f' % (imp, x))

plt.xticks(rotation='vertical')
plt.bar([x for x in importance_sorted], feature_sorted, width=0.3)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance of MLP Classifier')

# Adjust the layout to prevent labels from being cut off
plt.tight_layout()

# Display the feature importances plot
plt.show()