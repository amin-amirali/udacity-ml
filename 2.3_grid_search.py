from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

clf = DecisionTreeClassifier(random_state=42)
parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]}
scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

grid_fit = grid_obj.fit(X, y)

best_clf = grid_fit.best_estimator_

best_clf.fit(X_train, y_train)

best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

# Let's also explore what parameters ended up being used in the new model.
best_clf

