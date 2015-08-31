# decision_tree_visualize
Visualization of decision tree output from scikit-sklearn.

## Steps to use the visualization

Clone the repository 
```
> git clone https://github.com/deebuls/decision_tree_visualize.git
> cd decision_tree_visualize
```

Use the export.py  with decision tree to convert output to json file
```python
from sklearn.datasets import load_iris
from sklearn import tree
clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
```

```python
 from export import export_json  # loads export json from the export.py 
 out_file = export_json(clf, out_file="iris_tree.json")
 out_file.close()
```

Open in browser the visualization website
```
 http://deebuls.github.io/decision_tree_visualize/

```
Upload the json file created above in the website to visualize.

Enjoy the visualization !!! 

