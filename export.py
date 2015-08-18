import numpy as np

from sklearn.tree import _tree


def export_json(decision_tree, out_file=None, feature_names=None):
    """Export a decision tree in JSON format.

    This function generates a JSON representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to JSON.

    out : file object or string, optional (default=None)
        Handle or name of the output file.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    Returns
    -------
    out_file : file object
        The file object to which the tree was exported.  The user is
        expected to `close()` this object when done with it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> import tempfile
    >>> out_file = tree.export_json(clf, out_file=tempfile.TemporaryFile())
    >>> out_file.close()
    """
    import numpy as np

    from sklearn.tree import _tree

    def arr_to_py(arr):
        arr = arr.ravel()
        wrapper = float
        if np.issubdtype(arr.dtype, np.int):
            wrapper = int
        return map(wrapper, arr.tolist())


    def node_to_str(tree, node_id):
        node_repr = '"error": %.4f, "samples": %d, "value": %s' \
                    % (tree.impurity[node_id],
                       tree.n_node_samples[node_id],
                       arr_to_py(tree.value[node_id]))
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            label = '"label": "%s <= %.2f"' % (feature,
                                               tree.threshold[node_id])
            node_type = '"type": "split"'
        else:
            node_type = '"type": "leaf"'
            label = '"label": "Leaf - %d"' % node_id
        node_repr = ", ".join((node_repr, label, node_type))
        return node_repr

    def recurse(tree, node_id, parent=None):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Open node with description
        out_file.write('{%s' % node_to_str(tree, node_id))

        # write children
        if left_child != _tree.TREE_LEAF:  # and right_child != _tree.TREE_LEAF
            out_file.write(', "children": [')
            recurse(tree, left_child, node_id)
            out_file.write(', ')
            recurse(tree, right_child, node_id)
            out_file.write(']')

        # close node
        out_file.write('}')

    if out_file is None:
        out_file = open("tree.json", "w")
    elif isinstance(out_file, basestring):
        out_file = open(out_file, "w")

    if isinstance(decision_tree, _tree.Tree):
        recurse(decision_tree, 0)
    else:
        recurse(decision_tree.tree_, 0)

    return out_file
