import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    indexes = np.argsort(feature_vector)
    feature_vector = feature_vector[indexes].astype(np.float64)
    target_vector = target_vector[indexes].astype(np.float64)

    R_l = np.arange(1, len(feature_vector))
    R_r = np.arange(1, len(feature_vector))[::-1]
    R = len(feature_vector)

    cum_sum = np.cumsum(target_vector)
    p_1_l = (cum_sum[:-1]/np.arange(1, len(feature_vector)))
    p_0_l = 1 - p_1_l
    p_1_r = ((cum_sum[-1] - cum_sum[:-1])/np.arange(1, len(feature_vector))[::-1])
    p_0_r = 1 - p_1_r

    H_l = 1 - p_1_l**2 - p_0_l**2
    H_r = 1 - p_1_r**2 - p_0_r**2

    Q = (-R_l*H_l - R_r*H_r)/R

    thresholds = (feature_vector[1:] + feature_vector[:-1])/2

    indexes = np.nonzero(feature_vector[1:] != feature_vector[:-1])[0]
    unique_thresholds = thresholds[indexes]

    if indexes.shape[0]==0:
        return unique_thresholds, Q, thresholds[0], None
        
    Q = Q[indexes]

    ind = list(Q).index(max(Q))

    return unique_thresholds, Q, unique_thresholds[ind], Q[ind]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep):
        return {'feature_types': self._feature_types,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'min_samples_leaf': self._min_samples_leaf}

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]) or (self._max_depth is not None and node["depth"] == self._max_depth):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._min_samples_split is not None and sub_y.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count

                feature_vector = np.array(list(map(lambda x: ratio[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, ratio.items())))
                else:
                    raise ValueError

        if gini_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        if self._min_samples_leaf is not None and (sub_y[split].shape[0] < self._min_samples_leaf or sub_y[np.logical_not(split)].shape[0] < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        node["left_child"], node["right_child"] = {}, {}
        node["left_child"]["depth"] = node["depth"] + 1
        node["right_child"]["depth"] = node["depth"] + 1

        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"]=="terminal":
            return node["class"]
        else:
            feature_best = node["feature_split"]
            if self._feature_types[feature_best] == "real":
                if x[feature_best]<node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif self._feature_types[feature_best] == "categorical":
                if x[feature_best] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                raise ValueError


    def fit(self, X, y):
        self._tree["depth"] = 0
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)