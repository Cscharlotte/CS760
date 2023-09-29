import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

## Helper
class Node:
    def __init__(self, type=None, label=None, feature_index=None, threshold=None):
        self.type = type
        self.label = label
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = None
        self.right = None

def readfile(filename):
    with open(filename) as file:
        res = []
        for line in file:
            ll = line.rstrip().split(' ')
            res.append({'x1':float(ll[0]),'x2':float(ll[1]),'label':ll[2]})
        return res


def sklearn_read(filename):
    X = []
    y = []
    with open(filename) as file:
        for line in file:
            values = line.strip().split()
            x1 = float(values[0])
            x2 = float(values[1])
            label = int(values[2])
            X.append([x1, x2])
            y.append(label)
    return X, y

def DCNS(D):
    name_features = ['x1','x2']
    C = {}
    for xi in name_features:
        Cx = []
        v = sorted([point[xi] for point in D])
        sort_xi = sorted(D, key=lambda x: x[xi])
        for i in range(1, len(v)):
            vj_val = sort_xi[i][xi]
            vk_val = sort_xi[i - 1][xi]
            # vj = sort_xi[i]['label']
            # vk = sort_xi[i - 1]['label']
            # if vj != vk and vj_val != vk_val:
            if vj_val != vk_val:
                candidate_split = v[i]
                Cx.append(candidate_split)
        C[xi] = Cx
    return C

def Entropy(x0,x1,n):
    if x0 == 0 and x1 != 0:
        return -x1/n * math.log2(x1/n)
    if x1 == 0 and x0 != 0:
        return -x0 / n * math.log2(x0 / n)
    if x0 == 0 and x1 == 0:
        return 0
    return -x0/n * math.log2(x0/n)-x1/n * math.log2(x1/n)

def GR(class1,class2):
    n1 = len(class1)
    n2 = len(class2)
    n = n1 + n2
    x1_0 = class1.count('0')
    x1_1 = class1.count('1')
    x2_0 = class2.count('0')
    x2_1 = class2.count('1')
    entropy = Entropy(x1_0+x2_0,x1_1+x2_1,n)
    IG = entropy - (n1/n*Entropy(x1_0,x1_1,n1)+n2/n*Entropy(x2_0,x2_1,n2))
    ES = Entropy(n1,n2,n)
    return IG/ES

def FBS(D,C):
    best_gr = -1
    best_c = None
    best_feature = None
    C_keys = list(C.keys())
    for xi in C_keys:
        Cx = C[xi]
        sort_xi = sorted(D, key=lambda x: x[xi])
        xis = [point[xi] for point in sort_xi]
        label_class = [point['label'] for point in sort_xi]
        for cs in Cx:
            ind = len(xis)-1-xis[::-1].index(cs)
            if ind == len(xis)-1:
                ind = xis.index(cs)
            class1 = label_class[:ind]
            class2 = label_class[ind:]
            GainRatio = GR(class1,class2)
            if GainRatio > best_gr:
                best_gr = GainRatio
                best_c = cs
                best_feature = xi
    return best_c,best_feature

def MST(D, split_history=None):
    if split_history is None:
        split_history = []
    if len(D) == 0:
        return Node(None),split_history

    labels = [point['label'] for point in D]
    if len(set(labels)) == 1:
        predicted_class = list(set(labels))[0]
        return {'type': 'leaf', 'label': predicted_class}, split_history
    else:
        C = DCNS(D)
        best_c, best_feature = FBS(D,C)
        other_feature = 'x2' if best_feature == 'x1' else 'x1'
        min_val = min(point[other_feature] for point in D)
        max_val = max(point[other_feature] for point in D)
        split_history.append([best_feature, best_c,(min_val,max_val)])
        node_tree = {'type': 'node', 'feature': best_feature, 'threshold': best_c}
        sort_D = sorted(D, key=lambda x: x[best_feature])
        xi_list = [point[best_feature] for point in sort_D]
        ind_cut = xi_list.index(best_c)
        left_node, split_history = MST(sort_D[ind_cut:],split_history)
        right_node, split_history = MST(sort_D[:ind_cut],split_history)
        node_tree['left'] = left_node
        node_tree['right'] = right_node
        return node_tree, split_history


def predict(tree, data_point):
    if tree['type'] == 'leaf':
        return tree['label']
    else:
        feature = tree['feature']
        threshold = tree['threshold']
        if data_point[feature] >= threshold:
            return predict(tree['left'], data_point)
        else:
            return predict(tree['right'], data_point)


def trainDT(train_data, test_data):
    tree, split_history = MST(train_data)
    correct = 0
    for data_point in test_data:
        prediction = predict(tree, data_point)
        if prediction == data_point['label']:
            correct += 1
    err = 1-correct / len(test_data)
    return tree, split_history, err


def DT_training(D,num):
    training_data = D[:num]
    test_data = D[num:]
    tree, split_history, err = trainDT(training_data,test_data)
    n_node = count_inner_dicts(tree) + 1
    drawboundary(training_data,split_history)
    return n_node, err

def count_inner_dicts(d):
    count = 0
    for key, value in d.items():
        if isinstance(value, dict):
            count += 1
            count += count_inner_dicts(value)
    return count


def sktrain(X,y):
    permute_list = list(np.random.default_rng(seed=19).permutation(10000))
    X_perm = [X[i] for i in permute_list]
    y_perm = [y[i] for i in permute_list]
    num = [32, 128, 512, 2048, 8192]
    errs = []
    nodes = []
    for n in num:
        X_train = X_perm[:n]
        y_train = y_perm[:n]
        X_test = X_perm[n:]
        y_test = y_perm[n:]
        sktree = tree.DecisionTreeClassifier()
        sktree = sktree.fit(X_train, y_train)
        tree_structure = sktree.tree_
        nodes.append(tree_structure.node_count)
        res = sktree.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] != res[i]:
                count += 1
        err = count / len(y_test)
        errs.append(err)
    print(nodes)
    print(errs)
    sns.lineplot(x=num, y=errs)
    plt.xlabel('Training size')
    plt.ylabel('Error')


def drawboundary(D, split_history):
    xs, wl = plt.subplots(figsize=(8,7))
    x1 = [d['x1'] for d in D]
    x2 = [d['x2'] for d in D]
    y = [d['label'] for d in D]
    sorted_sl = sorted(split_history, key=lambda x: x[0])
    ind = [d[0] for d in sorted_sl].index('x2')
    x1_sl = [d[1] for d in sorted_sl[:ind]]
    x2_sl= [d[1] for d in sorted_sl[ind:]]
    x1_slmin = [dd[0] for dd in[d[2] for d in sorted_sl[:ind]]]
    x1_slmax = [dd[1] for dd in[d[2] for d in sorted_sl[:ind:]]]
    x2_slmin = [dd[0] for dd in[d[2] for d in sorted_sl[ind:]]]
    x2_slmax = [dd[1] for dd in[d[2] for d in sorted_sl[ind:]]]
    ax = sns.scatterplot(x=x1, y=x2, hue=y, ax=wl)
    ax.vlines(x=x1_sl,ymin=x1_slmin,ymax=x1_slmax,color = "black", linestyle = "dashed")
    ax.hlines(y=x2_sl,xmin=x2_slmin,xmax=x2_slmax,color = "black", linestyle = "dashed")
    plt.show()


def Dbigplots(D):
    permute_list = list(np.random.default_rng(seed=19).permutation(10000))
    D_perm = [D[i] for i in permute_list]
    num = [32, 128, 512, 2048, 8192]
    nodes = []
    errs = []
    for n in num:
        n_node, err = DT_training(D_perm,n)
        nodes.append(n_node)
        errs.append(err)
    print(nodes)
    sns.lineplot(x=num, y=errs)
    plt.xlabel('Traning size')
    plt.ylabel('Error')

# D = readfile('\\Dbig.txt')
# X,y = sklearn_read('\\Dbig.txt')
# tree,split_history = MST(D)
# drawboundary(D,split_history)


