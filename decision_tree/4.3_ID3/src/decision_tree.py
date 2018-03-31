# -*- coding: utf-8 -*
from math import log2
from pydotplus import graphviz


class Node(object):
    '''
    节点
    attr: 作为一个新的节点的父节点
    attr_down: dict: {key, value}
            key:   类别:  类别属性值
                   连续: '<= div_value' for small part
                               '> div_value' for big part
            value: 子节点
    label： 类别标签
    '''
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init  
        self.label = label_init 
        self.attr_down = attr_down_init
        pass
    pass


def TreeGenerate(df):
    ''' 
    生成决策树

    @param df: the pandas dataframe of the data_set
    @return root: Node, the root node of decision tree
    '''
    # generating a new root node
    new_node = Node(None, None, {})
    label_arr = df[df.columns[-1]]
    
    label_count = NodeLabel(label_arr)
    if label_count:
        new_node.label = max(label_count, key=label_count.get)
            
        # 如果当前节点只有一类或者当前属性为空，返回
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        
        # get the optimal attribution for a new branching
        new_node.attr, div_value = OptAttr(df)
        
        # recursion
        if div_value == 0:  # 离散属性
            value_count = ValueCount(df[new_node.attr]) 
            for value in value_count:
                df_v = df[ df[new_node.attr].isin([value]) ]  # get sub set
                # delete current attribution
                df_v = df_v.drop(new_node.attr, 1)  
                new_node.attr_down[value] = TreeGenerate(df_v)
                
        else:  # 连续属性
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df[df[new_node.attr] <= div_value]
            df_v_r = df[ df[new_node.attr] > div_value ]
 
            new_node.attr_down[value_l] = TreeGenerate(df_v_l)
            new_node.attr_down[value_r] = TreeGenerate(df_v_r)
        
    return new_node
    

def Predict(root, df_sample):
    '''
    在根上做预测

    @param root: 根节点
    @param df_sample: dataframe, a sample line 
    '''
    try :
        import re # using Regular Expression to get the number in string
    except ImportError :
        print("module re not found")
    
    while root.attr != None :        
        # continuous variable
        if df_sample[root.attr].dtype in (float, int):
            # get the div_value from root.attr_down
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*",key)
                div_value = float(num[0])
                break
            if df_sample[root.attr].values[0] <= div_value:
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]
                
        # categoric variable
        else:  
            key = df_sample[root.attr].values[0]
            # check whether the attr_value in the child branch
            if key in root.attr_down: 
                root = root.attr_down[key]
            else: 
                break
            
    return root.label

#  计算出现的类标和以及它的数目
def NodeLabel(label_arr):
    '''
    参数： label_arr: 列表
    返回值： label_count: dict, 出现的类标和以及它的数目
    '''
    # store count of label
    label_count = {}
    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        
    return label_count

# 计算出现的类别属性的值和它的数目
def ValueCount(data_arr):
    '''
  @param data_arr: 一个类别属性数组
    @return value_count: dict, 出现的类别属性的值和它的数目
    '''
    value_count = {}       # store count of value 
      
    for label in data_arr:
        if label in value_count: value_count[label] += 1
        else: value_count[label] = 1
        
    return value_count


def OptAttr(df):
    '''
    查找最优属性

    @param df: the pandas dataframe of the data_set 
    @return opt_attr:  the optimal attribution for branch
    @return div_value: for discrete variable value = 0
                       for continuous variable value = t for bisection divide value
    '''
    # 信息增益
    info_gain = 0
    
    for attr_id in df.columns[1:-1]:
        info_gian_tmp, div_value_tmp = InfoGain(df, attr_id)
        if info_gian_tmp > info_gain:
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
        
    return opt_attr, div_value
        
# 计算一个类别的增益率
def InfoGain(df, index):
    '''
    @param df:      dataframe, the pandas dataframe of the data_set
    @param attr_id: the target attribution in df
    @return info_gain: the information gain of current attribution
    @return div_value: for discrete variable, value = 0
                   for continuous variable, value = t (the division value)
    '''
    info_gain = InfoEnt(df.values[:,-1])  # info_gain for the whole label
    div_value = 0  # div_value for continuous attribute
    # 样例数目
    n = len(df[index])
    # 1. 对于连续属性使用二分法
    if df[index].dtype in [float, int]:
        sub_info_ent = {}  # store the div_value (div) and it's subset entropy
        
        df = df.sort_values([index], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)
        
        data_arr = df[index]
        label_arr = df[df.columns[-1]]
        
        for i in range(n-1):
            div = (data_arr[i] + data_arr[i+1]) / 2
            sub_info_ent[div] = ( (i+1) * InfoEnt(label_arr[0:i+1]) / n ) \
                              + ( (n-i-1) * InfoEnt(label_arr[i+1:-1]) / n )
        # our goal is to get the min subset entropy sum and it's divide value
        div_value, sub_info_ent_max = min(sub_info_ent.items(), key=lambda x: x[1])
        info_gain -= sub_info_ent_max
        
    # 2.离散性的， 类别变量
    else:
        data_arr = df[index]
        label_arr = df[df.columns[-1]]
        value_count = ValueCount(data_arr)
            
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n
    
    return info_gain, div_value
    
# 计算一个属性的信息熵
def InfoEnt(label_arr):
    '''
    @param label_arr: ndarray, class label array of data_arr
    @return ent: the information entropy of current attribution
    '''
    # 信息熵
    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)
    
    for key in label_count:
        ent -= (label_count[key] / n) * log2(label_count[key] / n)
    
    return ent

# 可视化决策树
def DrawPNG(root, out_file):
    '''
    @param root: 根节点
    @param out_file: 
    '''
    # generation of new dot
    g = graphviz.Dot()

    TreeToGraph(0, g, root)
    g2 = graphviz.graph_from_dot_data( g.to_string() )
    
    g2.write_png(out_file) 

#  build a graph from root on
def TreeToGraph(i, g, root):
    '''
    @param i: 树中节点的数目
    @param g: pydotplus.graphviz.Dot() object
    @param root: 根节点
    
    @return i: node number after modified  
     @return g: pydotplus.graphviz.Dot() object after modified
    @return g_node: the current root node in graphviz
    '''

    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node(graphviz.Node(g_node, label = g_node_label ))
    
    for value in list(root.attr_down):
        i, g_child = TreeToGraph(i+1, g, root.attr_down[value])
        g.add_edge( graphviz.Edge(g_node, g_child, label = value) ) 

    return i, g_node

