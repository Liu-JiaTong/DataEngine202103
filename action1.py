# 使用KMeans进行聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


def cluster():
    data = pd.read_csv('./DataSet/car_data.csv',  encoding='gbk')
    train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
    # 规范化到 [0,1] 空间
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train_x)
    predict_y = kmeans.predict(train_x)
    # 合并聚类结果，插入到原数据中
    result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
    result.rename({0: u'聚类结果'}, axis=1, inplace=True)
    # 将结果导出到CSV文件中
    result.to_csv("car_data_cluster_result.csv", index=False)
    result_print = result.groupby(result[u'聚类结果'])
    groupIdx = 0
    for res in result_print:
        print("group: %d" % groupIdx)
        groupIdx += 1
        print(res[1]['地区'])
        print('\n')
    print("cluster success")


if __name__ == '__main__':
    cluster()
