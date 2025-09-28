import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def data_processor():
    data_file = "./data/air_data.csv"
    data = pd.read_csv(data_file)

    # 删除第一年票价为空或第二年票价为空的数据，null数据会影响后续计算
    # 即保留第一年票价不为空，并且第二年票价也不为空的数据
    # 字段解释：
    # SUM_YR_1      第一年总票价
    # SUM_YR_2      第二年总票价
    # SEG_KM_SUM    观测窗口总飞行公里数
    # avg_discount  平均折扣率
    print("删除第一年票价为空或第二年票价为空的数据")
    data = data[data["SUM_YR_1"].notnull() & data["SUM_YR_2"].notnull()]

    # 删除数据集中票价为零，但飞行公里大于零的不合理值
    print("删除数据集中票价为零，但飞行公里或平均折扣率不等于0的数据")
    index1 = data["SUM_YR_1"] != 0
    index2 = data["SUM_YR_2"] != 0
    index3 = (data["SEG_KM_SUM"] == 0 & (data["avg_discount"] == 0))
    data = data[index1 | index2 | index3]

    # 选择数据集中较重要的特征值
    # 字段解释：
    # FFP_DATE          入会时间，办理会员卡的开始的时间
    # LOAD_TIME         观测窗口的结束时间，选取样本的时间宽度，距离现在最近的时间。
    # FLIGHT_COUNT      飞行次数，频数
    # SUM_YR_1          第一年总票价
    # SUM_YR_2          第二年总票价
    # AVG_INTERVAL      平均乘机时间间隔
    # MAX_INTERVAL      观察窗口内最大乘机间隔
    # avg_discount      平均折扣率
    # filter_data = data[["FFP_DATE", "LOAD_TIME", "FLIGHT_COUNT", "SUM_YR_1", "SUM_YR_2", "AVG_INTERVAL", "MAX_INTERVAL", "avg_discount"]]
    print("计算数据集中入会时间，平均每公里票价，和时间间隔差值")

    # 将字符串格式的日期转换为pandas的datetime格式
    data['FFP_DATE'] = pd.to_datetime(data['FFP_DATE'])
    data['LOAD_TIME'] = pd.to_datetime(data['LOAD_TIME'])
    data["入会时间"] = data["LOAD_TIME"] - data["FFP_DATE"]
    data["平均每公里票价"] = (data["SUM_YR_1"] + data["SUM_YR_2"]) / data["SEG_KM_SUM"]

    # 创建行为稳定性指标，计算最大间隔与平均间隔的差值。
    # 反映客户飞行行为的规律，差值越大，说明飞行时间越不规律。
    data["时间间隔差值"] = data["MAX_INTERVAL"] - data["AVG_INTERVAL"]

    # 将英文列明改为中文，提高可读性。inplace=False表示创建新的DataFrame，不修改原数据
    deal_data = data.rename(
        columns={"FLIGHT_COUNT": "飞行次数", "SEG_KM_SUM": "总里程", "avg_discount": "平均折扣率"},
        inplace=False
    )

    # 选取六个核心特征用于聚类分析
    filter_data = deal_data[["入会时间", "飞行次数", "平均每公里票价", "总里程", "时间间隔差值", "平均折扣率"]].copy()
    # 将入会时间转换为数值
    filter_data["入会时间"] = filter_data["入会时间"].astype(np.int64) / (60*60*24*10**9)
    # 数据标准化（Z-score标准化）
    # （原值 - 均值）÷ 标准差
    filter_zscore_data = (filter_data - filter_data.mean()) / filter_data.std()

    print("输出标准化后的核心特征数据集")
    print_data_info(filter_zscore_data)

    return filter_zscore_data

def print_data_info(data):
    print(data.shape)
    print(data.info())
    print(data.head())

def test_kmeans_nclusters(dataset):
    # 计算不同的k值时，SSE的大小变化
    dataset = dataset.values
    nums = range(2, 10)
    sse_list = []
    for num in nums:
        sse = 0
        kmodel = KMeans(n_clusters=num)
        kmodel.fit(dataset)
        # 簇中心
        cluster_center_list = kmodel.cluster_centers_
        # 各样本属于的簇序号列表
        cluster_list = kmodel.labels_.tolist()
        for index in range(len(dataset)):
            cluster_num = cluster_list[index]
            sse += dist_eclud(dataset[index], cluster_center_list[cluster_num])

        print("簇数是", num, "时：SSE是", sse)
        sse_list.append(sse)
    return nums, sse_list

def dist_eclud(vec1, vec2):
    # 计算两个向量的欧式距离的平方，并返回
    return np.sum(np.power(vec1 - vec2, 2))

def draw_sse_line(nums, sse_list):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['font.size'] = 12.0
    plt.rcParams['axes.unicode_minus'] = False
    # 使用ggplot的绘图风格
    plt.style.use('ggplot')
    ## 绘图观测SSE与簇个数的关系
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, sse_list, marker="+")
    ax.set_xlabel("n_clusters", fontsize=18)
    ax.set_ylabel("SSE", fontsize=18)
    fig.suptitle("KMeans", fontsize=20)
    plt.show()

def kmeans_nclusters_and_draw_radar(cluster):
    kmodel = KMeans(n_clusters=cluster)
    kmodel.fit(data_train)
    # 简单打印结果
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
    # 所有簇中心坐标值中最大值和最小值
    max_cluster_center = r2.values.max()
    min_cluster_center = r2.values.min()
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(data_train.columns) + [u'类别数目']  # 重命名表头

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    center_num = r.values
    feature = ["入会时间", "飞行次数", "平均每公里票价", "总里程", "时间间隔差值", "平均折扣率"]
    n = len(feature)
    for i, v in enumerate(center_num):
        # 设置雷达图的角度，用于平分切开一个圆面
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # 为了使雷达图一圈封闭起来，需要下面的步骤
        center = np.concatenate((v[:-1], [v[0]]))
        angles_closed = np.concatenate((angles, [angles[0]]))
        # 绘制折线图
        ax.plot(angles_closed, center, 'o-', linewidth=2, label="第%d簇人群,%d人" % (i + 1, v[-1]))
        # 填充颜色
        ax.fill(angles_closed, center, alpha=0.25)
        # 添加每个特征的标签
        ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=15)
        # 设置雷达图的范围
        ax.set_ylim(min_cluster_center - 0.1, max_cluster_center + 0.1)
        # 添加标题
        plt.title('客户群特征分析图', fontsize=20)
        # 添加网格线
        ax.grid(True)
        # 设置图例
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1, fancybox=True, shadow=True)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 加载数据，并进行数据处理
    data_train = data_processor()
    # 不同的簇的个数，计算误差平方和SSE
    numbers, sse_values = test_kmeans_nclusters(data_train)
    # 绘制SSE和簇的个数的曲线，寻找肘部
    draw_sse_line(numbers, sse_values)

    # 肘部不明显，选取k = 4, 5, 6重新进行聚类算法，并绘制雷达图
    for k in range(4, 7):
        kmeans_nclusters_and_draw_radar(k)
