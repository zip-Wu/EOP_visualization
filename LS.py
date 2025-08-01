import numpy as np
import pandas as pd
import Tide

def origin_data(data_path:str)->pd.DataFrame:
    """
    这个函数对输入数据进行预处理，生成对应的Dataframe格式并返回
    
    参数：
        data_path: 来自IRES原始数据的路径
    
    返回：
        pd.dataFrame: 索引date, MJD, X, Y, LOD
    """

    line_num = 0
    with open(data_path, 'r') as file:
        for line in file:
            # 使用正则表达式作为分隔符，匹配任何空白字符
            labels = line.strip().split('  ')  # 不传递分隔符参数，默认使用空白字符作为分隔符
            line_num += 1
            if line_num == 6:
                break

    # 删除所有为空格的元素
    labels = [item for item in labels if item.strip()]
    # 使用列表推导式删除每个元素开头的空格
    labels = [item.lstrip() for item in labels]
    labels[0] = 'YR'

    # 这段代码用来读取数据
    data_list = []
    temp_list = []
    line_num = 0
    # 逐行读取文本文件并添加到 DataFrame 中
    with open(data_path, 'r') as file:
        for line in file:
            line_num += 1
            if line_num < 7:
                continue
            # 假设每行以空格分隔，如果使用其他分隔符，需要相应地调整 split() 方法的参数
            temp_list = line.strip().split()
            data_list.append(temp_list)

    # 最后得到一个dataframe数据
    data = pd.DataFrame(data_list, columns=labels)
    # 将 YR 列转换为整数类型
    data['YR'] = data['YR'].astype(int)

    # 将日期改成datetime的格式，方便后续处理
    date = pd.to_datetime(data['YR'].astype(str) + '-' + data['MM'] + '-' + data['DD'])
    data.insert(0, 'date', date)

    normal_data = data[['date', 'MJD', 'x(")', 'y(")', 'LOD(s)']]
    normal_data.set_index('date', inplace=True)
    return normal_data

# 得到 LS_fit 和 LS_forecast，weights参数可选择是否用WLS
def LS_fit(train_data:pd.DataFrame, pred_len:int, 
           fit_type:str, weights: np.ndarray = None) -> np.ndarray :
    """
    这个函数用来对输入数据进行最小二乘拟合
    
    参数：
        train_data: 训练数据的DataFrame
        train_data: 测试数据的DataFrame
        fit_type: 目前拟合的变量,可选X,Y和LOD
    
    返回：
        训练数据的拟合值(np.array)
        测试数据的预测值(np.array)
    """
    def LS_fit_matrix(data: np.ndarray, func: callable, weights: np.ndarray = None):
        """最小二乘法拟合（线性回归）
    
        Args:
            data (np.ndarray): _description_
            func (_type_): 拟合函数
            weights (np.ndarray): 权重矩阵 1-d
    
        Returns:
            _type_: 拟合结果和参数
        """
        x = np.arange(len(data))
        B = np.array(func(x))
        if weights is not None:
            weights = np.diag(weights)
            popt = np.linalg.inv(B @ weights @ B.T) @ B @ weights @ data
        else:
            popt = np.linalg.inv(B @ B.T) @ B @ data
        lr_y = B.T @ popt
    
        return lr_y, popt
    # 对X,Y定义相同的最小二乘模型
    def LS_func_XY(x):
        return (
            x**0,
            x**1,
            np.cos(2 * np.pi * x / 435.00),
            np.sin(2 * np.pi * x / 435.00),
            np.cos(2 * np.pi * x / 365.24),
            np.sin(2 * np.pi * x / 365.24),
            np.cos(2 * np.pi * x / 182.62),
            np.sin(2 * np.pi * x / 182.62),
            np.cos(2 * np.pi * x / 450.00),
            np.sin(2 * np.pi * x / 450.00),
        )

    def LS_func_LOD(x):
        return (
            x**0,
            x**1,
            np.cos(2 * np.pi * x / 6793.464),
            np.sin(2 * np.pi * x / 6793.464),
            np.cos(2 * np.pi * x / 3396.732),
            np.sin(2 * np.pi * x / 3396.732),
            np.cos(2 * np.pi * x / 365.24),
            np.sin(2 * np.pi * x / 365.24),
            np.cos(2 * np.pi * x / 182.62),
            np.sin(2 * np.pi * x / 182.62),
            np.cos(2 * np.pi * x / 121.747),
            np.sin(2 * np.pi * x / 121.747),
        )

    def LS_forecast_matrix(start_x, n, W, func: callable):
        """根据参数计算拟合结果"""
        x = np.arange(start_x, start_x + n)
        return np.array(func(x)).T @ W

    if fit_type == 'X':
        train = train_data['x(")'].values.astype(float)     # 因变量
        X_fit, params = LS_fit_matrix(train, LS_func_XY, weights)
        # 接下来生成未来的预测形成forecast值
        X_forecast = LS_forecast_matrix(len(train), pred_len, params, LS_func_XY)
        return X_fit, X_forecast
    
    elif fit_type == 'Y':
        train = train_data['y(")'].values.astype(float)     # 因变量
        Y_fit, params = LS_fit_matrix(train, LS_func_XY, weights)
        # 接下来生成未来的预测形成forecast值
        Y_forecast = LS_forecast_matrix(len(train), pred_len, params, LS_func_XY)
        return Y_fit, Y_forecast
    
    elif fit_type == 'LOD':
        # 提取出MJD，并将其输入至潮汐项函数
        MJD_train  = train_data['MJD']
        MJD_start_test = int(train_data['MJD'].astype(float).iloc[-1] + 1)
        MJD_test = np.array(range(MJD_start_test, MJD_start_test+pred_len)).astype(float)

        # 得到潮汐数值
        tide_train = Tide.RG_ZONT2(MJD_train.astype(float))
        tide_test = Tide.RG_ZONT2(MJD_test.astype(float))

        train = train_data['LOD(s)'].astype('float')

        # 接下来用训练数据-潮汐数据，得到真正的需要训练的数据
        train_NonTide = train - tide_train

        LOD_fit_NonTide, params = LS_fit_matrix(train_NonTide, LS_func_LOD, weights)

        # 接下来生成未来的预测形成forecast值
        LOD_forecast_NonTide = LS_forecast_matrix(len(train_NonTide), pred_len, params, LS_func_LOD)
        
        # 最后把fit和forecas的值各自加上各自的潮汐项然后输出
        LOD_fit = LOD_fit_NonTide + tide_train
        LOD_forecast = LOD_forecast_NonTide + tide_test

        return LOD_fit, LOD_forecast

def gen_data(data_path:str, start_time, end_time):
    data = origin_data(data_path)
    train_data = data.loc[pd.to_datetime(start_time): pd.to_datetime(end_time)]
    return train_data

def gen_res(data, fit_type:str, weights):
    fit, _ = LS_fit(data, 10, fit_type, weights)
    selet_type = {'X': 'x(")', 'Y': 'y(")', 'LOD': 'LOD(s)'}
    res = data[selet_type[fit_type]].values.astype(float) - fit
    return res





def gen_test(data_path:str, start_time:str, end_time:str,
            pred_len:int, fit_type:str, weights):
    """
    这个函数用来生成LS的残差数据
    
    参数：
        data_path: 来自IRES原始数据的路径
        start_time: 训练数据的起始年月日
        end_time: 训练数据的终止年月日
        predict_span: 预测数据的长度（天）
        fit_type: 拟合的是X、Y还是LOD
    """
    data = origin_data(data_path)
    train_data = data.loc[pd.to_datetime(start_time): pd.to_datetime(end_time)]
    test_start = pd.to_datetime(end_time) + pd.Timedelta(days=1)
    test_data = data.loc[test_start: test_start+pd.Timedelta(days=pred_len-1)]
    fit, forecast = LS_fit(train_data, len(test_data), fit_type=fit_type, weights=weights)
    selet_type = {'X': 'x(")', 'Y': 'y(")', 'LOD': 'LOD(s)'}
    train_res = train_data[selet_type[fit_type]].values.astype(float) - fit
    test_res = test_data[selet_type[fit_type]].values.astype(float) - forecast

    return train_res, test_res

