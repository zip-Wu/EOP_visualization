import streamlit as st
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import time
import torch
import torch.nn as nn
from LS import gen_data, LS_fit, gen_res
from data.data_loader import create_dataLoader
from model.WZPNet import WZPNet
from utils_ANN import WeightedL1Loss
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="极移预报系统",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 应用自定义样式 - 增加侧边栏宽度
st.markdown("""
<style>
    /* 增加侧边栏宽度 */
    section[data-testid="stSidebar"] {
        width: 450px !important;
    }
    .sidebar-title {
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
        color: #1f77b4;
    }
    .module-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #1f77b4;
    }
    .module-header {
        font-weight: bold;
        margin-bottom: 10px;
        color: #1f77b4;
    }
    .param-expander {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .header-style {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        color: #1f77b4;
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 30px;
        font-size: 14px;
        color: #6c757d;
        border-top: 1px solid #eee;
    }
    .selectbox-compact {
        margin-bottom: 20px;
    }
    .date-section-title {
        font-weight: bold;
        margin: 15px 0 10px 0;
        color: #1f77b4;
        font-size: 16px;
    }
    .main-content {
        padding: 20px;
    }
    .mode-section {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #1890ff;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="header-style">欢迎进入极移预报系统</div>', unsafe_allow_html=True)

# 训练模型函数
def train_WZPNet(module_name, start_time, end_time, EOP, pred_len, seq_len, seq_out, num_epoch, batch_size, progress_bar, status_text):
    # 根据模块名称设置固定参数
    d_model = 64
    dropout = 0.0
    
    # 根据模块类型设置固定参数
    if module_name == "Linear":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = 0
        skip_num = 0
    elif module_name == "GRU":
        seq_ar = 0
        seq_cnn = 0
        seq_gru = seq_len
        gru_layer = 1
        gru_hidden = 64
        skip_num = 0
    elif module_name == "skipGRU":
        seq_ar = 0
        seq_cnn = 0
        seq_gru = 0
        skip_num = seq_len // 3
        skip_len = 3
        skip_layer = 1
        skip_hidden = 64
    elif module_name == "Linear-GRU":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = seq_len
        gru_layer = 1
        gru_hidden = 64
        skip_num = 0
    elif module_name == "Linear-skipGRU":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = 0
        skip_num = seq_len // 3
        skip_len = 3
        skip_layer = 1
        skip_hidden = 64
    
    # 更新状态
    status_text.text("正在生成数据...")
    data = gen_data('./data/data_origin.txt', start_time, end_time)
    LS_weights = None
    train_data = gen_res(data, EOP, LS_weights)
    progress_bar.progress(5)
    
    # 创建数据加载器
    status_text.text("正在准备数据加载器...")
    train_loader, val_loader = create_dataLoader(train_data, seq_len, seq_out, val_num=50, batch_size=batch_size)
    progress_bar.progress(10)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
    status_text.text("正在初始化模型...")
    model = WZPNet(seq_out=seq_out, d_model=d_model, dropout=dropout, seq_ar=seq_ar,
                seq_cnn=0, cnn_kernel=0, cnn_stride=0, cnn_channel=0,  # 固定不使用CNN
                seq_gru=seq_gru if module_name=='GRU' or module_name=='Linear-GRU' else 0, 
                gru_layer=gru_layer if module_name=='GRU' or module_name=='Linear-GRU' else 0, 
                gru_hidden=gru_hidden if module_name=='GRU' or module_name=='Linear-GRU' else 0,
                skip_num=skip_num if 'skipGRU' in module_name else 0, 
                skip_len=skip_len if 'skipGRU' in module_name else 0, 
                skip_layer=skip_layer if 'skipGRU' in module_name else 0, 
                skip_hidden=skip_hidden if 'skipGRU' in module_name else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    weights = torch.linspace(1.0, 0.1, steps=seq_out).to(device)
    criterion = WeightedL1Loss(weights)
    progress_bar.progress(15)
    
    # 训练模型
    temp_loss = 10
    epoch_save = 0
    temp_train = 10
    logs = []
    
    for epoch in range(num_epoch):
        # 更新进度和状态
        progress = 15 + int(70 * (epoch + 1) / num_epoch)
        progress_bar.progress(progress)
        status_text.text(f"训练中... 轮次 {epoch+1}/{num_epoch} (使用 {module_name} 模块)")
        
        model.train()
        train_loss = []
        for input, labels in train_loader:
            input = input.to(device)
            labels = labels.to(device)
            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        train_loss = np.mean(train_loss)
        
        # 验证
        model.eval()
        val_loss = []
        with torch.no_grad():
            for input, labels in val_loader:
                input = input.to(device)
                labels = labels.to(device)
                output = model(input)
                loss = criterion(output, labels)
                val_loss.append(loss.item())
        
        val_loss = np.mean(val_loss)
        
        # 记录日志
        log_entry = f"轮次 [{epoch + 1}], 训练损失: {train_loss*1000:.4f}, 验证损失: {val_loss*1000:.4f}"
        logs.append(log_entry)
        
        if train_loss < temp_train:
            temp_train = train_loss
        else:
            logs.append("训练损失没有下降...")
        
        if val_loss < temp_loss:
            temp_loss = val_loss
            torch.save(model.state_dict(), 'checkpoint.pt')
            epoch_save = epoch
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))
    
    # 生成预测
    status_text.text("正在生成预测结果...")
    train_res = torch.tensor(train_data[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(torch.device(device))
    _, LS_forecast = LS_fit(data, pred_len, EOP, LS_weights)
    model.eval()
    forecast = []
    with torch.no_grad():
        for i in range(pred_len // seq_out):
            res_output = model(train_res)
            forecast.append(res_output.view(-1).to('cpu'))
            train_res = torch.cat((train_res[:, seq_out:, :], res_output.unsqueeze(-1)), dim=1)
        forecast = np.ravel(np.array(forecast))
    
    final = LS_forecast + forecast
    
    # 更新进度条
    progress_bar.progress(100)
    status_text.text("训练完成！")
    
    return logs, final

# 使用预训练模型进行预测
def predict_with_pretrained(module_name, start_time, end_time, EOP, pred_len, seq_len, seq_out, progress_bar, status_text):
    # 更新状态
    status_text.text("正在加载预训练模型...")
    data = gen_data('./data/data_origin.txt', start_time, end_time)
    LS_weights = None
    train_data = gen_res(data, EOP, LS_weights)
    progress_bar.progress(30)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据模块名称设置固定参数
    d_model = 64
    dropout = 0.0
    
    if module_name == "Linear":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = 0
        skip_num = 0
    elif module_name == "GRU":
        seq_ar = 0
        seq_cnn = 0
        seq_gru = seq_len
        gru_layer = 1
        gru_hidden = 64
        skip_num = 0
    elif module_name == "skipGRU":
        seq_ar = 0
        seq_cnn = 0
        seq_gru = 0
        skip_num = seq_len // 3
        skip_len = 3
        skip_layer = 1
        skip_hidden = 64
    elif module_name == "Linear-GRU":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = seq_len
        gru_layer = 1
        gru_hidden = 64
        skip_num = 0
    elif module_name == "Linear-skipGRU":
        seq_ar = seq_len
        seq_cnn = 0
        seq_gru = 0
        skip_num = seq_len // 3
        skip_len = 3
        skip_layer = 1
        skip_hidden = 64
    
    # 初始化模型
    model = WZPNet(seq_out=seq_out, d_model=d_model, dropout=dropout, seq_ar=seq_ar,
                seq_cnn=0, cnn_kernel=0, cnn_stride=0, cnn_channel=0,  # 固定不使用CNN
                seq_gru=seq_gru if module_name=='GRU' or module_name=='Linear-GRU' else 0, 
                gru_layer=gru_layer if module_name=='GRU' or module_name=='Linear-GRU' else 0, 
                gru_hidden=gru_hidden if module_name=='GRU' or module_name=='Linear-GRU' else 0,
                skip_num=skip_num if 'skipGRU' in module_name else 0, 
                skip_len=skip_len if 'skipGRU' in module_name else 0, 
                skip_layer=skip_layer if 'skipGRU' in module_name else 0, 
                skip_hidden=skip_hidden if 'skipGRU' in module_name else 0).to(device)
    
    # 加载预训练模型
    model_path = f"./saved_model/{module_name}.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    progress_bar.progress(60)
    
    # 生成预测
    status_text.text("正在生成预测结果...")
    train_res = torch.tensor(train_data[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(torch.device(device))
    _, LS_forecast = LS_fit(data, pred_len, EOP, LS_weights)
    forecast = []
    with torch.no_grad():
        for i in range(pred_len // seq_out):
            res_output = model(train_res)
            forecast.append(res_output.view(-1).to('cpu'))
            train_res = torch.cat((train_res[:, seq_out:, :], res_output.unsqueeze(-1)), dim=1)
        forecast = np.ravel(np.array(forecast))
    
    final = LS_forecast + forecast
    
    # 更新进度条
    progress_bar.progress(100)
    status_text.text("预测完成！")
    
    return final

# 侧边栏参数选择区域
with st.sidebar:
    st.markdown('<div class="sidebar-title">参数配置</div>', unsafe_allow_html=True)
    
    # 模式选择
    st.markdown('</div>', unsafe_allow_html=True)
    mode = st.radio(
        "选择模型模式",
        ["使用预训练模型", "训练新模型"],
        index=0,
        help="选择使用预训练模型或自己训练模型", 
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 精简的selectbox，不加框
    st.markdown('<div class="selectbox-compact">', unsafe_allow_html=True)
    EOP_select = st.selectbox(
        '选择预报参数', 
        ['PMX', 'PMY'],
        help="选择极移预报所需的参数类型"
    )
    if EOP_select == 'PMX':
        EOP = 'X'
    elif EOP_select == 'PMY':
        EOP = 'Y'
    st.markdown('</div>', unsafe_allow_html=True)
    
    if mode == "使用预训练模型":
        st.markdown('<div class="date-section-title">请选择预报起始日期</div>', unsafe_allow_html=True)
        min_date = datetime(1982, 1, 1)
        max_date = datetime(2024, 1, 1)
        end_date = st.date_input(
            "预报起始日期：",
            min_value=min_date,
            max_value=max_date,
            value=datetime(2024, 1, 1),
            key="end_date",
            help="选择预报的起始日期"
        )
        start_date = end_date - pd.DateOffset(years=20)  # 默认使用过去20年的数据进行预报
        st.info("注意，当使用预训练模型时，该预训练的模型使用的训练数据集为 2000年1月2日-2020年1月2日 共20年已知数据。", icon="ℹ️")
    else:
        # 日期选择区域
        st.markdown('<div class="date-section-title">训练数据范围</div>', unsafe_allow_html=True)
        # 日期选择器
        col_date_1, col_date_2 = st.columns(2)
        min_date = datetime(1962, 1, 1)
        max_date = datetime(2024, 1, 1)
        with col_date_1:
            start_date = st.date_input(
                "起始日期：", 
                min_value=min_date, 
                max_value=max_date, 
                value=datetime(2004, 1, 1),
                key="start_date",
                help="选择训练数据的起始日期"
            )
        with col_date_2:
            end_date = st.date_input(
                "终止日期：", 
                min_value=min_date, 
                max_value=max_date, 
                value=max_date,
                key="end_date",
                help="选择训练数据的结束日期"
            )
    


    # 输入输出参数
    st.markdown('<div class="date-section-title">输入输出参数</div>', unsafe_allow_html=True)
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        pred_len = st.number_input("输出预报长度 (pred_len)", 
                                  min_value=1, max_value=500, value=360, step=1,
                                  key="pred_len")
    with col_pred2:
        seq_len = st.number_input("输入历史序列长度 (seq_len)", 
                            min_value=10, max_value=500, value=200, step=1,
                            key="seq_len")
    
    # 滑动窗口选项
    use_sliding = st.radio("是否使用滑动窗口预测？", 
                          ["是", "否"], 
                          index=0,
                          help="选择'是'时使用滑动窗口预测，选择'否'时一次性输出整个预测序列", horizontal=True)
    
    if use_sliding == "是":
        seq_out = st.number_input("滑动窗口单次输出长度 (seq_out ≤ pred_len)", 
                                min_value=1, max_value=pred_len, value=20, step=1,
                                key="seq_out")
    else:
        seq_out = pred_len
    
    # 根据模式显示不同内容
    if mode == "使用预训练模型":
        # 预训练模型选择
        st.markdown('<div class="date-section-title">模型选择</div>', unsafe_allow_html=True)
        module_name = st.selectbox(
            "选择预训练模型",
            ["Linear", "GRU", "skipGRU", "Linear-GRU", "Linear-skipGRU"],
            index=0,
            help="选择要使用的预训练模型"
        )
        
        # 开始预测按钮
        predict_button = st.button("开始预测", use_container_width=True, key="predict_button", type='primary')
    else:
        # 自己训练模型选项
        st.markdown('<div class="date-section-title">模型选择</div>', unsafe_allow_html=True)
        module_name = st.selectbox(
            "选择要训练的模型",
            ["Linear", "GRU", "skipGRU", "Linear-GRU", "Linear-skipGRU"],
            index=0,
            help="选择要训练的神经网络模型"
        )
        
        # 训练参数
        st.markdown('<div class="date-section-title">训练参数</div>', unsafe_allow_html=True)
        col_train1, col_train2 = st.columns(2)
        with col_train1:
            num_epoch = st.number_input("训练轮数 (num_epoch)", 
                                      min_value=1, max_value=4000, value=20, step=1,
                                      key="num_epoch")
        with col_train2:
            batch_size = st.number_input("批大小 (batch_size)", 
                                      min_value=8, max_value=1024, value=64, step=8,
                                      key="batch_size")
        
        # 开始训练按钮
        train_button = st.button("开始训练", use_container_width=True, key="train_button", type='primary')

# 主内容区域
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if mode == "训练新模型":
        st.warning("注意：训练新模型可能需要较长时间，请耐心等待训练完成。", icon="⚠️")
        # 显示当前选择的信息
        st.success(f"已选择参数：PM{EOP}，  \t数据范围: {start_date} 至 {end_date}，  第一个预报点的日期为: {end_date + pd.Timedelta(days=1)}，  \t模式：{mode}")
    else:
        st.info("提示：使用预训练模型进行预测，速度较快。", icon="ℹ️")
        # 显示当前选择的信息
        st.success(f"已选择参数：PM{EOP}，  \t第一个预报点的日期为: {end_date + pd.Timedelta(days=1)}，  \t模式：{mode}")

    
    # 显示模块选择状态
    st.info(f"当前选择的模型: {module_name}")
    
    # 初始化结果容器
    result_placeholder = st.empty()
    log_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 加载数据
    train_data_df = gen_data('./data/data_origin.txt', start_date, end_date)
    # 加载测试数据
    y_start = end_date + pd.Timedelta(days=1)
    y_end = y_start + pd.Timedelta(days=pred_len-1)
    test_data_df = gen_data('./data/data_origin.txt', y_start, y_end)

    selet_type = {'X': 'x(")', 'Y': 'y(")', 'LOD': 'LOD(s)'}
    x = np.array(train_data_df[selet_type[EOP]].values.astype(float))
    y = np.array(test_data_df[selet_type[EOP]].values.astype(float))
    t = train_data_df.index
    
    if 'compare_results' not in st.session_state:
        st.session_state['compare_results'] = []
    if 'global_model_index' not in st.session_state:
        st.session_state['global_model_index'] = 1

    # 上下布局：按钮在上，提示在下
    st.markdown('<div style="display:flex;flex-direction:column;align-items:flex-start;width:100%;">', unsafe_allow_html=True)
    clear_compare = st.button('🗑️ 清空所有预报', use_container_width=False, key='clear_compare', type='primary', help='点击后清空所有对比结果',)
    st.markdown('''
        <div style="background:#e6f7ff;padding:14px 16px 14px 16px;border-radius:6px;border-left:4px solid #1890ff;margin-top:8px;width:100%;min-width:220px;">
        <span style="color:#1890ff;font-weight:bold;">提示：</span>每次预测会保留上一次预测的结果以提供对比，点击“清空所有预报”按钮后将清空所有先前的预报结果。
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if clear_compare:
        st.session_state['compare_results'] = []
        st.session_state['global_model_index'] = 1

    # 显示数据图表
    with st.expander("📈 训练数据可视化", expanded=True):
        fig1 = go.Figure(data=go.Scatter(
            x=t,
            y=x,
            mode='lines',
            line=dict(color='royalblue', width=3),
            name='时间序列'
        ))
        fig1.update_layout(
            title=f'原始序列(PM{EOP})',
            xaxis_title='时间',
            yaxis_title='值(mas)',
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # 训练结果区域
    st.subheader("预测结果")

    # 按钮点击后，append 新结果并立即绘制图像和表格
    if mode == "训练新模型" and 'train_button' in globals() and train_button:
        status_text.text("准备训练中...")
        progress_bar.progress(0)
        logs, forecast_results = train_WZPNet(
            module_name,
            start_date,
            end_date,
            EOP,
            pred_len,
            seq_len,
            seq_out,
            num_epoch,
            batch_size,
            progress_bar,
            status_text
        )
        log_container.text_area("训练日志", "\n".join(logs), height=200)
        st.success("训练完成！预测结果如下：")
        last_date = t[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=len(forecast_results),
            freq='D'
        )
        model_label = f"{module_name}-{st.session_state['global_model_index']}"
        st.session_state['compare_results'].append({
            'model': model_label,
            'forecast': forecast_results.copy(),
            'dates': forecast_dates,
        })
        st.session_state['global_model_index'] += 1
        # 绘制图像（包含所有 compare_results，包括刚刚 append 的）
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t[-100:],
            y=x[-100:],
            mode='lines',
            name='历史数据',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_dates,
            y=y,
            mode='lines',
            name='预测区间实测值',
            line=dict(color='blue', dash='dot')
        ))
        color_list = ['red', 'green', 'orange', 'purple', 'brown', 'black']
        for idx, item in enumerate(st.session_state['compare_results']):
            fig2.add_trace(go.Scatter(
                x=item['dates'],
                y=item['forecast'],
                mode='lines+markers',
                name=f"{item['model']} 预测",
                line=dict(color=color_list[idx % len(color_list)], dash='dash', width=2)
            ))
        fig2.update_layout(
            title=f'{EOP} 预测结果',
            xaxis_title='时间',
            yaxis_title='值',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)
        # 显示预测结果表格
        with st.expander("查看详细预测数据", expanded=True):
            forecast_df = pd.DataFrame({
                '日期': forecast_dates,
                '实测值': y,
                '预测值': forecast_results,
                '平均绝对误差(mas)': np.abs(y-forecast_results)*1000
            })
            st.dataframe(forecast_df)
    elif mode == "使用预训练模型" and 'predict_button' in globals() and predict_button:
        status_text.text("准备预测中...")
        progress_bar.progress(0)
        forecast_results = predict_with_pretrained(
            module_name,
            start_date,
            end_date,
            EOP,
            pred_len,
            seq_len,
            seq_out,
            progress_bar,
            status_text
        )
        st.success("预测完成！结果如下：")
        last_date = t[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=len(forecast_results),
            freq='D'
        )
        model_label = f"{module_name}-{st.session_state['global_model_index']}"
        st.session_state['compare_results'].append({
            'model': model_label,
            'forecast': forecast_results.copy(),
            'dates': forecast_dates,
        })
        st.session_state['global_model_index'] += 1
        # 绘制图像（包含所有 compare_results，包括刚刚 append 的）
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t[-100:],
            y=x[-100:],
            mode='lines',
            name='历史数据',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_dates,
            y=y,
            mode='lines',
            name='预测区间实测值',
            line=dict(color='blue', dash='dot')
        ))
        color_list = ['red', 'green', 'orange', 'purple', 'brown', 'black']
        for idx, item in enumerate(st.session_state['compare_results']):
            fig2.add_trace(go.Scatter(
                x=item['dates'],
                y=item['forecast'],
                mode='lines+markers',
                name=f"{item['model']} 预测",
                line=dict(color=color_list[idx % len(color_list)], dash='dash', width=2)
            ))
        fig2.update_layout(
            title=f'{EOP} 预测结果',
            xaxis_title='时间',
            yaxis_title='值',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)
        # 显示预测结果表格
        with st.expander("查看详细预测数据", expanded=True):
            forecast_df = pd.DataFrame({
                '日期': forecast_dates,
                '实测值(mas)': y,
                '预测值(mas)': forecast_results,
                '平均绝对误差(mas)': np.abs(y-forecast_results)*1000
            })
            st.dataframe(forecast_df)
    # 非按钮点击时，只显示历史 compare_results 图像
    elif st.session_state['compare_results']:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t[-100:],
            y=x[-100:],
            mode='lines',
            name='历史数据',
            line=dict(color='blue')
        ))
        # 取最后一次预测的 forecast_dates 和 y
        last_item = st.session_state['compare_results'][-1]
        forecast_dates = last_item['dates']
        fig2.add_trace(go.Scatter(
            x=forecast_dates,
            y=y,
            mode='lines',
            name='预测区间实测值',
            line=dict(color='blue', dash='dot')
        ))
        color_list = ['red', 'green', 'orange', 'purple', 'brown', 'black']
        for idx, item in enumerate(st.session_state['compare_results']):
            fig2.add_trace(go.Scatter(
                x=item['dates'],
                y=item['forecast'],
                mode='lines+markers',
                name=f"{item['model']} 预测",
                line=dict(color=color_list[idx % len(color_list)], dash='dash', width=2)
            ))
        fig2.update_layout(
            title=f'{EOP} 预测结果',
            xaxis_title='时间',
            yaxis_title='值',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 底部信息
st.markdown("""
<div class="footer">
    极移预报系统 v1.0 &copy; 2025 中山大学人工智能学院 空天智能团队 | 数据最后更新: 2024-12-31
</div>
""", unsafe_allow_html=True)