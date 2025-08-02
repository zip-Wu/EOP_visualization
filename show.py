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
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="header-style">欢迎进入极移预报系统</div>', unsafe_allow_html=True)

# 训练模型函数
def train_WZPNet(start_time, end_time, EOP, pred_len, seq_len, seq_out, d_model, dropout,
                 seq_ar, seq_cnn, cnn_kernel, cnn_stride, cnn_channel, seq_gru, gru_layer, gru_hidden,
                 skip_num, skip_len, skip_layer, skip_hidden, num_epoch, batch_size, progress_bar, status_text):
    
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
                seq_cnn=seq_cnn, cnn_kernel=cnn_kernel, cnn_stride=cnn_stride, cnn_channel=cnn_channel,
                seq_gru=seq_gru, gru_layer=gru_layer, gru_hidden=gru_hidden,
                skip_num=skip_num, skip_len=skip_len, skip_layer=skip_layer, skip_hidden=skip_hidden).to(device)
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
        status_text.text(f"训练中... 轮次 {epoch+1}/{num_epoch}")
        
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

# 侧边栏参数选择区域
with st.sidebar:
    st.markdown('<div class="sidebar-title">参数配置</div>', unsafe_allow_html=True)
    
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
    
    # 神经网络模块选择
    st.markdown('<div class="date-section-title">神经网络模块</div>', unsafe_allow_html=True)
    
    # 使用多选下拉框选择模块
    selected_modules = st.multiselect(
        "选择要使用的模块:",
        options=["Linear", "CNN", "GRU", "skipGRU"],
        default=["Linear", "skipGRU"],
        help="选择要启用的神经网络模块"
    )
    
    # 初始化模块参数
    seq_ar = 0
    seq_cnn = 0
    cnn_kernel = 0
    cnn_stride = 0
    cnn_channel = 0
    seq_gru = 0
    gru_layer = 0
    gru_hidden = 0
    skip_num = 0
    skip_len = 0
    skip_layer = 0
    skip_hidden = 0
    
    # 为每个选中的模块创建可折叠的参数区域
    if selected_modules:
        st.markdown('<div class="module-section">', unsafe_allow_html=True)
        st.markdown('<div class="module-header">模块参数配置</div>', unsafe_allow_html=True)
        
        # Linear模块参数
        if "Linear" in selected_modules:
            with st.expander("Linear 参数", expanded=True):
                seq_ar = st.number_input("AR序列长度 (seq_ar)", 
                                        min_value=0, max_value=500, value=256, step=1,
                                        key="seq_ar")
        
        # CNN模块参数
        if "CNN" in selected_modules:
            with st.expander("CNN 参数", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    seq_cnn = st.number_input("CNN序列长度 (seq_cnn)", 
                                            min_value=1, max_value=500, value=256, step=1,
                                            key="seq_cnn")
                    cnn_kernel = st.number_input("CNN核大小 (cnn_kernel)", 
                                               min_value=1, max_value=10, value=4, step=1,
                                               key="cnn_kernel")
                with col2:
                    cnn_stride = st.number_input("CNN步幅 (cnn_stride)", 
                                               min_value=1, max_value=5, value=1, step=1,
                                               key="cnn_stride")
                    cnn_channel = st.number_input("CNN通道数 (cnn_channel)", 
                                                min_value=1, max_value=64, value=1, step=1,
                                                key="cnn_channel")
        
        # GRU模块参数
        if "GRU" in selected_modules:
            with st.expander("GRU 参数", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    seq_gru = st.number_input("GRU序列长度 (seq_gru)", 
                                            min_value=1, max_value=500, value=256, step=1,
                                            key="seq_gru")
                    gru_layer = st.number_input("GRU层数 (gru_layer)", 
                                             min_value=1, max_value=5, value=1, step=1,
                                             key="gru_layer")
                with col2:
                    gru_hidden = st.number_input("GRU隐藏层大小 (gru_hidden)", 
                                               min_value=16, max_value=256, value=64, step=16,
                                               key="gru_hidden")
        
        # skipGRU模块参数
        if "skipGRU" in selected_modules:
            with st.expander("skipGRU 参数", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    seq_skip = st.number_input("SkipGRU输入序列长度 (seq_skip)", 
                                             min_value=1, max_value=500, value=200, step=1,
                                             key="seq_skip")
                    skip_len = st.number_input("SkipGRU跨度 (skip_len)", 
                                             min_value=1, max_value=10, value=3, step=1,
                                             key="skip_len")
                with col2:
                    skip_layer = st.number_input("SkipGRU层数 (skip_layer)", 
                                               min_value=1, max_value=5, value=1, step=1,
                                               key="skip_layer")
                    skip_hidden = st.number_input("SkipGRU隐藏层大小 (skip_hidden)", 
                                                min_value=16, max_value=256, value=64, step=16,
                                                key="skip_hidden")
            skip_num = seq_skip // skip_len
        
        
        st.markdown('</div>', unsafe_allow_html=True)  # 关闭模块部分
    
    # 预测参数
    st.markdown('<div class="date-section-title">预测参数</div>', unsafe_allow_html=True)
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        pred_len = st.number_input("预报长度 (pred_len)", 
                                  min_value=100, max_value=500, value=360, step=10,
                                  key="pred_len")
    with col_pred2:
        seq_len = st.number_input("序列长度 (seq_len)", 
                                min_value=100, max_value=500, value=256, step=10,
                                key="seq_len")
    seq_out = st.number_input("单次输出长度 (seq_out)", 
                            min_value=1, max_value=100, value=20, step=1,
                            key="seq_out")
    
    # 公共参数
    st.markdown('<div class="date-section-title">公共参数</div>', unsafe_allow_html=True)
    col_common1, col_common2 = st.columns(2)
    with col_common1:
        d_model = st.number_input("模型维度 (d_model)", 
                                min_value=16, max_value=256, value=64, step=16,
                                key="d_model")
    with col_common2:
        dropout = st.number_input("Dropout率", 
                                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                format="%.2f",
                                key="dropout")
    
    # 训练参数
    st.markdown('<div class="date-section-title">训练参数</div>', unsafe_allow_html=True)
    col_train1, col_train2 = st.columns(2)
    with col_train1:
        num_epoch = st.number_input("训练轮数 (num_epoch)", 
                                  min_value=1, max_value=100, value=20, step=1,
                                  key="num_epoch")
    with col_train2:
        batch_size = st.number_input("批大小 (batch_size)", 
                                  min_value=8, max_value=128, value=64, step=8,
                                  key="batch_size")
    
    # 开始训练按钮
    train_button = st.button("开始训练", use_container_width=True, key="train_button")


# 主内容区域
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # 显示当前选择的信息
    st.success(f"已选择参数: {EOP}, 数据范围: {start_date} 至 {end_date}")
    
    # 显示模块选择状态
    modules = selected_modules if selected_modules else ["无"]
    st.info(f"当前启用的模块: {', '.join(modules)}")
    
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
    st.subheader("训练与预测结果")
    
    if train_button:
        # 准备训练
        status_text.text("准备训练中...")
        progress_bar.progress(0)
        
        # 调用训练函数
        logs, forecast_results = train_WZPNet(
            start_date, end_date, EOP, pred_len, seq_len, seq_out, d_model, dropout,
            seq_ar, seq_cnn, cnn_kernel, cnn_stride, cnn_channel, 
            seq_gru, gru_layer, gru_hidden,
            skip_num, skip_len, skip_layer, skip_hidden, 
            num_epoch, batch_size,
            progress_bar, status_text
        )
        
        # 显示训练日志
        log_container.text_area("训练日志", "\n".join(logs), height=200)
        
        # 显示预测结果
        st.success("训练完成！预测结果如下：")
        
        # 创建预测结果的时间序列
        last_date = t[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=len(forecast_results),
            freq='D'
        )
        
        # 绘制预测结果
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t[-100:],  # 显示最后100个真实值
            y=x[-100:],
            mode='lines',
            name='历史数据',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_results,
            mode='lines+markers',
            name='预测结果',
            line=dict(color='red', dash='dash')
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
        with st.expander("查看详细预测数据"):
            forecast_df = pd.DataFrame({
                '日期': forecast_dates,
                '实测值': y,
                '预测值': forecast_results,
                '平均绝对误差(mas)': np.abs(y-forecast_results)*1000
            })
            st.dataframe(forecast_df)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 底部信息
st.markdown("""
<div class="footer">
    极移预报系统 v1.0 &copy; 2025 中山大学人工智能学院 空天智能团队 | 数据最后更新: 2024-12-31
</div>
""", unsafe_allow_html=True)