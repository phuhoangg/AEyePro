import pandas as pd
import plotly.graph_objs as go

class ChartManager:
    """
    Quản lý xử lý dữ liệu và tạo các biểu đồ Plotly đơn giản từ DataFrame đã truyền vào.
    Hỗ trợ lọc ngày/giờ, trả về Figure object và nội dung bảng/ghi chú cho summary_view.
    Không tự đọc file, chỉ nhận dữ liệu đầu vào từ module khác.
    """
    @staticmethod
    def clean_data(df):
        df = df.copy()  # Tạo bản sao để tránh chỉnh sửa DataFrame gốc
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].mean())  # Gán trực tiếp thay vì inplace
        return df

    @staticmethod
    def filter_data(df, date_filter=None, time_range=None):
        if df.empty:
            return df
        if date_filter:
            df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == date_filter]
        if time_range:
            start, end = time_range
            df = df[(df['timestamp'].dt.strftime('%H:%M:%S') >= start) & (df['timestamp'].dt.strftime('%H:%M:%S') <= end)]
        return df

    @staticmethod
    def create_summary_bar_chart(df_summary, columns=['blink_per_minute', 'number_of_drowsiness', 'bad_posture_count'], date_filter=None):
        if df_summary.empty:
            return go.Figure()
        data = df_summary.copy()
        if date_filter and 'begin_timestamp' in data.columns:
            data = data[pd.to_datetime(data['begin_timestamp']).dt.strftime('%Y-%m-%d') == date_filter]
        fig = go.Figure()
        colors = ['#1E88E5', '#D32F2F', '#B0BEC5']
        for col, color in zip(columns, colors):
            if col in data.columns:
                fig.add_trace(go.Bar(
                    x=data['session_id'],
                    y=data[col],
                    name=col,
                    marker_color=color
                ))
        fig.update_layout(
            barmode='group',
            title={'text': 'So sánh các phiên', 'font': {'size': 18}},
            font=dict(family='Arial', size=14),
            plot_bgcolor='#B0BEC5',
            paper_bgcolor='#B0BEC5',
            legend_title_text='Chỉ số',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def create_session_line_chart(df_realtime, session_id, x='timestamp', y=['blink_count', 'distance', 'head_tilt', 'head_side', 'shoulder_angle'], date_filter=None, time_range=None):
        df = df_realtime[df_realtime['session_id'] == session_id].copy()
        df = ChartManager.filter_data(df, date_filter, time_range)
        if df.empty:
            return go.Figure()
        fig = go.Figure()
        colors = ['#1E88E5', '#43A047', '#FFEB3B', '#3949AB', '#FF9800']
        for idx, col in enumerate(y):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x], y=df[col], mode='lines+markers', name=col, line=dict(color=colors[idx % len(colors)])))
        fig.update_layout(
            title={'text': f'Chỉ số theo thời gian - Session {session_id}', 'font': {'size': 18}},
            font=dict(family='Arial', size=14),
            plot_bgcolor='#B0BEC5',
            paper_bgcolor='#B0BEC5',
            legend_title_text='Chỉ số',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def create_session_scatter_chart(df_realtime, session_id, x='timestamp', y='blink_count', color_by='drowsiness_detected', date_filter=None, time_range=None):
        df = df_realtime[df_realtime['session_id'] == session_id].copy()
        df = ChartManager.filter_data(df, date_filter, time_range)
        if df.empty:
            return go.Figure()
        colors = df[color_by].map(lambda v: '#D32F2F' if v==1 else '#1E88E5') if color_by in df.columns else '#1E88E5'
        marker_symbol = df.get('posture_status', '').map(lambda v: 'diamond' if v=='poor' else 'circle') if 'posture_status' in df.columns else 'circle'
        fig = go.Figure(data=go.Scatter(
            x=df[x], y=df[y],
            mode='markers',
            marker=dict(color=colors, symbol=marker_symbol, size=10),
            text=[f"Drowsy: {d}" for d in df.get('drowsiness_detected', [0]*len(df))],
            hoverinfo='text+x+y'))
        fig.update_layout(
            title={'text': f'Phân tán số lần chớp mắt - Session {session_id}', 'font': {'size': 18}},
            font=dict(family='Arial', size=14),
            plot_bgcolor='#B0BEC5',
            paper_bgcolor='#B0BEC5',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def create_highlight_chart(df_realtime, session_id, x='timestamp', y='distance', color_by='anomaly', date_filter=None, time_range=None):
        df = df_realtime[df_realtime['session_id'] == session_id].copy()
        df = ChartManager.filter_data(df, date_filter, time_range)
        if df.empty:
            return go.Figure()
        colors = df[color_by].map(lambda v: '#D32F2F' if v else '#1E88E5') if color_by in df.columns else '#1E88E5'
        fig = go.Figure(data=go.Scatter(
            x=df[x], y=df[y],
            mode='markers+lines',
            marker=dict(color=colors, size=10),
            hoverinfo='x+y'))
        fig.update_layout(
            title={'text': f'Khoảng cách & điểm bất thường - Session {session_id}', 'font': {'size': 18}},
            font=dict(family='Arial', size=14),
            plot_bgcolor='#B0BEC5',
            paper_bgcolor='#B0BEC5',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def get_charts(df_summary, df_realtime, session_id=None, date_filter=None, time_range=None):
        results = []
        if session_id:
            results.append(ChartManager.create_session_line_chart(df_realtime, session_id, date_filter=date_filter, time_range=time_range))
            results.append(ChartManager.create_session_scatter_chart(df_realtime, session_id, date_filter=date_filter, time_range=time_range))
            results.append(ChartManager.create_highlight_chart(df_realtime, session_id, date_filter=date_filter, time_range=time_range))
            notes = []
            df = df_realtime[df_realtime['session_id'] == session_id]
            df = ChartManager.filter_data(df, date_filter, time_range)
            if not df.empty:
                if (df['distance'] > 100).any():
                    t = df[df['distance'] > 100]['timestamp'].iloc[0]
                    notes.append(f"Khoảng cách tăng đột ngột lên {df['distance'].max():.1f}cm vào {t.strftime('%H:%M:%S')}")
                drowsy_seq = (df['drowsiness_detected'] == 1).astype(int).groupby((df['drowsiness_detected'] != 1).cumsum()).cumsum()
                if (drowsy_seq > 5).any():
                    t1 = df[drowsy_seq > 5]['timestamp'].iloc[0]
                    t2 = df[drowsy_seq > 5]['timestamp'].iloc[-1]
                    notes.append(f"Chuỗi buồn ngủ kéo dài từ {t1.strftime('%H:%M:%S')} đến {t2.strftime('%H:%M:%S')}")
            results.append("\n".join(notes))
        else:
            results.append(ChartManager.create_summary_bar_chart(df_summary, date_filter=date_filter))
            results.append(df_summary)
        return results