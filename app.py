import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import io
import matplotlib.font_manager as fm
import traceback

# --- 한글 폰트 설정 ---
try:
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_name = 'NanumGothic' if any('NanumGothic' in f for f in font_files) else 'Malgun Gothic'
except Exception:
    font_name = 'sans-serif'

# --- 차트 스타일 설정 ---
CHART_COLORS = {
    'primary': '#3B82F6', 'secondary': '#10B981', 'accent': '#F59E0B',
    'danger': '#EF4444', 'success': '#22C55E', 'warning': '#F59E0B',
    'info': '#06B6D4', 'neutral': '#6B7280'
}

PLOTLY_THEME = {
    'layout': {
        'font': {'family': font_name, 'size': 12, 'color': '#374151'},
        'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
        'title': {'font': {'size': 16, 'color': '#111827'}, 'x': 0.02, 'xanchor': 'left'},
        'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40},
        'showlegend': True,
        'legend': {
            'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02,
            'xanchor': 'left', 'x': 0, 'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': '#E5E7EB', 'borderwidth': 1
        },
        'xaxis': {'gridcolor': '#F3F4F6', 'zerolinecolor': '#E5E7EB', 'title': {'font': {'size': 11, 'color': '#6B7280'}}},
        'yaxis': {'gridcolor': '#F3F4F6', 'zerolinecolor': '#E5E7EB', 'title': {'font': {'size': 11, 'color': '#6B7280'}}}
    }
}

# --- 데이터 처리 함수들 ---
def get_week_of_month(dt):
    try:
        first_day = dt.replace(day=1); dom = dt.day; adjusted_dom = dom + first_day.weekday()
        return int(np.ceil(adjusted_dom / 7.0))
    except:
        return 1

def process_data(data_file, mapping_dict, date_format_code, date_format_custom):
    if data_file is None: return None
    try:
        file_path = data_file.name
        df = pd.read_csv(file_path, dtype=str) if file_path.endswith('.csv') else pd.read_excel(file_path, dtype=str)
        processed_df = pd.DataFrame()
        date_series = df[mapping_dict['date']]
        date_format = date_format_custom if date_format_code == 'custom' else date_format_code
        processed_df['date'] = pd.to_datetime(date_series, format=date_format, errors='coerce') if date_format else pd.to_datetime(date_series, errors='coerce')
        processed_df.dropna(subset=['date'], inplace=True)
        if processed_df.empty: raise gr.Error("유효한 날짜 데이터를 찾을 수 없습니다.")
        valid_indices = processed_df.index
        df_filtered = df.iloc[valid_indices].copy()
        
        def clean_numeric(series):
            return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

        for key, col in mapping_dict.items():
            if key not in ['date', 'channel']:
                processed_df[key] = clean_numeric(df_filtered[col])
        processed_df['channel'] = df_filtered[mapping_dict['channel']].astype(str)
        processed_df.fillna(0, inplace=True)
        
        processed_df['month'] = processed_df['date'].dt.strftime('%Y-%m')
        processed_df['week'] = processed_df.apply(lambda row: get_week_of_month(row['date']), axis=1)
        return processed_df
    except Exception as e:
        raise gr.Error(f"파일 처리 중 오류 발생: {e}")

def aggregate_data(df):
    if df is None or df.empty: return None
    try:
        agg_metrics = {'cost': 'sum', 'conversions': 'sum', 'revenue': 'sum', 'impressions': 'sum', 'clicks': 'sum'}
        
        def calculate_efficiency(df_agg):
            df_agg = df_agg.copy()
            df_agg['cpa'] = np.where(df_agg['conversions'] > 0, df_agg['cost'] / df_agg['conversions'], 0)
            df_agg['roas'] = np.where(df_agg['cost'] > 0, (df_agg['revenue'] / df_agg['cost']) * 100, 0)
            df_agg['ctr'] = np.where(df_agg['impressions'] > 0, (df_agg['clicks'] / df_agg['impressions']) * 100, 0)
            df_agg['cvr'] = np.where(df_agg['clicks'] > 0, (df_agg['conversions'] / df_agg['clicks']) * 100, 0)
            return df_agg

        overall = df.agg(agg_metrics)
        overall = calculate_efficiency(pd.DataFrame(overall).T).iloc[0]
        by_channel = calculate_efficiency(df.groupby('channel').agg(agg_metrics).reset_index())
        by_week = calculate_efficiency(df.groupby(['month', 'week']).agg(agg_metrics).reset_index())
        by_week['week_str'] = by_week.apply(lambda r: f"{int(r['month'].split('-')[1])}월 {r['week']}주차", axis=1)
        by_week = by_week.sort_values(['month', 'week'])
        by_day = calculate_efficiency(df.groupby('date').agg(agg_metrics).reset_index())
        return {'overall': overall, 'by_channel': by_channel, 'by_week': by_week, 'by_day': by_day}
    except Exception as e:
        print(f"Error in aggregate_data: {e}\n{traceback.format_exc()}")
        return None

def create_plots(aggregated_data, kpi_type, target_value):
    if not aggregated_data: return None, None, None
    try:
        by_channel = aggregated_data['by_channel']
        by_week = aggregated_data['by_week']

        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(x=by_week['week_str'], y=by_week['cost'], name='광고비', marker_color=CHART_COLORS['primary'], opacity=0.8, hovertemplate='<b>%{x}</b><br>광고비: ₩%{y:,.0f}<extra></extra>'), secondary_y=False)
        fig1.add_trace(go.Scatter(x=by_week['week_str'], y=by_week[kpi_type], name=kpi_type.upper(), mode='lines+markers', line=dict(color=CHART_COLORS['secondary'], width=3), marker=dict(size=8, color=CHART_COLORS['secondary']), hovertemplate=f'<b>%{{x}}</b><br>{kpi_type.upper()}: %{{y:.2f}}<extra></extra>'), secondary_y=True)
        fig1.update_layout(title_text="📈 주간 성과 트렌드", **PLOTLY_THEME['layout']); fig1.update_yaxes(title_text="광고비 (₩)", secondary_y=False); fig1.update_yaxes(title_text=f"{kpi_type.upper()}", secondary_y=True)

        fig2 = px.pie(by_channel, values='cost', names='channel', title='💰 채널별 비용 비중', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig2.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>비용: ₩%{value:,.0f}<br>비중: %{percent}<extra></extra>')
        fig2.update_layout(**PLOTLY_THEME['layout'])
        
        colors = [CHART_COLORS['danger'] if (kpi_type == 'cpa' and c > target_value and target_value > 0) or (kpi_type == 'roas' and c < target_value and target_value > 0) else CHART_COLORS['success'] for c in by_channel[kpi_type]]
        fig3 = px.bar(by_channel.sort_values(by=kpi_type, ascending=(kpi_type=='cpa')), x=kpi_type, y='channel', orientation='h', title=f'🎯 채널별 {kpi_type.upper()} 비교', text=kpi_type)
        fig3.update_traces(marker_color=colors, texttemplate='%{text:,.0f}' if kpi_type == 'cpa' else '%{text:.1f}%', textposition='outside', hovertemplate=f'<b>%{{y}}</b><br>{kpi_type.upper()}: %{{x:.2f}}<extra></extra>')
        fig3.update_layout(**PLOTLY_THEME['layout'], yaxis={'categoryorder': 'total ascending' if kpi_type == 'roas' else 'total descending'})
        return fig1, fig2, fig3
    except Exception as e:
        print(f"Error in create_plots: {e}\n{traceback.format_exc()}")
        return None, None, None

def create_kpi_html(overall, kpi_type, target_cpa, target_roas, prev_month_overall):
    try:
        def get_trend_html(current, prev, kpi_name, vs_text):
            if prev is None or pd.isna(prev) or prev == 0: return "<div class='kpi-trend-placeholder'>-</div>"
            diff_pct = (current - prev) / prev * 100
            is_bad = (kpi_name in ['cpa', 'cost'] and diff_pct > 0) or (kpi_name not in ['cpa', 'cost'] and diff_pct < 0)
            color = CHART_COLORS['danger'] if is_bad else CHART_COLORS['success']
            arrow = "↗" if diff_pct > 0 else "↘"
            icon = "📉" if is_bad else "📈"
            bar_html = f"<div class='progress-container'><div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {min(100, abs(diff_pct) * 1.5)}%; background: linear-gradient(90deg, {color}, {color}80);'></div></div></div>" if vs_text == "KPI 대비" else ""
            return f"<div class='kpi-trend'><span style='color: {color}; font-weight: 600;'>{icon} {arrow} {abs(diff_pct):.1f}%</span> <div class='vs-text'>vs {vs_text}</div></div>{bar_html}"

        def format_card(title, value, trend_html="<div class='kpi-trend-placeholder'>-</div>", icon="📊"):
            return f"<div class='kpi-card'><div class='kpi-header'><span class='kpi-icon'>{icon}</span><p class='kpi-title'>{title}</p></div><p class='kpi-value'>{value}</p>{trend_html}</div>"

        cost_trend = get_trend_html(overall['cost'], prev_month_overall['cost'], 'cost', "전월 대비") if prev_month_overall is not None else ""
        ctr_trend = get_trend_html(overall['ctr'], prev_month_overall['ctr'], 'ctr', "전월 대비") if prev_month_overall is not None else ""
        cvr_trend = get_trend_html(overall['cvr'], prev_month_overall['cvr'], 'cvr', "전월 대비") if prev_month_overall is not None else ""
        conv_trend = get_trend_html(overall['conversions'], prev_month_overall['conversions'], 'conversions', "전월 대비") if prev_month_overall is not None else ""
        roas_trend = get_trend_html(overall['roas'], prev_month_overall['roas'], 'roas', "전월 대비") if prev_month_overall is not None else ""
        cpa_trend = get_trend_html(overall['cpa'], prev_month_overall['cpa'], 'cpa', "전월 대비") if prev_month_overall is not None else ""
        if kpi_type == 'roas': roas_trend = get_trend_html(overall['roas'], target_roas, 'roas', "KPI 대비")
        if kpi_type == 'cpa': cpa_trend = get_trend_html(overall['cpa'], target_cpa, 'cpa', "KPI 대비")
        
        cards = [
            format_card("총 광고비", f"₩{overall['cost']:,.0f}", cost_trend, "💰"), format_card("클릭률 (CTR)", f"{overall['ctr']:.2f}%", ctr_trend, "👆"),
            format_card("전환율 (CVR)", f"{overall['cvr']:.2f}%", cvr_trend, "🎯"), format_card("총 전환수", f"{overall['conversions']:,.0f}건", conv_trend, "✅"),
            format_card("ROAS", f"{overall['roas']:.2f}%", roas_trend, "📊"), format_card("CPA", f"₩{overall['cpa']:,.0f}", cpa_trend, "💸"),
        ]
        return f"<div class='kpi-dashboard'><div class='kpi-grid'>{''.join(cards)}</div></div>"
    except Exception as e:
        print(f"Error in create_kpi_html: {e}\n{traceback.format_exc()}")
        return "<div>KPI 데이터 생성 중 오류 발생</div>"

def get_previous_month_str(month_str):
    try:
        year, month = map(int, month_str.split('-')); return f"{year-1}-12" if month == 1 else f"{year}-{month-1:02d}"
    except: return None

# --- Gradio 상호작용 함수들 ---
def show_mapping_ui(data_file):
    if data_file is None: return gr.update(visible=False), *(gr.update() for _ in range(8))
    try:
        df = pd.read_csv(data_file.name, nrows=1, dtype=str) if data_file.name.endswith('.csv') else pd.read_excel(data_file.name, nrows=1, dtype=str)
        headers = df.columns.tolist()
        auto_map = {'date': next((h for h in headers if '날짜' in h or 'date' in h.lower()), None), 'cost': next((h for h in headers if '비용' in h or 'cost' in h.lower()), None), 'impressions': next((h for h in headers if '노출' in h or 'imp' in h.lower()), None), 'clicks': next((h for h in headers if '클릭' in h or 'click' in h.lower()), None), 'conversions': next((h for h in headers if '전환' in h or 'conv' in h.lower()), None), 'channel': next((h for h in headers if '채널' in h or 'channel' in h.lower()), None), 'revenue': next((h for h in headers if '매출' in h or 'revenue' in h.lower()), None)}
        return gr.update(visible=True), gr.update(choices=headers, value=auto_map['date']), gr.update(choices=headers, value=auto_map['cost']), gr.update(choices=headers, value=auto_map['impressions']), gr.update(choices=headers, value=auto_map['clicks']), gr.update(choices=headers, value=auto_map['conversions']), gr.update(choices=headers, value=auto_map['channel']), gr.update(choices=headers, value=auto_map['revenue']), gr.update(visible=False)
    except Exception as e:
        print(f"Error reading headers: {e}"); return gr.update(visible=False), *(gr.update() for _ in range(8))

def update_dashboard_display(df_full_json, month_filter, channel_filter, kpi_type, target_cpa, target_roas):
    if df_full_json is None: return [None] * 8
    try:
        df_full = pd.read_json(io.StringIO(df_full_json), orient='split'); df_full['date'] = pd.to_datetime(df_full['date'], unit='ms')
        df_current = df_full.copy()
        if month_filter != "전체 월": df_current = df_current[df_current['month'] == month_filter]
        if channel_filter != "전체 매체": df_current = df_current[df_current['channel'] == channel_filter]
        
        aggregated_current = aggregate_data(df_current)
        if not aggregated_current: return [None] * 8
        
        prev_month_overall = None
        sorted_months = sorted(df_full['month'].unique())
        if month_filter != "전체 월" and month_filter in sorted_months and month_filter != sorted_months[0]:
            prev_month_str = get_previous_month_str(month_filter)
            if prev_month_str:
                df_prev_month = df_full[df_full['month'] == prev_month_str]
                if not df_prev_month.empty:
                    if channel_filter != "전체 매체": df_prev_month = df_prev_month[df_prev_month['channel'] == channel_filter]
                    aggregated_prev = aggregate_data(df_prev_month)
                    if aggregated_prev: prev_month_overall = aggregated_prev['overall']

        kpi_html = create_kpi_html(aggregated_current['overall'], kpi_type, target_cpa, target_roas, prev_month_overall)
        summary = f"🎯 총 광고비 ₩{aggregated_current['overall']['cost']:,.0f}으로 {aggregated_current['overall']['conversions']:,.0f}건의 전환을 달성했습니다."
        plot1, plot2, plot3 = create_plots(aggregated_current, kpi_type, target_cpa if kpi_type == 'cpa' else target_roas)
        
        wk_cols = ['week_str', 'cost', 'impressions', 'clicks', 'conversions', 'ctr', 'cvr', 'cpa', 'roas']; day_cols = ['date', 'cost', 'impressions', 'clicks', 'conversions', 'ctr', 'cvr', 'cpa', 'roas']
        wk_rename = {'week_str':'주차','cost':'비용','impressions':'노출','clicks':'클릭','conversions':'전환', 'ctr':'CTR(%)', 'cvr':'CVR(%)', 'cpa':'CPA', 'roas':'ROAS(%)'}; day_rename = {'date':'날짜','cost':'비용','impressions':'노출','clicks':'클릭','conversions':'전환', 'ctr':'CTR(%)', 'cvr':'CVR(%)', 'cpa':'CPA', 'roas':'ROAS(%)'}
        
        by_week_df = aggregated_current['by_week'][wk_cols].rename(columns=wk_rename)
        by_day_df = aggregated_current['by_day'][day_cols].rename(columns=day_rename)
        
        for col in ['비용', '노출', '클릭', '전환', 'CPA']:
            by_week_df[col] = by_week_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '0')
            by_day_df[col] = by_day_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '0')
        for col in ['CTR(%)', 'CVR(%)', 'ROAS(%)']:
            by_week_df[col] = by_week_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '0.00')
            by_day_df[col] = by_day_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '0.00')
        by_day_df['날짜'] = by_day_df['날짜'].dt.strftime('%Y-%m-%d')
        return kpi_html, summary, plot1, plot2, by_week_df, by_day_df, plot3
    except Exception as e:
        print(f"Error in update_dashboard_display: {e}\n{traceback.format_exc()}")
        return [f"대시보드 업데이트 중 오류: {e}"] + [None] * 7

def process_and_init_dashboard(data_file, date_col, cost_col, imp_col, click_col, conv_col, channel_col, rev_col, date_format_code, date_format_custom):
    try:
        mapping_dict = {'date': date_col, 'cost': cost_col, 'impressions': imp_col, 'clicks': click_col, 'conversions': conv_col, 'channel': channel_col, 'revenue': rev_col}
        if not all(mapping_dict[k] for k in ['date', 'cost', 'impressions', 'clicks', 'conversions', 'channel']): raise gr.Error("필수 컬럼을 모두 지정해야 합니다.")
        df_full = process_data(data_file, mapping_dict, date_format_code, date_format_custom)
        df_full_json = df_full.to_json(orient='split', date_format='iso')
        months = ["전체 월"] + sorted(df_full['month'].unique(), reverse=True)
        channels = ["전체 매체"] + sorted(df_full['channel'].unique())
        dashboard_updates = update_dashboard_display(df_full_json, "전체 월", "전체 매체", "cpa", 95000, 450)
        return df_full_json, gr.update(visible=True), gr.update(choices=months, value="전체 월"), gr.update(choices=channels, value="전체 매체"), *dashboard_updates
    except Exception as e:
        print(f"Error in process_and_init_dashboard: {e}\n{traceback.format_exc()}")
        return None, gr.update(visible=False), gr.update(choices=[]), gr.update(choices=[]), f"처리 중 오류: {e}", *(None for _ in range(7))

# --- CSS 스타일링 ---
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
:root {
    --primary-color: #3B82F6; --secondary-color: #10B981; --accent-color: #F59E0B;
    --danger-color: #EF4444; --success-color: #22C55E; --neutral-color: #6B7280;
    --background-color: #F8FAFC; --surface-color: #FFFFFF; --border-color: #E5E7EB;
    --text-primary: #111827; --text-secondary: #6B7280;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}
body, .gradio-container { font-family: 'Noto Sans KR', sans-serif !important; background-color: var(--background-color) !important; }
.gradio-container h1 {
    color: var(--text-primary) !important; font-weight: 700 !important; font-size: 2.25rem !important; margin-bottom: 0.25rem !important;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.gradio-container h1 + p { color: var(--text-secondary) !important; font-size: 1rem !important; margin-bottom: 2rem !important; }
.kpi-dashboard { margin-bottom: 1.5rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; }
.kpi-card {
    background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.5rem;
    box-shadow: var(--shadow-sm); transition: all 0.3s ease; position: relative; overflow: hidden;
}
.kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); opacity: 0.8; }
.kpi-header { display: flex; align-items: center; margin-bottom: 1rem; }
.kpi-icon { font-size: 1.25rem; margin-right: 0.75rem; }
.kpi-title { color: var(--text-secondary); font-size: 0.875rem; font-weight: 500; margin: 0; }
.kpi-value { color: var(--text-primary); font-size: 2.25rem; font-weight: 700; line-height: 1; margin: 0 0 0.5rem 0; }
.kpi-trend { font-size: 0.875rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
.vs-text { color: var(--text-secondary); }
.kpi-trend-placeholder { color: #9ca3af; font-style: italic; height: 38px; display: flex; align-items: center; justify-content: center; }
.progress-container { width: 100%; height: 1.25rem; display: flex; flex-direction: column; justify-content: flex-end; margin-top: 0.25rem; }
.progress-bar-bg { background-color: #e5e7eb; border-radius: 9999px; height: 6px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 9999px; transition: width 0.3s ease; }
"""

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=css, title="광고 성과 분석 대시보드") as app:
    gr.Markdown("<h1>광고 성과 분석 대시보드</h1><p>성과를 분석할 광고 데이터 CSV 또는 XLSX 파일을 업로드해주세요.</p>")
    df_state = gr.State()

    with gr.Row():
        file_input = gr.File(label="데이터 파일 업로드", file_types=[".csv", ".xlsx"], scale=1)
    
    with gr.Group(visible=False) as mapping_group:
        gr.Markdown("### 데이터 컬럼 매핑")
        with gr.Row():
            date_col = gr.Dropdown(label="날짜 컬럼", scale=2); date_format_selector = gr.Dropdown(label="날짜 형식", choices=[("자동인식", ""), ("2024.12.31", "%Y.%m.%d"), ("24. 12. 31.", "%y. %m. %d."), ("12/31/2024", "%m/%d/%Y"), ("직접 입력", "custom")], value="", scale=2); date_format_custom = gr.Textbox(label="사용자 정의 날짜 형식", placeholder="%Y-%m-%d", visible=False, scale=1)
        with gr.Row():
            cost_col = gr.Dropdown(label="비용 컬럼"); imp_col = gr.Dropdown(label="노출수 컬럼"); click_col = gr.Dropdown(label="클릭수 컬럼")
        with gr.Row():
            conv_col = gr.Dropdown(label="전환 컬럼"); channel_col = gr.Dropdown(label="채널 컬럼"); rev_col = gr.Dropdown(label="매출 컬럼 (선택)")
        analyze_button = gr.Button("분석 시작", variant="primary")

    with gr.Column(visible=False) as dashboard_group:
        with gr.Blocks():
            with gr.Row():
                month_filter = gr.Dropdown(label="월 선택"); channel_filter = gr.Dropdown(label="매체 선택")
                with gr.Column(min_width=300):
                    kpi_type = gr.Radio(label="목표 KPI", choices=["cpa", "roas"], value="cpa", interactive=True)
                    target_cpa = gr.Number(label="목표 CPA", value=95000, interactive=True)
                    target_roas = gr.Number(label="목표 ROAS", value=450, visible=False, interactive=True)
            
            kpi_output_md = gr.Markdown()
            summary_output = gr.Textbox(label="성과 요약", lines=2, interactive=False)
            with gr.Row():
                plot_weekly = gr.Plot(label="주간 성과 그래프")
            with gr.Row():
                plot_channel_cost = gr.Plot(label="채널별 비용 비중"); plot_channel_kpi = gr.Plot(label="채널별 KPI 비교")
            with gr.Row():
                df_weekly = gr.Dataframe(label="주간별 상세 데이터", interactive=False); df_daily = gr.Dataframe(label="일별 상세 데이터", interactive=False)

    file_input.upload(show_mapping_ui, inputs=file_input, outputs=[mapping_group, date_col, cost_col, imp_col, click_col, conv_col, channel_col, rev_col, dashboard_group])
    def toggle_custom_format(choice): return gr.update(visible=(choice == "custom"))
    date_format_selector.change(toggle_custom_format, inputs=date_format_selector, outputs=date_format_custom)

    dashboard_components = [kpi_output_md, summary_output, plot_weekly, plot_channel_cost, df_weekly, df_daily, plot_channel_kpi]
    
    analyze_button.click(
        fn=process_and_init_dashboard, 
        inputs=[file_input, date_col, cost_col, imp_col, click_col, conv_col, channel_col, rev_col, date_format_selector, date_format_custom], 
        outputs=[df_state, dashboard_group, month_filter, channel_filter] + dashboard_components
    )
    
    filter_inputs = [df_state, month_filter, channel_filter, kpi_type, target_cpa, target_roas]
    for comp in [month_filter, channel_filter, kpi_type, target_cpa, target_roas]:
        comp.change(fn=update_dashboard_display, inputs=filter_inputs, outputs=dashboard_components)

    def toggle_kpi_input(kpi_choice): return gr.update(visible=kpi_choice == "cpa"), gr.update(visible=kpi_choice == "roas")
    kpi_type.change(toggle_kpi_input, inputs=kpi_type, outputs=[target_cpa, target_roas])

if __name__ == "__main__":
    app.launch()
