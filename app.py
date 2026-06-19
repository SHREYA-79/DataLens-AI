"""
DataLens AI - Intelligent Data Analysis Assistant
A complete, API-free data exploration tool for professionals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="DataLens AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — instrument-panel design system
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --ink: #0E1117;
        --panel: #161B22;
        --panel-raised: #1C232D;
        --hairline: #2A323D;
        --amber: #E8A33D;
        --amber-dim: #8A6228;
        --steel: #5B8AB0;
        --good: #5FA777;
        --warn: #D89A4A;
        --bad: #C75450;
        --text: #E8E9EB;
        --text-dim: #8B93A0;
        --mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
        --sans: 'Inter', -apple-system, sans-serif;
    }

    .main { padding: 0rem 1.2rem; }
    body, .stApp { font-family: var(--sans); }

    /* ── Header: instrument faceplate ───────────────────────────── */
    .header-container {
        background: linear-gradient(180deg, var(--panel) 0%, var(--ink) 100%);
        border: 1px solid var(--hairline);
        border-top: 2px solid var(--amber);
        padding: 1.8rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 18px;
        background-image: repeating-linear-gradient(
            90deg, var(--hairline) 0px, var(--hairline) 1px,
            transparent 1px, transparent 14px
        );
        opacity: 0.6;
    }

    .header-eyebrow {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: var(--amber);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .header-container h1 {
        font-family: var(--sans);
        font-size: 2.1rem;
        margin: 0;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .header-container .subtitle {
        font-size: 0.98rem;
        color: var(--text-dim);
        margin-top: 0.45rem;
    }

    .header-container .tagline {
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--text-dim);
        opacity: 0.8;
        margin-top: 0.7rem;
        letter-spacing: 0.02em;
    }

    /* ── Metric / readout cards ─────────────────────────────────── */
    .metric-card {
        background: var(--panel);
        padding: 1rem 1.1rem;
        border-radius: 5px;
        border: 1px solid var(--hairline);
        border-left: 2px solid var(--amber-dim);
        transition: border-color 0.15s ease;
    }

    .metric-card:hover {
        border-left-color: var(--amber);
    }

    .metric-card .label {
        font-family: var(--mono);
        font-size: 0.68rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-card .value {
        font-family: var(--mono);
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.25rem;
        letter-spacing: -0.01em;
    }

    /* ── Insight cards ──────────────────────────────────────────── */
    .insight-card {
        background: var(--panel);
        padding: 1rem 1.2rem;
        border-radius: 5px;
        border: 1px solid var(--hairline);
        border-left: 2px solid var(--steel);
        margin: 0.45rem 0;
        transition: border-color 0.15s ease;
        color: var(--text);
    }

    .insight-card:hover {
        border-left-color: var(--amber);
    }

    .insight-card .icon {
        font-size: 1.2rem;
        margin-right: 0.6rem;
    }

    /* ── Section headers ────────────────────────────────────────── */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.4rem 0 0.9rem 0;
        color: var(--text);
        border-bottom: 1px solid var(--hairline);
        padding-bottom: 0.5rem;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        background: var(--amber);
        color: var(--ink);
        border: none;
        padding: 0.55rem 1.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-family: var(--sans);
        transition: all 0.15s ease;
        width: 100%;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #F2B458;
        transform: translateY(-1px);
    }

    /* ── Tabs: control strip ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: var(--panel);
        border-radius: 6px;
        border: 1px solid var(--hairline);
        padding: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 3px;
        padding: 0.5rem 1.1rem;
        color: var(--text-dim);
        font-family: var(--mono);
        font-size: 0.85rem;
    }

    .stTabs [aria-selected="true"] {
        background: var(--panel-raised);
        color: var(--amber) !important;
        box-shadow: inset 0 -2px 0 var(--amber);
    }

    /* ── Dataframe container ────────────────────────────────────── */
    .dataframe-container {
        background: var(--panel);
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid var(--hairline);
    }

    /* ── Error / alert container ────────────────────────────────── */
    .error-container {
        background: #211514;
        border: 1px solid var(--bad);
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* ── Health score badge ─────────────────────────────────────── */
    .health-badge {
        display: inline-flex;
        align-items: baseline;
        justify-content: center;
        font-family: var(--mono);
        font-size: 1.7rem;
        font-weight: 800;
        padding: 0.55rem 1.3rem;
        border-radius: 5px;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    /* ── Misc text harmonization ────────────────────────────────── */
    h3, h4 { font-family: var(--sans); letter-spacing: -0.01em; }
    .stCaption, [data-testid="stCaptionContainer"] { font-family: var(--mono) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-eyebrow">DATA INSTRUMENT // v2.1</div>
    <h1>DataLens AI</h1>
    <p class="subtitle">Statistical profiling, correlation analysis, and automated insight extraction for tabular data.</p>
    <p class="tagline">NO API KEY · LOCAL PROCESSING · INSTANT READOUT</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##### CONTROL PANEL")
    st.divider()
    
    # Display options
    st.markdown("**Display**")
    preview_rows = st.slider("Preview Rows", 5, 50, 10)
    chart_height = st.slider("Chart Height", 300, 600, 400)
    
    st.divider()
    
    # Analysis options
    st.markdown("**Analysis**")
    corr_method = st.selectbox("Correlation", ["pearson", "spearman"])
    outlier_threshold = st.slider("Outlier Threshold", 2.0, 4.0, 3.0, 0.5)
    
    st.divider()
    st.caption("local processing · no data leaves this session")

# ── File Upload ────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload Your Dataset (CSV)",
    type=["csv"],
    help="Upload a CSV file to begin analysis"
)

if uploaded_file is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### 👋 Welcome to DataLens AI
        
        **Key Features:**
        - 📊 **Instant Data Profiling** - Get insights in seconds
        - 📈 **Interactive Visualizations** - Explore your data
        - 📋 **Business Insights** - Actionable recommendations
        - 🔍 **Data Quality Check** - Identify issues quickly
        
        **Getting Started:**
        1. Upload a CSV file
        2. Explore the tabs
        3. Download insights
        """)
    st.stop()

# ── Data Processing ────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    """Load and process CSV file with duplicate column handling"""
    df = pd.read_csv(file)

    duplicate_warning = False
    datetime_converted = []

    # Handle duplicate column names
    if df.columns.duplicated().any():
        # Rename duplicate columns
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                f"{dup}_{i}" if i > 0 else dup 
                for i in range(sum(cols == dup))
            ]
        df.columns = cols
        duplicate_warning = True
    
    # Smart type conversion
    for col in df.columns:
        # Try numeric conversion
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
            except:
                pass
        
        # Try datetime conversion - only commit if it's confidently a date column
        if df[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                # Require at least 80% of non-null values to parse successfully,
                # otherwise leave the column as-is (avoids false positives on
                # short numeric/code columns that pandas can misread as dates)
                non_null = df[col].notna().sum()
                if non_null > 0 and (converted.notna().sum() / non_null) >= 0.8:
                    df[col] = converted
                    datetime_converted.append(col)
            except:
                pass
    
    return df, duplicate_warning, datetime_converted

# ── Helper function for safe plotting ──────────────────────────────────
def safe_plot(func, *args, **kwargs):
    """Wrapper to safely create plots with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ Could not create plot: {str(e)[:100]}...")
        return None

with st.spinner("🔄 Loading and analyzing data..."):
    df, duplicate_col_warning, datetime_converted_cols = load_data(uploaded_file)

if duplicate_col_warning:
    st.warning("⚠️ Duplicate column names detected and have been renamed for analysis")

# Identify column types
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

if datetime_cols:
    st.success(f"📅 Detected {len(datetime_cols)} date column(s): {', '.join(datetime_cols)}")

# Calculate metrics
total_rows = len(df)
total_cols = len(df.columns)
missing_values = df.isnull().sum().sum()
missing_pct = round(missing_values / (total_rows * total_cols) * 100, 1) if (total_rows * total_cols) > 0 else 0
duplicates = df.duplicated().sum()
duplicates_pct = round((duplicates / total_rows * 100), 1) if total_rows > 0 else 0

# ── Dataset Quality Score ───────────────────────────────────────────────
# Normalized so a few duplicate rows in a huge dataset don't tank the score
# the same way they would in a tiny one.
quality_score = 100
quality_score -= missing_pct * 2
quality_score -= duplicates_pct
quality_score = max(0, round(quality_score))

if quality_score >= 90:
    quality_color = "#5FA777"
    quality_label = "Excellent"
elif quality_score >= 70:
    quality_color = "#D89A4A"
    quality_label = "Good"
else:
    quality_color = "#C75450"
    quality_label = "Needs Attention"

# ── Metrics Dashboard ──────────────────────────────────────────────────
st.markdown("##### DATASET READOUT")
col0, col1, col2, col3, col4, col5 = st.columns(6)

with col0:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {quality_color};">
        <div class="label">💯 Quality Score</div>
        <div class="value">{quality_score}<span style="font-size:0.9rem;color:#888;">/100</span></div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">📋 Rows</div>
        <div class="value">{total_rows:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">📑 Columns</div>
        <div class="value">{total_cols}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    color = "#5FA777" if missing_pct == 0 else "#D89A4A" if missing_pct < 5 else "#C75450"
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div class="label">⚠️ Missing</div>
        <div class="value">{missing_pct}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    color = "#5FA777" if duplicates == 0 else "#D89A4A"
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div class="label">🔄 Duplicates</div>
        <div class="value">{duplicates:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">📊 Numeric</div>
        <div class="value">{len(numeric_cols)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Main Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Profiling",
    "📊 Exploratory Analysis",
    "🔗 Correlation Intelligence",
    "📈 Time-Series Analytics",
    "💡 Business Insights"
])

# ── TAB 1: Data Preview ──────────────────────────────────────────────
with tab1:
    st.markdown("#### 🔍 Data Preview")
    st.dataframe(df.head(preview_rows), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Numeric Summary")
        if numeric_cols:
            st.dataframe(
                df[numeric_cols].describe().round(2),
                use_container_width=True
            )
        else:
            st.info("No numeric columns found")
    
    with col2:
        st.markdown("#### 📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Unique': df.nunique().values,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(1).values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

# ── TAB 2: Visualizations ─────────────────────────────────────────────
with tab2:
    if not numeric_cols:
        st.warning("No numeric columns available for visualization")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_col = st.selectbox("Select Column", numeric_cols)
        
        with col2:
            chart_type = st.selectbox(
                "Chart Type",
                ["Histogram", "Box Plot", "Violin Plot", "Density"]
            )
        
        # Create visualization with safe plotting
        fig = None
        
        try:
            if chart_type == "Histogram":
                fig = px.histogram(
                    df, x=selected_col,
                    nbins=30,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#E8A33D']
                )
                fig.update_layout(
                    showlegend=False,
                    height=chart_height,
                    bargap=0.1
                )
            
            elif chart_type == "Box Plot":
                fig = px.box(
                    df, y=selected_col,
                    title=f"Box Plot of {selected_col}",
                    color_discrete_sequence=['#E8A33D']
                )
                fig.update_layout(height=chart_height)
            
            elif chart_type == "Violin Plot":
                fig = px.violin(
                    df, y=selected_col,
                    box=True,
                    title=f"Violin Plot of {selected_col}",
                    color_discrete_sequence=['#E8A33D']
                )
                fig.update_layout(height=chart_height)
            
            elif chart_type == "Density":
                fig = px.density_contour(
                    df, x=selected_col,
                    title=f"Density Plot of {selected_col}",
                    color_discrete_sequence=['#E8A33D']
                )
                fig.update_layout(height=chart_height)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"⚠️ Could not create visualization: {str(e)[:150]}...")
            st.info("Try selecting a different column or chart type")
        
        # Categorical visualizations
        if categorical_cols:
            st.divider()
            st.markdown("#### 🏷️ Categorical Analysis")
            
            cat_col = st.selectbox("Select Category", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    value_counts = df[cat_col].value_counts().head(10)
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Categories - {cat_col}",
                        labels={'x': cat_col, 'y': 'Count'},
                        color_discrete_sequence=['#E8A33D']
                    )
                    fig_bar.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Could not create bar chart: {str(e)[:100]}...")
            
            with col2:
                try:
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution - {cat_col}",
                        hole=0.4
                    )
                    fig_pie.update_layout(height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Could not create pie chart: {str(e)[:100]}...")

# ── TAB 3: Relationships ──────────────────────────────────────────────
with tab3:
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
    else:
        # Correlation matrix
        st.markdown("#### 🔥 Correlation Matrix")
        
        try:
            corr_matrix = df[numeric_cols].corr(method=corr_method).round(2)
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Correlation Matrix ({corr_method})"
            )
            fig_corr.update_layout(height=max(400, len(numeric_cols) * 35))
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not create correlation matrix: {str(e)[:150]}...")
        
        # Top correlations
        st.divider()
        st.markdown("#### 🏆 Strongest Correlations")
        
        try:
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            if not corr_df.empty:
                corr_df['|Correlation|'] = corr_df['Correlation'].abs()
                corr_df = corr_df.sort_values('|Correlation|', ascending=False).head(10)
                corr_df['Strength'] = corr_df['Correlation'].apply(
                    lambda x: '🟢 Strong' if abs(x) >= 0.7 else 
                             '🟡 Moderate' if abs(x) >= 0.4 else '🔴 Weak'
                )
                corr_df['Direction'] = corr_df['Correlation'].apply(
                    lambda x: '📈 Positive' if x > 0 else '📉 Negative'
                )
                
                st.dataframe(
                    corr_df[['Feature 1', 'Feature 2', 'Correlation', 'Strength', 'Direction']],
                    use_container_width=True,
                    hide_index=True
                )
        except Exception as e:
            st.error(f"⚠️ Could not calculate correlations: {str(e)[:150]}...")
        
        # Scatter plot
        st.divider()
        st.markdown("#### 🔵 Scatter Plot")
        
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X-Axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-Axis", numeric_cols, key="scatter_y")
            with col3:
                color_col = st.selectbox(
                    "Color By",
                    ["None"] + categorical_cols,
                    key="scatter_color"
                )
            
            if x_col and y_col and x_col != y_col:
                try:
                    color = None if color_col == "None" else color_col
                    
                    # Create clean dataset with unique column names
                    plot_data = df[[x_col, y_col]].copy()
                    if color:
                        plot_data[color] = df[color]
                    
                    fig_scatter = px.scatter(
                        plot_data,
                        x=x_col,
                        y=y_col,
                        color=color,
                        title=f"{y_col} vs {x_col}",
                        trendline="ols" if color is None else None,
                        opacity=0.7
                    )
                    fig_scatter.update_layout(height=chart_height)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Could not create scatter plot: {str(e)[:150]}...")
                    st.info("Try selecting different columns or ensure columns have valid data")

# ── TAB 4: Trends ──────────────────────────────────────────────────────
with tab4:
    if datetime_cols:
        st.markdown("#### 📈 Time Series Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", datetime_cols)
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols)
        
        if date_col and value_col:
            try:
                # Prepare time series data
                ts_data = df[[date_col, value_col]].dropna()
                ts_data = ts_data.sort_values(date_col)
                
                # Resample options
                resample_freq = st.selectbox(
                    "Resample",
                    ["None", "Daily", "Weekly", "Monthly", "Quarterly"]
                )
                
                if resample_freq != "None":
                    freq_map = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "ME",
                        "Quarterly": "QE"
                    }
                    ts_data = ts_data.set_index(date_col)
                    ts_data = ts_data.resample(freq_map[resample_freq]).mean().reset_index()
                
                # Plot
                fig_ts = px.line(
                    ts_data,
                    x=date_col,
                    y=value_col,
                    title=f"{value_col} Over Time",
                    markers=len(ts_data) < 50
                )
                fig_ts.update_layout(height=chart_height)
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Rolling average
                if len(ts_data) > 10:
                    window = st.slider("Rolling Window", 2, 20, 5)
                    
                    ts_data['Rolling Avg'] = ts_data[value_col].rolling(window).mean()
                    
                    fig_roll = px.line(
                        ts_data,
                        x=date_col,
                        y=[value_col, 'Rolling Avg'],
                        title=f"{value_col} with {window}-Period Rolling Average",
                        labels={'value': value_col}
                    )
                    fig_roll.update_layout(height=chart_height)
                    st.plotly_chart(fig_roll, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Could not create time series plot: {str(e)[:150]}...")
    
    else:
        st.info("No date columns found. Please ensure your data has datetime columns for trend analysis.")

# ── TAB 5: Insights ────────────────────────────────────────────────────
with tab5:
    st.markdown("#### 💡 Automated Business Insights")

    # ── AI-style summary banner ──────────────────────────────────────
    summary_lines = []
    if missing_pct == 0:
        summary_lines.append(("✓", "Low missing values", "#5FA777"))
    elif missing_pct < 5:
        summary_lines.append(("✓", f"Acceptable missing values ({missing_pct}%)", "#5FA777"))
    else:
        summary_lines.append(("⚠", f"High missing values ({missing_pct}%)", "#C75450"))

    if duplicates == 0:
        summary_lines.append(("✓", "No duplicate records", "#5FA777"))
    else:
        summary_lines.append(("⚠", f"{duplicates} duplicate records found", "#C75450"))

    # Check for highly skewed numeric features for the summary
    skewed_features = []
    for col in numeric_cols:
        try:
            s = df[col].dropna().skew()
            if abs(s) > 1.5:
                skewed_features.append(col)
        except:
            pass

    if skewed_features:
        summary_lines.append(("⚠", f"{len(skewed_features)} highly skewed numeric feature(s)", "#D89A4A"))
    else:
        summary_lines.append(("✓", "Clean numeric distributions", "#5FA777"))

    if categorical_cols:
        summary_lines.append(("✓", "Categorical structure identified", "#5FA777"))

    summary_html = "".join(
        f'<div style="margin:0.3rem 0;color:{c};">{icon} {text}</div>'
        for icon, text, c in summary_lines
    )

    st.markdown(f"""
    <div class="insight-card" style="display:flex; align-items:center; gap:2rem;">
        <div>
            <span class="health-badge" style="background:{quality_color};">{quality_score}/100</span>
            <div style="margin-top:0.4rem; color:#888; font-size:0.85rem;">Dataset Health Score</div>
        </div>
        <div>{summary_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    # Generate insights
    insights = []
    
    # Data quality insights
    if missing_pct == 0:
        insights.append(("✅", "Perfect data quality - no missing values"))
    elif missing_pct < 5:
        insights.append(("⚠️", f"Good data quality - only {missing_pct}% missing values"))
    else:
        insights.append(("🚨", f"Data quality needs attention - {missing_pct}% missing values"))
    
    if duplicates > 0:
        insights.append(("🔄", f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)"))
    
    # Numeric insights
    for col in numeric_cols[:5]:  # Limit to top 5
        try:
            data = df[col].dropna()
            if len(data) > 1:
                skew = data.skew()
                cv = (data.std() / data.mean()) * 100 if data.mean() != 0 else 0
                
                if abs(skew) > 1.5:
                    direction = "right" if skew > 0 else "left"
                    insights.append((
                        "📊",
                        f"{col} is heavily skewed {direction} (skew={skew:.2f}) - consider transformation"
                    ))
                
                if cv > 80:
                    insights.append((
                        "📊",
                        f"{col} has high variability (CV={cv:.1f}%) - investigate outliers"
                    ))
        except:
            pass
    
    # Correlation insights
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        insights.append((
                            "🔗",
                            f"Strong correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]} "
                            f"(r={corr_matrix.iloc[i, j]:.2f})"
                        ))
        except:
            pass
    
    # Categorical insights
    for col in categorical_cols[:3]:
        try:
            unique_count = df[col].nunique()
            if unique_count == 1:
                insights.append(("⚠️", f"{col} has only one unique value - consider dropping"))
            elif unique_count > len(df) * 0.8:
                insights.append((
                    "🆔",
                    f"{col} behaves like an identifier with {unique_count} unique values"
                ))
            elif unique_count > 20:
                insights.append(("🏷️", f"{col} has high cardinality ({unique_count} categories)"))
        except:
            pass
    
    # Display insights
    if insights:
        for icon, text in insights[:15]:  # Limit to 15 insights
            st.markdown(f"""
            <div class="insight-card">
                <span class="icon">{icon}</span> {text}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 No significant issues found! Your data looks clean and well-structured.")
    
    # Download insights
    st.divider()
    st.markdown("#### 📥 Export Insights")
    
    insight_text = "DataLens AI - Insights Report\n"
    insight_text += "=" * 50 + "\n\n"
    insight_text += f"Dataset: {uploaded_file.name}\n"
    insight_text += f"Rows: {total_rows:,} | Columns: {total_cols}\n"
    insight_text += f"Missing: {missing_pct}% | Duplicates: {duplicates}\n"
    insight_text += f"Dataset Health Score: {quality_score}/100 ({quality_label})\n\n"
    insight_text += "Key Insights:\n"
    insight_text += "-" * 30 + "\n"
    
    for icon, text in insights:
        insight_text += f"{icon} {text}\n"
    
    st.download_button(
        "📥 Download Insights Report",
        insight_text,
        file_name="data_insights_report.txt",
        mime="text/plain"
    )

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Built with Streamlit • Pandas • Plotly • Statistical Profiling</p>
    <p style="font-size: 0.8rem;">No API required • 100% Free • Open Source</p>
</div>
""", unsafe_allow_html=True)
