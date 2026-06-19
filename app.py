import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io, json, hashlib, os, re, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0a0f1e; --surface:#111827; --surface2:#1a2235; --border:#1e2d45;
    --accent:#3b82f6; --accent2:#8b5cf6; --success:#10b981;
    --warning:#f59e0b; --text:#e2e8f0; --muted:#64748b;
    --glow:rgba(59,130,246,0.15);
}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem!important;}

/* ── Auth card ── */
.auth-wrap{display:flex;align-items:center;justify-content:center;min-height:80vh;}
.auth-card{background:var(--surface);border:1px solid var(--border);border-radius:20px;
    padding:2.5rem 2.8rem;width:100%;max-width:420px;box-shadow:0 20px 60px rgba(0,0,0,0.5);}
.auth-logo{font-size:2rem;text-align:center;margin-bottom:0.3rem;}
.auth-title{font-size:1.5rem;font-weight:700;text-align:center;
    background:linear-gradient(90deg,#e2e8f0,#93c5fd,#c4b5fd);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:0.2rem;}
.auth-sub{text-align:center;color:var(--muted);font-size:0.85rem;margin-bottom:1.8rem;}
.auth-tabs{display:flex;gap:4px;background:var(--surface2);border-radius:10px;
    padding:4px;margin-bottom:1.5rem;}
.auth-tab{flex:1;text-align:center;padding:0.45rem;border-radius:7px;cursor:pointer;
    font-size:0.85rem;font-weight:500;color:var(--muted);border:none;background:transparent;}
.auth-tab.active{background:var(--accent);color:#fff;}

/* ── Hero ── */
.hero{background:linear-gradient(135deg,#0a0f1e,#111827,#0f1829);
    border:1px solid var(--border);border-radius:16px;padding:2.2rem 2.8rem;
    margin-bottom:1.8rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-40px;right:-40px;width:260px;height:260px;
    background:radial-gradient(circle,rgba(59,130,246,0.18),transparent 70%);pointer-events:none;}
.hero-title{font-size:2.4rem;font-weight:700;letter-spacing:-0.03em;line-height:1.1;
    margin:0 0 0.4rem;background:linear-gradient(90deg,#e2e8f0,#93c5fd 60%,#c4b5fd);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:1rem;color:var(--muted);margin:0;}
.badge{display:inline-block;background:var(--surface2);border:1px solid var(--border);
    border-radius:50px;padding:0.2rem 0.75rem;font-size:0.72rem;
    font-family:'JetBrains Mono',monospace;color:var(--accent);
    margin-right:0.4rem;margin-top:0.9rem;letter-spacing:0.04em;}

/* ── Metric cards ── */
.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:1.2rem 1.4rem;transition:border-color 0.2s;}
.metric-card:hover{border-color:var(--accent);}
.metric-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:0.4rem;font-family:'JetBrains Mono',monospace;}
.metric-value{font-size:1.9rem;font-weight:700;color:var(--text);line-height:1;}
.metric-sub{font-size:0.78rem;color:var(--muted);margin-top:0.25rem;}

/* ── Section headers ── */
.section-header{display:flex;align-items:center;gap:0.6rem;margin:2rem 0 1rem;}
.section-icon{width:28px;height:28px;background:var(--glow);
    border:1px solid rgba(59,130,246,0.3);border-radius:8px;
    display:flex;align-items:center;justify-content:center;font-size:0.85rem;}
.section-title{font-size:1.1rem;font-weight:600;color:var(--text);margin:0;}

/* ── Insight block ── */
.insight-block{background:linear-gradient(135deg,var(--surface),var(--surface2));
    border:1px solid var(--border);border-left:3px solid var(--accent);
    border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;
    font-size:0.92rem;line-height:1.7;white-space:pre-wrap;}

/* ── Upload zone ── */
.upload-zone{border:2px dashed var(--border);border-radius:14px;padding:3rem 2rem;
    text-align:center;background:var(--surface);}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px!important;
    padding:4px!important;border:1px solid var(--border)!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;color:var(--muted)!important;
    font-size:0.85rem!important;font-family:'Space Grotesk',sans-serif!important;
    font-weight:500!important;padding:0.4rem 1rem!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:#fff!important;}

/* ── Inputs ── */
.stTextInput input,.stTextArea textarea,.stSelectbox>div>div{
    background:var(--surface2)!important;border:1px solid var(--border)!important;
    color:var(--text)!important;border-radius:8px!important;}
.stButton>button{
    background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
    color:#fff!important;border:none!important;border-radius:8px!important;
    font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;
    font-size:0.88rem!important;padding:0.55rem 1.5rem!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .stMarkdown p{color:var(--muted)!important;font-size:0.85rem;}

/* ── User chip ── */
.user-chip{display:inline-flex;align-items:center;gap:0.5rem;
    background:var(--surface2);border:1px solid var(--border);
    border-radius:50px;padding:0.3rem 0.9rem;font-size:0.82rem;}
.user-dot{width:8px;height:8px;background:var(--success);border-radius:50%;}
</style>
""", unsafe_allow_html=True)

# ── Auth helpers ──────────────────────────────────────────────────────────────
USERS_FILE = "users.json"

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

def _valid_password(pw: str) -> bool:
    return (len(pw) >= 8 and
            any(c.isupper() for c in pw) and
            any(c.isdigit() for c in pw))

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("authenticated", False),
    ("username", ""),
    ("email", ""),
    ("auth_mode", "signin"),   # "signin" | "signup"
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Auth screen ───────────────────────────────────────────────────────────────
def show_auth():
    st.markdown('<div class="auth-wrap"><div style="width:100%;max-width:420px">', unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-card">
        <div class="auth-logo">🔬</div>
        <div class="auth-title">DataLens AI</div>
        <div class="auth-sub">Instant EDA + AI insights for any CSV</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.session_state.auth_mode

    col_in, col_up = st.columns(2)
    with col_in:
        if st.button("Sign In", use_container_width=True,
                     type="primary" if mode == "signin" else "secondary"):
            st.session_state.auth_mode = "signin"
            st.rerun()
    with col_up:
        if st.button("Sign Up", use_container_width=True,
                     type="primary" if mode == "signup" else "secondary"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    st.markdown("---")

    if mode == "signup":
        name  = st.text_input("Full name", placeholder="Shrey Reddy")
        email = st.text_input("Email", placeholder="you@example.com")
        pw    = st.text_input("Password", type="password",
                               placeholder="Min 8 chars, 1 uppercase, 1 number")
        pw2   = st.text_input("Confirm password", type="password", placeholder="Repeat password")

        if st.button("Create account", use_container_width=True):
            users = _load_users()
            if not name.strip():
                st.error("Enter your full name.")
            elif not _valid_email(email):
                st.error("Enter a valid email address.")
            elif email in users:
                st.error("An account with this email already exists. Sign in instead.")
            elif not _valid_password(pw):
                st.error("Password must be ≥8 chars with at least 1 uppercase letter and 1 number.")
            elif pw != pw2:
                st.error("Passwords don't match.")
            else:
                users[email] = {"name": name.strip(), "hash": _hash(pw)}
                _save_users(users)
                st.session_state.authenticated = True
                st.session_state.username = name.strip()
                st.session_state.email = email
                st.success(f"Welcome, {name.strip()}! Account created.")
                st.rerun()

    else:  # signin
        email = st.text_input("Email", placeholder="you@example.com")
        pw    = st.text_input("Password", type="password", placeholder="Your password")

        if st.button("Sign in", use_container_width=True):
            users = _load_users()
            if email not in users:
                st.error("No account found for this email. Sign up first.")
            elif users[email]["hash"] != _hash(pw):
                st.error("Incorrect password.")
            else:
                st.session_state.authenticated = True
                st.session_state.username = users[email]["name"]
                st.session_state.email = email
                st.rerun()

        st.markdown('<p style="color:#64748b;font-size:0.8rem;text-align:center;margin-top:1rem">Don\'t have an account? Click Sign Up above.</p>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    show_auth()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (only reached when authenticated)
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=20, r=20, t=40, b=30),
    colorway=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4"],
    xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
)

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <p class="section-title">{title}</p>
    </div>""", unsafe_allow_html=True)

def metric_cards(df, numeric_cols, cat_cols):
    missing = df.isnull().sum().sum()
    miss_pct = round(missing / df.size * 100, 1)
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Rows</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-sub">observations</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Columns</div>
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-sub">{len(numeric_cols)} numeric · {len(cat_cols)} categorical</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value">{missing:,}</div>
            <div class="metric-sub">{miss_pct}% of total cells</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Memory</div>
            <div class="metric-value">{round(df.memory_usage(deep=True).sum()/1024,1)}<span style="font-size:1rem"> KB</span></div>
            <div class="metric-sub">in-memory size</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div class="user-chip">
            <div class="user-dot"></div>
            <span>{st.session_state.username}</span>
        </div>
        <p style="font-size:0.75rem;color:#64748b;margin-top:0.4rem">{st.session_state.email}</p>
    </div>""", unsafe_allow_html=True)

    if st.button("Sign out", use_container_width=True):
        for k in ["authenticated","username","email","auth_mode"]:
            del st.session_state[k]
        st.rerun()

    st.divider()
    st.markdown("**Settings**")
    max_rows_preview = st.slider("Preview rows", 5, 50, 10)
    chart_height     = st.slider("Chart height (px)", 300, 700, 420)
    st.divider()
    st.markdown("**Built with**")
    for lib in ["Streamlit","Pandas","Plotly","Claude AI (Anthropic)"]:
        st.markdown(f"· {lib}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <p class="hero-title">DataLens AI</p>
    <p class="hero-sub">Welcome back, {st.session_state.username} — drop a CSV to get started.</p>
    <span class="badge">v2.0</span>
    <span class="badge">LLaMA-3.3-70B</span>
    <span class="badge">Plotly</span>
    <span class="badge">Zero config</span>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem;margin-bottom:0.75rem">📂</div>
        <p style="color:#e2e8f0;font-weight:600;font-size:1.05rem;margin:0 0 0.4rem">Drag & drop your CSV here</p>
        <p style="color:#64748b;font-size:0.95rem;margin:0">or use the button above · supports any tabular CSV</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load(file):
    df = pd.read_csv(file)
    for col in df.columns:
        if df[col].dtype == "object":
            try:    df[col] = pd.to_numeric(df[col])
            except: df[col] = df[col].astype(str)
    return df

with st.spinner("Loading dataset…"):
    df = load(uploaded_file)

numeric_cols = df.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
cat_cols     = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

metric_cards(df, numeric_cols, cat_cols)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["📋 Overview","📊 Charts","🔥 Correlations","🧹 Data Quality","🤖 AI Insights"])

# ── TAB 1 – OVERVIEW ─────────────────────────────────────────────────────────
with t1:
    section("📋", "Dataset Preview")
    st.dataframe(df.head(max_rows_preview), use_container_width=True, height=300)

    col_a, col_b = st.columns(2)
    with col_a:
        section("🔢", "Numeric Summary")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")
    with col_b:
        section("🔤", "Categorical Summary")
        if cat_cols:
            cat_summary = pd.DataFrame({
                "Unique":    [df[c].nunique() for c in cat_cols],
                "Top Value": [df[c].value_counts().idxmax() for c in cat_cols],
                "Top Freq":  [df[c].value_counts().max() for c in cat_cols],
                "Missing":   [df[c].isnull().sum() for c in cat_cols],
            }, index=cat_cols)
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    section("🗂️", "Column Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype":  [str(d) for d in df.dtypes],
        "Nulls":  df.isnull().sum().values,
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ── TAB 2 – CHARTS ───────────────────────────────────────────────────────────
with t2:
    if not numeric_cols:
        st.warning("No numeric columns available for charting.")
    else:
        c1, c2, c3 = st.columns([2,2,1])
        with c1: chart_col  = st.selectbox("Primary column", numeric_cols, key="cc")
        with c2: chart_type = st.selectbox("Chart type",
                              ["Histogram","Box Plot","Violin","Line","Area","ECDF"], key="ct")
        with c3: color_col  = st.selectbox("Color by", ["None"]+cat_cols, key="color")

        color_arg = None if color_col == "None" else color_col
        tmp = df[[chart_col]+([color_col] if color_arg else [])].dropna()

        if   chart_type == "Histogram": fig = px.histogram(tmp, x=chart_col, color=color_arg, nbins=40, barmode="overlay", opacity=0.75, height=chart_height)
        elif chart_type == "Box Plot":  fig = px.box(tmp, y=chart_col, color=color_arg, points="outliers", height=chart_height)
        elif chart_type == "Violin":    fig = px.violin(tmp, y=chart_col, color=color_arg, box=True, points="outliers", height=chart_height)
        elif chart_type == "Line":      fig = px.line(tmp.reset_index(), x="index", y=chart_col, color=color_arg, height=chart_height)
        elif chart_type == "Area":      fig = px.area(tmp.reset_index(), x="index", y=chart_col, color=color_arg, height=chart_height)
        else:                           fig = px.ecdf(tmp, x=chart_col, color=color_arg, height=chart_height)

        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.divider()
            section("🔵", "Scatter Explorer")
            s1, s2, s3 = st.columns(3)
            with s1: x_col    = st.selectbox("X axis", numeric_cols, key="sx")
            with s2: y_col    = st.selectbox("Y axis", numeric_cols, index=min(1,len(numeric_cols)-1), key="sy")
            with s3: size_col = st.selectbox("Size by", ["None"]+numeric_cols, key="ss")

            size_arg = None if size_col == "None" else size_col
            cols_needed = list({x_col, y_col}|({size_arg} if size_arg else set())|({color_col} if color_arg else set()))
            scatter_df = df[cols_needed].dropna()

            fig2 = px.scatter(scatter_df, x=x_col, y=y_col,
                              color=color_arg, size=size_arg,
                              opacity=0.7, height=chart_height)
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        if cat_cols:
            st.divider()
            section("📊", "Category Breakdown")
            bc1, bc2 = st.columns(2)
            with bc1: bar_cat = st.selectbox("Category column", cat_cols, key="bc")
            with bc2: bar_n   = st.number_input("Top N values", 3, 30, 10, key="bn")

            top = df[bar_cat].value_counts().head(int(bar_n)).reset_index()
            top.columns = [bar_cat, "Count"]
            fig3 = px.bar(top, x=bar_cat, y="Count",
                          color="Count", color_continuous_scale="Blues",
                          height=min(chart_height, 380))
            fig3.update_layout(**{**PLOTLY_LAYOUT, "coloraxis_showscale": False})
            st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3 – CORRELATIONS ─────────────────────────────────────────────────────
with t3:
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr_method = st.radio("Method", ["pearson","spearman","kendall"], horizontal=True)
        corr = df[numeric_cols].corr(method=corr_method).round(3)

        section("🔥", "Correlation Heatmap")
        fig4 = px.imshow(corr, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, text_auto=True, aspect="auto",
                         height=max(400, len(numeric_cols)*42))
        fig4.update_traces(textfont_size=10)
        fig4.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=10,r=10,t=30,b=10)})
        st.plotly_chart(fig4, use_container_width=True)

        section("🏆", "Strongest Correlations")
        corr_pairs = (corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
                      .stack().reset_index())
        corr_pairs.columns = ["Column A","Column B","Correlation"]
        corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values("Abs",ascending=False).drop("Abs",axis=1).head(15)
        st.dataframe(corr_pairs.style.background_gradient(subset=["Correlation"],cmap="RdBu_r"),
                     use_container_width=True, hide_index=True)

# ── TAB 4 – DATA QUALITY ─────────────────────────────────────────────────────
with t4:
    section("🧹", "Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column","Missing"]
    missing_df["Pct"] = (missing_df["Missing"]/len(df)*100).round(2)
    missing_df = missing_df[missing_df["Missing"]>0].sort_values("Missing",ascending=False)

    if missing_df.empty:
        st.success("✅  No missing values — your dataset is clean.")
    else:
        fig5 = px.bar(missing_df, x="Column", y="Pct",
                      color="Pct", color_continuous_scale="Reds",
                      labels={"Pct":"Missing %"}, height=350, text="Missing")
        fig5.update_layout(**{**PLOTLY_LAYOUT, "coloraxis_showscale": False})
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    section("🔍", "Duplicate Rows")
    dupes = df.duplicated().sum()
    if dupes == 0:
        st.success("✅  No duplicate rows found.")
    else:
        st.warning(f"⚠️  {dupes:,} duplicate rows detected ({dupes/len(df)*100:.1f}% of data).")

    if numeric_cols:
        section("📐", "Outlier Detection (IQR)")
        outlier_data = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3-q1
            n_out = ((df[col]<q1-1.5*iqr)|(df[col]>q3+1.5*iqr)).sum()
            outlier_data.append({"Column":col,"Outliers":n_out,
                                  "Pct":round(n_out/len(df)*100,2),"IQR":round(iqr,4)})
        out_df = pd.DataFrame(outlier_data).sort_values("Outliers",ascending=False)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

# ── TAB 5 – AI INSIGHTS ──────────────────────────────────────────────────────
with t5:
    section("🤖", "AI-Powered Analysis")
    st.markdown("Ask DataLens AI anything about your dataset, or pick a quick-fire analysis.")

    quick = st.selectbox("Quick analysis", [
        "Custom question ↓",
        "Give me 5 key business insights from this data",
        "What anomalies or data quality issues do you see?",
        "Which columns are most predictive of interesting patterns?",
        "Summarize this dataset for a non-technical stakeholder",
        "What additional data would improve this analysis?",
    ])

    user_question = st.text_area(
        "Your question",
        value="" if quick == "Custom question ↓" else quick,
        placeholder="e.g. What drives the highest values in the sales column?",
        height=80,
    )

    if st.button("🔍 Analyze with AI"):
        if not user_question.strip():
            st.warning("Type a question or pick a quick analysis above.")
        else:
            summary   = df.describe(include="all").round(3).to_string()
            miss_info = df.isnull().sum().to_string()
            prompt    = f"""You are a senior data analyst. Analyze this dataset and answer the user's question with precision and depth.

Dataset overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Numeric columns: {numeric_cols}
- Categorical columns: {cat_cols}
- Duplicate rows: {df.duplicated().sum()}

Summary statistics:
{summary}

Missing values:
{miss_info}

Sample (first 5 rows):
{df.head(5).to_string()}

User question: {user_question}

Provide a thorough, structured analysis. Use clear sections with headers. Reference actual column names and values. Highlight actionable insights and caveats."""

            with st.spinner("DataLens AI is thinking…"):
                try:
                    resp = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 2000,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    answer = resp.json()["content"][0]["text"]
                    st.success("Analysis complete")
                    st.markdown(f'<div class="insight-block">{answer}</div>',
                                unsafe_allow_html=True)
                    buf = io.StringIO()
                    buf.write(f"DataLens AI Analysis\n{'='*60}\n\nQuestion: {user_question}\n\n{answer}")
                    st.download_button("⬇️ Download insights", buf.getvalue(),
                                       file_name="datalens_insights.txt", mime="text/plain")
                except Exception as e:
                    st.error(f"AI error: {e}")
