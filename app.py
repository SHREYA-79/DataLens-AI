from groq import Groq
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, warnings, requests
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataLens AI", page_icon="🔬", layout="wide")

st.markdown("""

""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PL = dict(
paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
font=dict(family="Space Grotesk", color="#e2e8f0", size=11),
margin=dict(l=10, r=10, t=36, b=10),
colorway=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#f43f5e","#a3e635"],
xaxis=dict(gridcolor="#1e2d45", zeroline=False, showgrid=True),
yaxis=dict(gridcolor="#1e2d45", zeroline=False, showgrid=True),
legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)

def sec(icon, title):
st.markdown(f'{icon}'
f'{title}', unsafe_allow_html=True)

def finding(icon, text):
st.markdown(f'{icon}'
f'{text}', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
st.markdown("### 🔬 DataLens AI")
st.caption("Business-ready EDA for data roles")
st.divider()
st.markdown("Display")
preview_n = st.slider("Preview rows", 5, 100, 15)
chart_h = st.slider("Chart height", 300, 650, 400)
st.divider()
st.markdown("Analysis")
corr_method = st.radio("Correlation", ["pearson","spearman"], horizontal=True)
outlier_z = st.slider("Outlier Z-threshold", 2.0, 4.0, 3.0, 0.5)
st.divider()
st.markdown("AI Model")
st.caption("Powered by Groq Llama 3.3 70B")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""

# ── Upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your CSV",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded is None:

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        st.info(
            "📊 Drop your CSV here\n\n"
            "Sales data • Customer data • Financial reports • Any tabular CSV"
        )

    st.stop()

# ── Load & profile ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_profile(file):

    df = pd.read_csv(file)

    # Smart dtype coercion
    for col in df.columns:

        if df[col].dtype == "object":

            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "")
                )
            except:
                pass

        if df[col].dtype == "object":

            try:
                df[col] = pd.to_datetime(
                    df[col],
                    infer_datetime_format=True
                )
            except:
                pass

    return df


with st.spinner("Profiling dataset..."):
    df = load_and_profile(uploaded)

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist()
missing = int(df.isnull().sum().sum())
miss_pct = round(missing / df.size * 100, 1)
dupes = int(df.duplicated().sum())

# ── KPI bar ───────────────────────────────────────────────────────────────────
miss_cls = "kpi-good" if miss_pct == 0 else ("kpi-warn" if miss_pct < 5 else "kpi-bad")
dupe_cls = "kpi-good" if dupes == 0 else "kpi-warn"
st.markdown(f"""

══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📋 Overview","📊 Distributions","🔗 Relationships",
"📈 Trends","🧹 Data Quality","💼 Business Summary","🤖 Ask AI"])

# ─── TAB 1: OVERVIEW ─────────────────────────────────────────────────────────
with tabs[0]:
sec("📋","Dataset Preview")
st.dataframe(df.head(preview_n), use_container_width=True, height=280)

c1, c2 = st.columns(2)
with c1:
sec("🔢","Numeric Profile")
if num_cols:
desc = df[num_cols].describe().T.round(3)
desc["cv%"] = (desc["std"]/desc["mean"].abs()*100).round(1)
desc["skew"] = df[num_cols].skew().round(3)
st.dataframe(desc[["count","mean","std","min","50%","max","cv%","skew"]],
use_container_width=True)
else:
st.info("No numeric columns.")
with c2:
sec("🔤","Categorical Profile")
if cat_cols:
rows = []
for c in cat_cols:
vc = df[c].value_counts()
rows.append({"Column","Unique".nunique(),
"Top".index[0],"Top %""{vc.iloc[0]/len(df)*100:.1f}%",
"Missing".isnull().sum()})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
st.info("No categorical columns.")

if num_cols:
sec("📐","Column-level Stats")
rows = []
for c in df.columns:
rows.append({
"Column": c, "Type": str(df[c].dtype),
"Nulls": df[c].isnull().sum(),
"Null%": f"{df[c].isnull().mean()*100:.1f}%",
"Unique": df[c].nunique(),
"Sample": str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── TAB 2: DISTRIBUTIONS ────────────────────────────────────────────────────
with tabs[1]:
if not num_cols:
st.warning("No numeric columns to plot.")
else:
c1, c2, c3 = st.columns([2,2,1])
with c1: col = st.selectbox("Column", num_cols, key="d_col")
with c2: kind = st.selectbox("Chart type",
["Histogram + KDE","Box + Strip","Violin","ECDF","QQ Plot"], key="d_kind")
with c3: grp = st.selectbox("Group by", ["None"]+cat_cols, key="d_grp")
grp_arg = None if grp=="None" else grp

tmp = df[[col]+([grp] if grp_arg else [])].dropna()

if kind == "Histogram + KDE":
fig = px.histogram(tmp, x=col, color=grp_arg, nbins=40,
barmode="overlay", opacity=0.7,
marginal="violin", height=chart_h)
elif kind == "Box + Strip":
fig = px.box(tmp, y=col, color=grp_arg, points="all",
height=chart_h, notched=True)
elif kind == "Violin":
fig = px.violin(tmp, y=col, color=grp_arg,
box=True, points="outliers", height=chart_h)
elif kind == "ECDF":
fig = px.ecdf(tmp, x=col, color=grp_arg, height=chart_h)
else: # QQ
from scipy import stats as scipy_stats
vals = tmp[col].dropna().sort_values()
theoretical = scipy_stats.norm.ppf(
np.linspace(0.01, 0.99, len(vals)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=theoretical, y=vals.values,
mode="markers", marker=dict(color="#3b82f6", size=4, opacity=0.6),
name="Data"))
mn, mx = theoretical.min(), theoretical.max()
fig.add_trace(go.Scatter(x=[mn,mx],
y=[vals.mean()+vals.std()*mn, vals.mean()+vals.std()*mx],
mode="lines", line=dict(color="#ef4444", dash="dash"),
name="Normal ref"))
fig.update_layout(xaxis_title="Theoretical Quantiles",
yaxis_title="Sample Quantiles",
title=f"QQ Plot — {col}", height=chart_h)

fig.update_layout(**PL)
st.plotly_chart(fig, use_container_width=True)

Multi-distribution overview
if len(num_cols) > 1:
st.divider()
sec("📦","All Numeric Distributions")
sel_cols = st.multiselect("Select columns", num_cols, default=num_cols[(6,len(num_cols))], key="multi_dist")
if sel_cols:
cols_per_row = 3
rows_needed = (len(sel_cols)+cols_per_row-1)//cols_per_row
fig_sub = make_subplots(rows=rows_needed, cols=cols_per_row,
subplot_titles=sel_cols)
for i, sc in enumerate(sel_cols):
r, c_ = divmod(i, cols_per_row)
vals = df[sc].dropna()
fig_sub.add_trace(
go.Histogram(x=vals, nbinsx=30, name=sc,
marker_color=PL["colorway"][i % len(PL["colorway"])],
showlegend=False, opacity=0.8),
row=r+1, col=c_+1)
fig_sub.update_layout(**{**PL, "height": rows_needed*220,
"showlegend": False})
fig_sub.update_annotations(font_size=10)
st.plotly_chart(fig_sub, use_container_width=True)

Category distributions
if cat_cols:
st.divider()
sec("🏷️","Category Distributions")
cat_pick = st.selectbox("Category column", cat_cols, key="cat_dist")
top_n = st.slider("Top N", 3, 30, 10, key="cat_n")
vc = df[cat_pick].value_counts().head(top_n).reset_index()
vc.columns = [cat_pick,"Count"]
vc["Pct"] = (vc["Count"]/len(df)*100).round(1)
col_a, col_b = st.columns(2)
with col_a:
fig_bar = px.bar(vc, x=cat_pick, y="Count",
color="Count", color_continuous_scale="Blues",
text="Pct", height=chart_h)
fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
fig_bar.update_layout(**{**PL,"coloraxis_showscale"})
st.plotly_chart(fig_bar, use_container_width=True)
with col_b:
fig_pie = px.pie(vc, names=cat_pick, values="Count",
hole=0.45, height=chart_h,
color_discrete_sequence=PL["colorway"])
fig_pie.update_layout(**PL)
st.plotly_chart(fig_pie, use_container_width=True)

# ─── TAB 3: RELATIONSHIPS ────────────────────────────────────────────────────
with tabs[2]:
if len(num_cols) < 2:
st.warning("Need at least 2 numeric columns.")
else:

Correlation heatmap
sec("🔥","Correlation Matrix")
corr = df[num_cols].corr(method=corr_method).round(3)
fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
zmin=-1, zmax=1, text_auto=True, aspect="auto",
height=max(380, len(num_cols)*40))
fig_corr.update_traces(textfont_size=9)
fig_corr.update_layout(**{**PL,"margin"(l=5,r=5,t=30,b=5)})
st.plotly_chart(fig_corr, use_container_width=True)

Top pairs
sec("🏆","Strongest Relationships")
pairs = (corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
.stack().reset_index())
pairs.columns = ["Feature A","Feature B","Correlation"]
pairs["|r|"] = pairs["Correlation"].abs()
pairs = pairs.sort_values("|r|",ascending=False).drop("|r|",axis=1).head(12)
pairs["Strength"] = pairs["Correlation"].abs().apply(
lambda x: "🔴 Strong" if x>=0.7 else ("🟡 Moderate" if x>=0.4 else "🟢 Weak"))
pairs["Direction"] = pairs["Correlation"].apply(
lambda x: "↑ Positive" if x>0 else "↓ Negative")
st.dataframe(pairs.style.background_gradient(subset=["Correlation"],cmap="RdBu_r"),
use_container_width=True, hide_index=True)

Scatter
st.divider()
sec("🔵","Scatter Analysis")
s1,s2,s3,s4 = st.columns(4)
with s1: x_col = st.selectbox("X axis", num_cols, key="sc_x")
with s2: y_col = st.selectbox("Y axis", num_cols, index=min(1,len(num_cols)-1), key="sc_y")
with s3: sz_col= st.selectbox("Size", ["None"]+num_cols, key="sc_sz")
with s4: cl_col= st.selectbox("Color", ["None"]+cat_cols, key="sc_cl")
sz_arg = None if sz_col=="None" else sz_col
cl_arg = None if cl_col=="None" else cl_col
needed = list({x_col,y_col}|({sz_arg} if sz_arg else set())|({cl_arg} if cl_arg else set()))
sdf = df[needed].dropna()
fig_sc = px.scatter(sdf, x=x_col, y=y_col, color=cl_arg, size=sz_arg,
opacity=0.65, height=chart_h,
trendline="ols" if cl_arg is None else None)
fig_sc.update_layout(**PL)
st.plotly_chart(fig_sc, use_container_width=True)

Pair plot (small)
if len(num_cols) >= 3:
st.divider()
sec("🔷","Pair Plot")
pp_cols = st.multiselect("Columns", num_cols,
default=num_cols[(4,len(num_cols))], key="pp")
pp_color = st.selectbox("Color by", ["None"]+cat_cols, key="pp_cl")
if len(pp_cols) >= 2:
fig_pp = px.scatter_matrix(
df[pp_cols+([pp_color] if pp_color!="None" else [])].dropna(),
dimensions=pp_cols,
color=None if pp_color=="None" else pp_color,
opacity=0.5, height=550)
fig_pp.update_traces(marker=dict(size=3))
fig_pp.update_layout(**{**PL,"margin"(l=10,r=10,t=30,b=10)})
st.plotly_chart(fig_pp, use_container_width=True)

# ─── TAB 4: TRENDS ───────────────────────────────────────────────────────────
with tabs[3]:
sec("📈","Time Series & Trends")
if date_cols:
d1,d2,d3 = st.columns(3)
with d1: dt_col = st.selectbox("Date column", date_cols, key="ts_dt")
with d2: val_col = st.selectbox("Value column", num_cols, key="ts_val")
with d3: freq = st.selectbox("Resample", ["Raw","Daily","Weekly","Monthly","Quarterly"], key="ts_freq")

ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
freq_map = {"Daily":"D","Weekly":"W","Monthly":"ME","Quarterly":"QE"}
if freq != "Raw":
ts = ts.set_index(dt_col)[val_col].resample(freq_map[freq]).agg(["sum","mean","count"]).reset_index()
ts.columns = [dt_col,"Sum","Mean","Count"]
metric_pick = st.radio("Metric", ["Sum","Mean","Count"], horizontal=True)
y_val = metric_pick
else:
y_val = val_col

fig_ts = px.line(ts, x=dt_col, y=y_val, height=chart_h,
markers=True if len(ts)<200 else False)
fig_ts.update_traces(line_color="#3b82f6")
fig_ts.update_layout(**PL)
st.plotly_chart(fig_ts, use_container_width=True)

Rolling average overlay
if freq == "Raw" and len(ts) > 20:
win = st.slider("Rolling average window", 2, min(90,len(ts)//3), 7)
ts2 = ts.copy()
ts2["Rolling"] = ts2[val_col].rolling(win).mean()
fig_r = go.Figure()
fig_r.add_trace(go.Scatter(x=ts2[dt_col], y=ts2[val_col],
mode="lines", name="Raw",
line=dict(color="#3b82f6", width=1), opacity=0.4))
fig_r.add_trace(go.Scatter(x=ts2[dt_col], y=ts2["Rolling"],
mode="lines", name=f"{win}-period MA",
line=dict(color="#f59e0b", width=2)))
fig_r.update_layout(**{**PL, "height": chart_h})
st.plotly_chart(fig_r, use_container_width=True)

else:
st.info("No date columns detected. Showing index-based trends.")
if num_cols:
tr_col = st.selectbox("Column", num_cols, key="trend_col")
win2 = st.slider("Rolling window", 2, min(50, len(df)//3), 5)
tdf = df[tr_col].dropna().reset_index(drop=True)
roll = tdf.rolling(win2).mean()
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(y=tdf, mode="lines", name="Raw",
line=dict(color="#3b82f6",width=1), opacity=0.4))
fig_t.add_trace(go.Scatter(y=roll, mode="lines",
name=f"{win2}-point MA",
line=dict(color="#f59e0b",width=2)))
fig_t.update_layout(**{**PL,"height",
"xaxis_title":"Index","yaxis_title"})
st.plotly_chart(fig_t, use_container_width=True)

# ─── TAB 5: DATA QUALITY ─────────────────────────────────────────────────────
with tabs[4]:
c1, c2 = st.columns(2)

with c1:
sec("🕳️","Missing Values")
miss_df = df.isnull().sum().reset_index()
miss_df.columns = ["Column","Missing"]
miss_df["Pct"] = (miss_df["Missing"]/len(df)*100).round(2)
miss_df = miss_df[miss_df["Missing"]>0].sort_values("Missing",ascending=False)
if miss_df.empty:
st.success("✅ Zero missing values — dataset is complete.")
else:
fig_m = px.bar(miss_df, x="Column", y="Pct", text="Missing",
color="Pct", color_continuous_scale="Reds", height=300)
fig_m.update_traces(textposition="outside")
fig_m.update_layout(**{**PL,"coloraxis_showscale",
"margin"(l=5,r=5,t=30,b=5)})
st.plotly_chart(fig_m, use_container_width=True)
st.dataframe(miss_df, use_container_width=True, hide_index=True)

with c2:
sec("📐","Outlier Report (Z-score)")
if num_cols:
out_rows = []
for col in num_cols:
s = df[col].dropna()
z = np.abs((s - s.mean()) / s.std())
n_out = int((z > outlier_z).sum())
out_rows.append({
"Column": col,
"Outliers": n_out,
"Pct%": round(n_out/len(df)*100,2),
"Min": round(s.min(),3),
"Max": round(s.max(),3),
"Skew": round(s.skew(),3),
"Flag": "⚠️" if n_out/len(df)>0.03 else "✅"
})
out_df = pd.DataFrame(out_rows).sort_values("Outliers",ascending=False)
st.dataframe(out_df, use_container_width=True, hide_index=True)

sec("🔍","Duplicate Analysis")
dupes_df = df[df.duplicated(keep=False)]
dc1, dc2, dc3 = st.columns(3)
dc1.metric("Total duplicates", f"{dupes:,}")
dc2.metric("Pct of data", f"{dupes/len(df)*100:.2f}%")
dc3.metric("Unique rows", f"{len(df)-dupes:,}")
if dupes > 0:
st.warning(f"⚠️ {dupes} duplicate rows found. Consider deduplication before modeling.")
if st.checkbox("Preview duplicate rows"):
st.dataframe(dupes_df.head(20), use_container_width=True)

sec("🧮","Data Type Summary")
dtype_df = pd.DataFrame({
"Column": df.columns,
"Dtype": df.dtypes.astype(str).values,
"Non-Null": df.count().values,
"Null": df.isnull().sum().values,
"Unique": [df[c].nunique() for c in df.columns],
"Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
})
st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ─── TAB 6: BUSINESS SUMMARY ─────────────────────────────────────────────────
with tabs[5]:
sec("💼","Automated Business Findings")

findings = []

Volume
findings.append(("📦", f"Dataset contains {df.shape[0]:,} records across {df.shape[1]} fields."))

Data quality
if miss_pct == 0 and dupes == 0:
findings.append(("✅", "Data quality is excellent — no missing values or duplicates detected."))
else:
if miss_pct > 0:
findings.append(("⚠️", f"{miss_pct}% missing data detected. Imputation or removal required before modeling."))
if dupes > 0:
findings.append(("⚠️", f"{dupes:,} duplicate rows ({dupes/len(df)*100:.1f}%) may skew aggregations and model training."))

Numeric insights
if num_cols:
for col in num_cols:
s = df[col].dropna()
cv = s.std()/s.mean()*100 if s.mean() != 0 else 0
if abs(s.skew()) > 1.5:
direction = "right (positive)" if s.skew() > 0 else "left (negative)"
findings.append(("📊", f"{col} is heavily skewed {direction} (skew={s.skew():.2f}). Consider log transformation for modeling."))
if cv > 80:
findings.append(("📊", f"{col} has high variability (CV={cv:.0f}%). Investigate outliers or natural subgroups."))

Correlation insights
if len(num_cols) >= 2:

corr2 = df[num_cols].corr().abs().copy()

Safe diagonal reset
for i in range(len(corr2)):
corr2.iat[i, i] = 0

max_corr = corr2.max().max()

if pd.notna(max_corr):

if max_corr >= 0.9:

cols_hc = corr2[corr2 >= 0.9].stack().index.tolist()

if cols_hc:
a, b = cols_hc[0]

findings.append(
(
"🔗",
f"High multicollinearity detected between "
f"{a} and {b} "
f"(r={corr2.loc[a,b]:.2f}). "
f"May cause issues in regression models."
)
)

elif max_corr >= 0.7:

findings.append(
(
"🔗",
f"Moderate-to-strong correlations exist "
f"(max r={max_corr:.2f}). "
f"Good signal for predictive modeling."
)
)

Cardinality
for col in cat_cols:
un = df[col].nunique()
if un == len(df):
findings.append(("🆔", f"{col} has all unique values — likely an ID column. Exclude from modeling."))
elif un == 1:
findings.append(("⚠️", f"{col} has only 1 unique value — zero variance. Drop this column."))
elif un > 50:
findings.append(("🏷️", f"{col} has high cardinality ({un} categories). Consider encoding or grouping before ML."))

Date insight
if date_cols:
for dc in date_cols:
span = (df[dc].max() - df[dc].min()).days
findings.append(("📅", f"{dc} spans {span} days ({df[dc].min().date()} → {df[dc].max().date()})."))

for icon, text in findings:
st.markdown(f'{icon}'
f'{text}', unsafe_allow_html=True)

Numeric summary table
if num_cols:
st.divider()
sec("📊","Key Metrics at a Glance")
kpi_rows = []
for col in num_cols:
s = df[col].dropna()
kpi_rows.append({
"Metric": col,
"Mean": f"{s.mean():,.2f}",
"Median": f"{s.median():,.2f}",
"Std Dev": f"{s.std():,.2f}",
"Min": f"{s.min():,.2f}",
"Max": f"{s.max():,.2f}",
"CV%": f"{s.std()/s.mean()*100:.1f}%" if s.mean()!=0 else "—",
"Skew": f"{s.skew():.2f}",
})
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

Download report
st.divider()
report_lines = ["DataLens AI — Business Summary Report", "="*50, ""]
for icon, text in findings:
clean = text.replace("","").replace("","")
report_lines.append(f"{icon} {clean}")
report_lines += ["", "="*50, f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]} | Missing: {miss_pct}% | Duplicates: {dupes}"]
st.download_button("⬇️ Download Business Report",
"\n".join(report_lines),
file_name="datalens_business_report.txt",
mime="text/plain")

# ─── TAB 7: ASK AI ───────────────────────────────────────────────────────────
with tabs[6]:
sec("🤖","Ask AI About Your Data")
st.markdown('Powered by Groq Llama 3.3 70B · Ask anything about your dataset — business questions, modeling advice, anomaly explanations.', unsafe_allow_html=True)

quick = st.selectbox("Quick questions", [
"Custom question ↓",
"What are the top 5 business insights from this data?",
"Which columns are most useful for predicting outcomes?",
"What data quality issues should I fix before analysis?",
"Explain the key distributions and what they mean for the business",
"What KPIs can I derive from this dataset?",
"How should I handle the missing values and outliers?",
"What visualizations would best communicate insights from this data?",
"Suggest a data-driven strategy based on the patterns in this dataset",
], key="ai_quick")

question = st.text_area("Your question",
value="" if quick=="Custom question ↓" else quick,
placeholder="e.g. Which customer segments drive the most revenue?",
height=85, key="ai_q")

if st.button("🔍 Analyze", use_container_width=False):
if not question.strip():
st.warning("Type a question above or pick one from the list.")
else:
summary = df.describe(include="all").round(3).to_string()
miss_info = df.isnull().sum().to_string()
finding_txt= "\n".join([f"{i}. {t.replace('','').replace('','')}" for i,(_,t) in enumerate(findings,1)])

prompt = f"""You are a senior data analyst and business intelligence expert. Analyze this dataset thoroughly and answer the user's question.

DATASET PROFILE:

Shape: {df.shape[0]:,} rows × {df.shape[1]} columns

Numeric columns: {num_cols}

Categorical columns: {cat_cols}

Date columns: {date_cols}

Missing: {miss_pct}% | Duplicates: {dupes}

STATISTICAL SUMMARY:
{summary}

MISSING VALUES:
{miss_info}

AUTO-DETECTED FINDINGS:
{finding_txt}

SAMPLE DATA (first 5 rows):
{df.head(5).to_string()}

USER QUESTION: {question}

Respond as a data professional presenting to a business stakeholder. Be specific — reference actual column names, numbers, and percentages from the data. Use structured sections with clear headers. Include:

Direct answer to the question

Supporting evidence from the data

Business implications

Recommended next steps or actions
Keep it concise, sharp, and actionable."""

with st.spinner("AI is analyzing your data..."):
    try:

        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        st.success("Analysis complete")

        st.markdown(
            f'<div class="insight-card"><p>{answer}</p></div>',
            unsafe_allow_html=True
        )

        buf = io.StringIO()
        buf.write(
            f"DataLens AI\nQuestion: {question}\n\n{answer}"
        )

        st.download_button(
            "⬇️ Download analysis",
            buf.getvalue(),
            file_name="datalens_ai_analysis.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"AI error: {e}")
