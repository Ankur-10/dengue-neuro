import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import shap

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Neuro Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2em; font-weight: 700; }
    .main-header p  { margin: 5px 0 0; opacity: 0.85; font-size: 0.95em; }

    .result-cns {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #1565c0;
        padding: 20px 25px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .result-pns {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #2e7d32;
        padding: 20px 25px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .result-na {
        background: linear-gradient(135deg, #fafafa, #f5f5f5);
        border-left: 5px solid #9e9e9e;
        padding: 20px 25px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .badge-cns {
        background: #1565c0; color: white;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; font-weight: 600; display: inline-block;
    }
    .badge-pns {
        background: #2e7d32; color: white;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; font-weight: 600; display: inline-block;
    }
    .section-header {
        font-size: 1.05em; font-weight: 600; color: #37474f;
        border-bottom: 2px solid #e0e0e0; padding-bottom: 6px;
        margin: 18px 0 12px;
    }
    .stAlert { border-radius: 8px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border-radius: 8px; padding: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Model Training ──────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_excel(r'data\Newdata_dengue_neuro.xlsx')

    # Drop F/U columns, mostly-empty cols, address, and identifier cols
    drop_cols = [
        'name', 'add',                              # identifiers
        'PAST EPI', 'POWER', 'adm_GBS', 'F/U_GBS', # >50% null (CPK and NCS_pattern kept as optional features)
        'F/U_MBI', 'F/U_MRS',                       # F/U columns
        'KFT', 'LFT',                               # binary flags duplicate of detail cols
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop rows where target is null
    df = df.dropna(subset=['CNS/PNS', 'diagnosis'])

    # Fill remaining NaN with median (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

    # Encode sex (already 0/1 but ensure int)
    df['sex'] = df['sex'].astype(int)

    # ── Feature sets ──
    # Shared features
    shared = ['age', 'sex', 'fever ', 'rash', 'bleeding',
              'vitals', 'GCS', 'Hb', 'TLC', 'neutrophls', 'lymphocytes',
              'Hct', 'platelet', 'Na', 'K', 'RBS',
              'CREAT', 'UREA', 'SGOT', 'SGPT', 'Billirubin',
              'protein', 'albumin']

    # CNS-specific features
    cns_features = ['headache', 'alteredsensorium', 'siezure',
                    'CN involv', 'B/B invol', 'CN deficit',
                    'adm _MBI', 'adm_MRS',
                    'CSF_TLC', 'CSF_P', 'CSF_L', 'CSF_protein', 'CSF_sugar']

    # PNS-specific features
    pns_features = ['weakness', 'sensory', 'CPK', 'NCS_pattern']

    all_feats = list(dict.fromkeys(shared + cns_features + pns_features))
    all_feats = [f for f in all_feats if f in df.columns]

    X_all = df[all_feats]
    y_cns = (df['CNS/PNS'] == 'CNS').astype(int)

    # Step-1 model: CNS vs PNS
    rf_step1 = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       random_state=42, class_weight='balanced')
    rf_step1.fit(X_all, y_cns)

    # Step-2a model: CNS diagnosis
    all_symptoms = cns_features + pns_features
    cns_mask = df['CNS/PNS'] == 'CNS'
    cns_feats = [f for f in shared + all_symptoms if f in df.columns]
    X_cns = df[cns_mask][cns_feats]
    y_cns_diag = df[cns_mask]['diagnosis']
    le_cns = LabelEncoder()
    y_cns_enc = le_cns.fit_transform(y_cns_diag)
    rf_cns = RandomForestClassifier(n_estimators=200, max_depth=6,
                                     random_state=42, class_weight='balanced')
    rf_cns.fit(X_cns, y_cns_enc)

    # Step-2b model: PNS diagnosis
    pns_mask = df['CNS/PNS'] == 'PNS'
    pns_feats = [f for f in shared + all_symptoms if f in df.columns]
    X_pns = df[pns_mask][pns_feats]
    y_pns_diag = df[pns_mask]['diagnosis']
    le_pns = LabelEncoder()
    y_pns_enc = le_pns.fit_transform(y_pns_diag)
    rf_pns = RandomForestClassifier(n_estimators=200, max_depth=6,
                                     random_state=42, class_weight='balanced')
    rf_pns.fit(X_pns, y_pns_enc)

    return (rf_step1, rf_cns, rf_pns,
            le_cns, le_pns,
            all_feats, cns_feats, pns_feats,
            X_all, X_cns, X_pns)

models = load_and_train()
(rf_step1, rf_cns, rf_pns,
 le_cns, le_pns,
 all_feats, cns_feats, pns_feats,
 X_all, X_cns, X_pns) = models

# ─── SHAP explainer (cached) ─────────────────────────────────────────────────
@st.cache_resource
def get_explainers():
    ex1   = shap.TreeExplainer(rf_step1)
    ex_c  = shap.TreeExplainer(rf_cns)
    ex_p  = shap.TreeExplainer(rf_pns)
    return ex1, ex_c, ex_p

ex1, ex_c, ex_p = get_explainers()

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 Dengue Neurological Complication Predictor</h1>
  <p>AI-assisted CNS / PNS classification with diagnosis prediction and SHAP explainability</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar – Patient Info ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Patient Information")
    pat_name = st.text_input("Patient Name", placeholder="e.g. Ramesh Kumar")
    pat_age  = st.number_input("Age (years)", min_value=1, max_value=120, value=35, step=1)
    pat_sex  = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")
    st.markdown("### 📋 Diagnosis Reference")
    st.markdown("""
    **CNS:**
    - Encephalitis  
    - Encephalopathy  
    - Seizure Disorder  
    - Stroke  
    - Transverse Myelitis  

    **PNS:**
    - Guillain-Barré Syndrome  
    - Hypokalemic Periodic Paralysis  
    - Myopathy  

   
    """)

# ─── Main Input Layout ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🩺 Clinical Symptoms</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**General Symptoms**")
    fever      = st.selectbox("Fever",               ["No", "Yes"])
    rash       = st.selectbox("Rash",                ["No", "Yes"])
    bleeding   = st.selectbox("Bleeding",            ["No", "Yes"])
    vitals_abn = st.selectbox("Vitals Abnormal",     ["No", "Yes"])
    gcs        = st.number_input("GCS Score", min_value=3, max_value=15, value=15, step=1)

with col2:
    st.markdown("**Neurological**")
    headache   = st.selectbox("Headache",            ["No", "Yes"])
    alt_sen    = st.selectbox("Altered Sensorium",   ["No", "Yes"])
    seizure    = st.selectbox("Seizure",             ["No", "Yes"])
    weakness   = st.selectbox("Weakness",            ["No", "Yes"])
    sensory    = st.selectbox("Sensory Deficit",     ["No", "Yes"])

with col3:
    st.markdown("**Neurological Signs**")
    cn_involv  = st.selectbox("Cranial Nerve Involvement",   ["No", "Yes"])
    bb_invol   = st.selectbox("Bowel/Bladder Involvement",   ["No", "Yes"])
    cn_deficit = st.selectbox("Cranial Nerve Deficit",       ["No", "Yes"])
    adm_mbi    = st.number_input("Admission MBI Score",   min_value=0, max_value=20, value=10, step=1)
    adm_mrs    = st.number_input("Admission MRS Score",   min_value=0, max_value=6,  value=2,  step=1)

st.markdown('<div class="section-header">🔬 Laboratory Values</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**Haematology**")
    hb         = st.number_input("Haemoglobin (g/dL)",   min_value=0.0, max_value=20.0, value=12.0, step=0.1, format="%.1f")
    tlc        = st.number_input("TLC (cells/µL)",        min_value=0,   max_value=50000, value=8000, step=100)
    neutro     = st.number_input("Neutrophils (%)",       min_value=0,   max_value=100,   value=65,   step=1)
    lympho     = st.number_input("Lymphocytes (%)",       min_value=0,   max_value=100,   value=30,   step=1)
    hct        = st.number_input("Haematocrit (%)",       min_value=0.0, max_value=70.0,  value=38.0, step=0.5, format="%.1f")
    platelet   = st.number_input("Platelet (×10³/µL)",   min_value=0.0, max_value=1000.0,value=150.0,step=1.0, format="%.1f")

with col5:
    st.markdown("**Electrolytes & Metabolic**")
    na         = st.number_input("Sodium (mEq/L)",        min_value=100, max_value=170,  value=138, step=1)
    k          = st.number_input("Potassium (mEq/L)",     min_value=1.0, max_value=8.0,  value=4.0, step=0.1, format="%.1f")
    rbs        = st.number_input("Random Blood Sugar (mg/dL)", min_value=40, max_value=600, value=110, step=1)
    creat      = st.number_input("Creatinine (mg/dL)",    min_value=0.0, max_value=20.0, value=0.9, step=0.1, format="%.1f")
    urea       = st.number_input("Urea (mg/dL)",          min_value=0.0, max_value=200.0,value=25.0,step=0.5, format="%.1f")

with col6:
    st.markdown("**LFT & CSF**")
    sgot       = st.number_input("SGOT (U/L)",            min_value=0,   max_value=2000, value=35,   step=1)
    sgpt       = st.number_input("SGPT (U/L)",            min_value=0,   max_value=2000, value=30,   step=1)
    bili       = st.number_input("Bilirubin (mg/dL)",     min_value=0.0, max_value=30.0, value=0.8,  step=0.1, format="%.1f")
    protein    = st.number_input("Protein (g/dL)",        min_value=0.0, max_value=10.0, value=6.5,  step=0.1, format="%.1f")
    albumin    = st.number_input("Albumin (g/dL)",        min_value=0.0, max_value=6.0,  value=3.5,  step=0.1, format="%.1f")

st.markdown('<div class="section-header">🧪 CSF Analysis (if available)</div>', unsafe_allow_html=True)
csf_available = st.checkbox("CSF analysis was performed", value=False)
if csf_available:
    col7, col8, col9, col10, col11 = st.columns(5)
    csf_tlc  = col7.number_input("CSF TLC",              min_value=0.0, max_value=5000.0, value=0.0,  step=1.0, format="%.1f")
    csf_p    = col8.number_input("CSF Polymorphs (%)",   min_value=0.0, max_value=100.0,  value=0.0,  step=1.0, format="%.1f")
    csf_l    = col9.number_input("CSF Lymphocytes (%)",  min_value=0.0, max_value=100.0,  value=0.0,  step=1.0, format="%.1f")
    csf_prot = col10.number_input("CSF Protein (mg/dL)", min_value=0.0, max_value=500.0,  value=40.0, step=1.0, format="%.1f")
    csf_sug  = col11.number_input("CSF Sugar (mg/dL)",   min_value=0.0, max_value=300.0,  value=60.0, step=1.0, format="%.1f")
else:
    csf_tlc = csf_p = csf_l = 0.0
    csf_prot = 40.0   # population median used as neutral default
    csf_sug  = 60.0

# ── CPK (optional) ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💉 CPK — Creatine Phosphokinase (if available)</div>', unsafe_allow_html=True)
cpk_available = st.checkbox("CPK value is available", value=False,
                             help="Relevant for myopathy / PNS workup. Leave unchecked if not measured.")
if cpk_available:
    cpk_val = st.number_input("CPK (U/L)", min_value=0.0, max_value=100000.0,
                               value=100.0, step=10.0, format="%.1f",
                               help="Normal range: 30–170 U/L (F), 55–170 U/L (M). Elevated in myopathy / rhabdomyolysis.")
else:
    cpk_val = 0.0   # treated as missing → model median imputation equivalent

# ── NCS Pattern (optional) ────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ NCS / EMG Pattern (if available)</div>', unsafe_allow_html=True)
ncs_available = st.checkbox("NCS / EMG was performed", value=False,
                             help="Nerve Conduction Study. Relevant for GBS / peripheral neuropathy workup.")
if ncs_available:
    ncs_pattern_input = st.selectbox(
        "NCS Pattern",
        options=["Not done / Normal", "Abnormal (demyelinating / axonal / mixed)"],
        help="Select 'Abnormal' if NCS showed demyelinating, axonal, or mixed pattern."
    )
    ncs_val = 1.0 if ncs_pattern_input == "Abnormal (demyelinating / axonal / mixed)" else 0.0
else:
    ncs_val = 0.0

# ─── Predict Button ──────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")

if predict_btn:
    sex_int = 0 if pat_sex == "Female" else 1
    yn = lambda v: 1 if v == "Yes" else 0

    # Build full feature dict
    feat_vals = {
        'age':           pat_age,
        'sex':           sex_int,
        'fever ':        yn(fever),
        'rash':          yn(rash),
        'bleeding':      yn(bleeding),
        'alteredsensorium': yn(alt_sen),
        'headache':      yn(headache),
        'siezure':       yn(seizure),
        'weakness':      yn(weakness),
        'sensory':       yn(sensory),
        'CN involv':     yn(cn_involv),
        'B/B invol':     yn(bb_invol),
        'vitals':        yn(vitals_abn),
        'GCS':           gcs,
        'CN deficit':    yn(cn_deficit),
        'adm _MBI':      adm_mbi,
        'adm_MRS':       adm_mrs,
        'Hb':            hb,
        'TLC':           tlc,
        'neutrophls':    neutro,
        'lymphocytes':   lympho,
        'Hct':           hct,
        'platelet':      platelet,
        'Na':            na,
        'K':             k,
        'RBS':           rbs,
        'CREAT':         creat,
        'UREA':          urea,
        'SGOT':          sgot,
        'SGPT':          sgpt,
        'Billirubin':    bili,
        'protein':       protein,
        'albumin':       albumin,
        'CSF_TLC':       csf_tlc,
        'CSF_P':         csf_p,
        'CSF_L':         csf_l,
        'CSF_protein':   csf_prot,
        'CSF_sugar':     csf_sug,
        'CPK':           cpk_val,
        'NCS_pattern':   ncs_val,
    }

    X_input = pd.DataFrame([[feat_vals.get(f, 0) for f in all_feats]], columns=all_feats)

    # ── Step 1: CNS vs PNS ──
    cns_prob = rf_step1.predict_proba(X_input)[0][1]
    pns_prob = 1 - cns_prob
    pred_system = "CNS" if cns_prob >= 0.5 else "PNS"

    # ── Step 2: Specific diagnosis ──
    name_display = pat_name.strip() if pat_name.strip() else "Patient"

    st.markdown("---")
    st.markdown(f"### 📊 Results for **{name_display}** | {pat_age}y {pat_sex}")

    r1, r2 = st.columns(2)
    r1.metric("CNS Probability", f"{cns_prob*100:.1f}%")
    r2.metric("PNS Probability", f"{pns_prob*100:.1f}%")

    # Probability bar
    fig_bar, ax_bar = plt.subplots(figsize=(7, 1.1))
    ax_bar.barh(0, cns_prob, color="#1565c0", height=0.5, label="CNS")
    ax_bar.barh(0, pns_prob, left=cns_prob, color="#2e7d32", height=0.5, label="PNS")
    ax_bar.set_xlim(0, 1); ax_bar.axis('off')
    ax_bar.legend(loc='lower right', ncol=2, fontsize=9, framealpha=0.7)
    ax_bar.set_title("CNS vs PNS Confidence", fontsize=10, pad=5)
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    if pred_system == "CNS":
        X_cns_input = pd.DataFrame([[feat_vals.get(f, 0) for f in cns_feats]], columns=cns_feats)
        diag_probs  = rf_cns.predict_proba(X_cns_input)[0]
        diag_idx    = np.argmax(diag_probs)
        diag_label  = le_cns.inverse_transform([diag_idx])[0]
        diag_conf   = diag_probs[diag_idx] * 100

        full_names = {'E': 'Encephalitis', 'En': 'Encephalopathy',
                      'SD': 'Seizure Disorder',
                      'STROKE': 'Stroke', 'TM': 'Transverse Myelitis'}

        st.markdown(f"""
        <div class="result-cns">
          <span class="badge-cns">CNS Involvement</span>
          <h3 style="margin:10px 0 4px;color:#1565c0">
            {full_names.get(diag_label, diag_label)}
          </h3>
          <p style="margin:0;color:#555">Confidence: <b>{diag_conf:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # CNS diagnosis probabilities
        st.markdown("**Diagnosis probability breakdown:**")
        diag_df = pd.DataFrame({
            'Diagnosis': [full_names.get(le_cns.classes_[i], le_cns.classes_[i])
                          for i in range(len(le_cns.classes_))],
            'Probability': diag_probs
        }).sort_values('Probability', ascending=False)
        fig_d, ax_d = plt.subplots(figsize=(7, 3))
        colors = ['#1565c0' if i == 0 else '#90caf9' for i in range(len(diag_df))]
        ax_d.barh(diag_df['Diagnosis'], diag_df['Probability'], color=colors)
        ax_d.set_xlim(0, 1); ax_d.set_xlabel('Probability')
        ax_d.invert_yaxis(); ax_d.grid(axis='x', alpha=0.3)
        fig_d.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        # ── SHAP for Step 1 ──
        st.markdown("---")
        st.markdown("### 🔍 SHAP Explainability")

        tab1, tab2 = st.tabs(["CNS vs PNS Decision", "Specific Diagnosis (CNS)"])

        with tab1:
            shap_vals1 = ex1.shap_values(X_input)
            sv_arr1 = np.array(shap_vals1)
            # shape: (samples, features, classes) → take sample 0, class 1 (CNS)
            sv = sv_arr1[0, :, 1] if sv_arr1.ndim == 3 else sv_arr1[0]
            sv_series = pd.Series(sv, index=all_feats)
            top = sv_series.abs().nlargest(12).index
            sv_top = sv_series[top].sort_values()

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors_shap = ['#d32f2f' if v > 0 else '#1565c0' for v in sv_top]
            ax1.barh(sv_top.index, sv_top.values, color=colors_shap)
            ax1.axvline(0, color='black', linewidth=0.8)
            ax1.set_title(f"Top Features Driving CNS Prediction\n(Red = pushes toward CNS, Blue = pushes toward PNS)",
                          fontsize=10)
            ax1.set_xlabel("SHAP Value"); ax1.grid(axis='x', alpha=0.3)
            red_p  = mpatches.Patch(color='#d32f2f', label='→ CNS')
            blue_p = mpatches.Patch(color='#1565c0', label='→ PNS')
            ax1.legend(handles=[red_p, blue_p], fontsize=9)
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            st.info("Each bar shows how much that feature pushed the prediction toward CNS (red) or PNS (blue).")

        with tab2:
            shap_vals_c = ex_c.shap_values(X_cns_input)
            sv_c_arr = np.array(shap_vals_c)
            sv_c = sv_c_arr[0, :, diag_idx] if sv_c_arr.ndim == 3 else sv_c_arr[0]
            sv_c_series = pd.Series(sv_c, index=cns_feats)
            top_c = sv_c_series.abs().nlargest(12).index
            sv_c_top = sv_c_series[top_c].sort_values()

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors_c = ['#d32f2f' if v > 0 else '#7b1fa2' for v in sv_c_top]
            ax2.barh(sv_c_top.index, sv_c_top.values, color=colors_c)
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_title(f"Top Features Driving '{full_names.get(diag_label, diag_label)}' Prediction",fontsize=10)
            ax2.set_xlabel("SHAP Value"); ax2.grid(axis='x', alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    else:  # PNS
        X_pns_input = pd.DataFrame([[feat_vals.get(f, 0) for f in pns_feats]], columns=pns_feats)
        diag_probs  = rf_pns.predict_proba(X_pns_input)[0]
        diag_idx    = np.argmax(diag_probs)
        diag_label  = le_pns.inverse_transform([diag_idx])[0]
        diag_conf   = diag_probs[diag_idx] * 100

        full_names_pns = {'GBS': 'Guillain-Barré Syndrome',
                          'HPP': 'Hypokalemic Periodic Paralysis',
                          'MYO': 'Myopathy'}

        st.markdown(f"""
        <div class="result-pns">
          <span class="badge-pns">PNS Involvement</span>
          <h3 style="margin:10px 0 4px;color:#2e7d32">
            {full_names_pns.get(diag_label, diag_label)}
          </h3>
          <p style="margin:0;color:#555">Confidence: <b>{diag_conf:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Diagnosis probability breakdown:**")
        diag_df = pd.DataFrame({
            'Diagnosis': [full_names_pns.get(le_pns.classes_[i], le_pns.classes_[i])
                          for i in range(len(le_pns.classes_))],
            'Probability': diag_probs
        }).sort_values('Probability', ascending=False)
        fig_d, ax_d = plt.subplots(figsize=(7, 2.5))
        colors = ['#2e7d32' if i == 0 else '#a5d6a7' for i in range(len(diag_df))]
        ax_d.barh(diag_df['Diagnosis'], diag_df['Probability'], color=colors)
        ax_d.set_xlim(0, 1); ax_d.set_xlabel('Probability')
        ax_d.invert_yaxis(); ax_d.grid(axis='x', alpha=0.3)
        fig_d.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        st.markdown("---")
        st.markdown("### 🔍 SHAP Explainability")

        tab1, tab2 = st.tabs(["CNS vs PNS Decision", "Specific Diagnosis (PNS)"])

        with tab1:
            shap_vals1 = ex1.shap_values(X_input)
            sv_arr1 = np.array(shap_vals1)
            sv = sv_arr1[0, :, 1] if sv_arr1.ndim == 3 else sv_arr1[0]
            sv_series = pd.Series(sv, index=all_feats)
            top = sv_series.abs().nlargest(12).index
            sv_top = sv_series[top].sort_values()

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors_shap = ['#d32f2f' if v > 0 else '#1565c0' for v in sv_top]
            ax1.barh(sv_top.index, sv_top.values, color=colors_shap)
            ax1.axvline(0, color='black', linewidth=0.8)
            ax1.set_title("Top Features Driving CNS/PNS Classification\n(Red = toward CNS, Blue = toward PNS)",
                          fontsize=10)
            ax1.set_xlabel("SHAP Value"); ax1.grid(axis='x', alpha=0.3)
            red_p  = mpatches.Patch(color='#d32f2f', label='→ CNS')
            blue_p = mpatches.Patch(color='#1565c0', label='→ PNS')
            ax1.legend(handles=[red_p, blue_p], fontsize=9)
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with tab2:
            shap_vals_p = ex_p.shap_values(X_pns_input)
            sv_p_arr = np.array(shap_vals_p)
            sv_p = sv_p_arr[0, :, diag_idx] if sv_p_arr.ndim == 3 else sv_p_arr[0]
            sv_p_series = pd.Series(sv_p, index=pns_feats)
            top_p = sv_p_series.abs().nlargest(12).index
            sv_p_top = sv_p_series[top_p].sort_values()

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors_p = ['#d32f2f' if v > 0 else '#7b1fa2' for v in sv_p_top]
            ax2.barh(sv_p_top.index, sv_p_top.values, color=colors_p)
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_title(f"Features Driving '{full_names_pns.get(diag_label, diag_label)}' Prediction", fontsize=10)
            ax2.set_xlabel("SHAP Value"); ax2.grid(axis='x', alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    st.markdown("---")
    st.caption("⚠️ This tool is intended for clinical decision support only. Final diagnosis must be made by a qualified physician.")

# ─── Footer – Model Info ──────────────────────────────────────────────────────

