import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Miami Open 2026 Predictor", page_icon="🎾", layout="wide")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'miami_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

bundle       = load_model()
model        = bundle['model']
scaler       = bundle['scaler']
feature_cols = bundle['features']

# ── Player database ───────────────────────────────────────────
players = {
    'Alcaraz':           {'rank': 1,  'elo': 2180, 'hard_win_pct': 0.72, 'form5': 0.8,  'form10': 0.85},
    'Sinner':            {'rank': 2,  'elo': 2160, 'hard_win_pct': 0.74, 'form5': 1.0,  'form10': 0.90},
    'Zverev':            {'rank': 3,  'elo': 2050, 'hard_win_pct': 0.65, 'form5': 0.6,  'form10': 0.65},
    'Musetti':           {'rank': 5,  'elo': 1950, 'hard_win_pct': 0.58, 'form5': 0.4,  'form10': 0.55},
    'De Minaur':         {'rank': 6,  'elo': 1980, 'hard_win_pct': 0.62, 'form5': 0.4,  'form10': 0.55},
    'Fritz':             {'rank': 7,  'elo': 1990, 'hard_win_pct': 0.64, 'form5': 0.6,  'form10': 0.60},
    'Auger-Aliassime':   {'rank': 8,  'elo': 1970, 'hard_win_pct': 0.61, 'form5': 0.8,  'form10': 0.65},
    'Shelton':           {'rank': 9,  'elo': 1960, 'hard_win_pct': 0.61, 'form5': 0.6,  'form10': 0.60},
    'Medvedev':          {'rank': 10, 'elo': 2040, 'hard_win_pct': 0.70, 'form5': 0.8,  'form10': 0.80},
    'Bublik':            {'rank': 11, 'elo': 1940, 'hard_win_pct': 0.58, 'form5': 0.4,  'form10': 0.50},
    'Ruud':              {'rank': 12, 'elo': 1930, 'hard_win_pct': 0.56, 'form5': 0.4,  'form10': 0.50},
    'Mensik':            {'rank': 13, 'elo': 1960, 'hard_win_pct': 0.60, 'form5': 0.6,  'form10': 0.62},
    'Cobolli':           {'rank': 14, 'elo': 1900, 'hard_win_pct': 0.55, 'form5': 0.6,  'form10': 0.58},
    'Khachanov':         {'rank': 15, 'elo': 1910, 'hard_win_pct': 0.56, 'form5': 0.4,  'form10': 0.50},
    'Rublev':            {'rank': 16, 'elo': 1940, 'hard_win_pct': 0.60, 'form5': 0.4,  'form10': 0.52},
    'Draper':            {'rank': 26, 'elo': 1970, 'hard_win_pct': 0.63, 'form5': 0.8,  'form10': 0.72},
    'Fils':              {'rank': 29, 'elo': 1920, 'hard_win_pct': 0.58, 'form5': 0.6,  'form10': 0.60},
    'Fonseca':           {'rank': 38, 'elo': 1900, 'hard_win_pct': 0.58, 'form5': 0.8,  'form10': 0.68},
    'Korda':             {'rank': 33, 'elo': 1890, 'hard_win_pct': 0.55, 'form5': 0.4,  'form10': 0.50},
    'Tiafoe':            {'rank': 20, 'elo': 1900, 'hard_win_pct': 0.56, 'form5': 0.4,  'form10': 0.50},
    'Paul':              {'rank': 23, 'elo': 1910, 'hard_win_pct': 0.58, 'form5': 0.6,  'form10': 0.58},
    'Tien':              {'rank': 21, 'elo': 1890, 'hard_win_pct': 0.55, 'form5': 0.6,  'form10': 0.58},
    'Tsitsipas':         {'rank': 30, 'elo': 1920, 'hard_win_pct': 0.60, 'form5': 0.4,  'form10': 0.52},
}

player_names = sorted(players.keys())

# ── Prediction function ───────────────────────────────────────
def predict_match(p1, p2, round_num=4):
    s1 = players[p1]
    s2 = players[p2]
    features = pd.DataFrame([[
        s2['rank']         - s1['rank'],
        s1['elo']          - s2['elo'],
        s1['hard_win_pct'] - s2['hard_win_pct'],
        s1['form5']        - s2['form5'],
        s1['form10']       - s2['form10'],
        0,
        np.sin(2 * np.pi * 3 / 12),
        np.cos(2 * np.pi * 3 / 12),
        1, round_num
    ]], columns=feature_cols)
    if scaler:
        prob = model.predict_proba(scaler.transform(features))[0][1]
    else:
        prob = model.predict_proba(features.values)[0][1]
    return float(prob)

# ── Simulation ────────────────────────────────────────────────
@st.cache_data
def run_simulation(n=10000):
    q1 = ['Alcaraz', 'Fonseca', 'Korda', 'Khachanov']
    q2 = ['Fritz', 'Draper', 'Ruud', 'Musetti']
    q3 = ['Zverev', 'Shelton', 'Mensik', 'Auger-Aliassime']
    q4 = ['Medvedev', 'Bublik', 'De Minaur', 'Sinner']
    all_players = q1 + q2 + q3 + q4

    prob_cache = {}
    for p1 in all_players:
        for p2 in all_players:
            if p1 != p2:
                for r in [4, 5, 6, 7]:
                    prob_cache[(p1, p2, r)] = predict_match(p1, p2, r)

    def fast_sim(p1, p2, r):
        prob = prob_cache.get((p1, p2, r), 0.5)
        return p1 if np.random.random() < prob else p2

    win_counts = {p: 0 for p in all_players}
    sf_counts  = {p: 0 for p in all_players}
    f_counts   = {p: 0 for p in all_players}

    for _ in range(n):
        r16_q1 = [fast_sim(q1[0], q1[1], 4), fast_sim(q1[2], q1[3], 4)]
        r16_q2 = [fast_sim(q2[0], q2[1], 4), fast_sim(q2[2], q2[3], 4)]
        r16_q3 = [fast_sim(q3[0], q3[1], 4), fast_sim(q3[2], q3[3], 4)]
        r16_q4 = [fast_sim(q4[0], q4[1], 4), fast_sim(q4[2], q4[3], 4)]
        qf1 = fast_sim(r16_q1[0], r16_q1[1], 5)
        qf2 = fast_sim(r16_q2[0], r16_q2[1], 5)
        qf3 = fast_sim(r16_q3[0], r16_q3[1], 5)
        qf4 = fast_sim(r16_q4[0], r16_q4[1], 5)
        sf1 = fast_sim(qf1, qf2, 6)
        sf2 = fast_sim(qf3, qf4, 6)
        for p in [qf1, qf2, qf3, qf4]:
            sf_counts[p] += 1
        for p in [sf1, sf2]:
            f_counts[p] += 1
        winner = fast_sim(sf1, sf2, 7)
        win_counts[winner] += 1

    results = []
    for p in all_players:
        results.append({
            'Player': p,
            'Title %': round(win_counts[p] / n * 100, 1),
            'Final %': round(f_counts[p] / n * 100, 1),
            'SF %':    round(sf_counts[p] / n * 100, 1),
        })
    return pd.DataFrame(results).sort_values('Title %', ascending=False).reset_index(drop=True)

# ── UI ────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#006400;'>🎾 Miami Open 2026 Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Random Forest · Hard Court ATP Data 2000–2025 · AUC 0.71</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["🏆 Tournament Simulation", "⚡ Head to Head"])

# ── TAB 1: Simulation ─────────────────────────────────────────
with tab1:
    st.markdown("### Miami Open 2026 — Title Probabilities")
    st.caption("Based on 10,000 Monte Carlo simulations of the actual draw")

    with st.spinner("Running simulations..."):
        sim_df = run_simulation(10000)

    # Bar chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    colors = ['#006400' if i == 0 else '#228B22' if i == 1 else '#90EE90' for i in range(len(sim_df))]
    bars = ax.barh(sim_df['Player'][::-1], sim_df['Title %'][::-1], color=colors[::-1])
    ax.set_xlabel('Title Probability (%)', fontsize=12)
    ax.set_title('Miami Open 2026 — Predicted Title Probabilities', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, sim_df['Title %'][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=10)
    ax.set_xlim(0, sim_df['Title %'].max() + 5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Full Results Table")
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

# ── TAB 2: Head to Head ───────────────────────────────────────
with tab2:
    st.markdown("### Head to Head Match Predictor")
    col1, col2, col3 = st.columns(3)

    with col1:
        p1 = st.selectbox("Player 1", player_names, index=player_names.index('Alcaraz'))
    with col3:
        p2 = st.selectbox("Player 2", player_names, index=player_names.index('Sinner'))
    with col2:
        round_choice = st.selectbox("Round", ['R32','R16','QF','SF','Final'])
        round_map = {'R32':3,'R16':4,'QF':5,'SF':6,'Final':7}

    if p1 == p2:
        st.warning("Please select two different players!")
    else:
        prob = predict_match(p1, p2, round_map[round_choice])

        st.markdown("---")
        st.markdown("### Win Probability")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<h2 style='color:#006400; text-align:center;'>{p1}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center; color:#006400;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
            st.progress(prob)
        with col_b:
            st.markdown(f"<h2 style='color:#cc0000; text-align:center;'>{p2}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center; color:#cc0000;'>{(1-prob)*100:.1f}%</h1>", unsafe_allow_html=True)
            st.progress(1-prob)

        st.markdown("---")
        if prob >= 0.65:
            st.success(f"🟢 {p1} is the clear favourite at {prob*100:.1f}%")
        elif prob <= 0.35:
            st.error(f"🔴 {p2} is the clear favourite at {(1-prob)*100:.1f}%")
        else:
            st.warning(f"⚖️ Closely contested — {p1} {prob*100:.1f}% vs {p2} {(1-prob)*100:.1f}%")

        # Show player stats comparison
        st.markdown("### Player Stats Comparison")
        s1, s2 = players[p1], players[p2]
        comp_df = pd.DataFrame({
            'Stat': ['ATP Rank', 'Hard Court Elo', 'Hard Win %', 'Last 5 Form', 'Last 10 Form'],
            p1: [s1['rank'], s1['elo'], f"{s1['hard_win_pct']*100:.0f}%",
                 f"{s1['form5']*100:.0f}%", f"{s1['form10']*100:.0f}%"],
            p2: [s2['rank'], s2['elo'], f"{s2['hard_win_pct']*100:.0f}%",
                 f"{s2['form5']*100:.0f}%", f"{s2['form10']*100:.0f}%"],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>Built by Paul Cassady · MSc Data Science · NTU · 2026</p>", unsafe_allow_html=True)