import html
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.predict import SentimentPredictor

ROOT = Path(__file__).parent
CKPT = ROOT / "models" / "sentimentscope.pt"
METRICS = ROOT / "models" / "metrics.json"
SAMPLES = ROOT / "assets" / "sample_reviews.csv"
REPO_URL = "https://github.com/Kalestahr/sentiment-scope"
EXAMPLES = {
    "Enthusiastic": "The movie was a rollercoaster of emotions, and I loved every moment of it!",
    "Disappointed": "The plot was predictable, and the acting was subpar. A waste of time.",
    "Lukewarm": "Not the best film I've seen, but the soundtrack carried it.",
}
FONTS = "@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Inter:wght@400;500;600&display=swap');"

st.set_page_config(page_title="SentimentScope", page_icon="🎞️", layout="wide")
ss = st.session_state

st.markdown(f"""<style>
{FONTS}
html, body, p, li, label, .stMarkdown {{font-family: 'Inter', sans-serif;}}
h1, h2, h3 {{font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 600 !important; letter-spacing: -0.01em;}}
.block-container {{max-width: 1080px; padding-top: 2.5rem;}}
.eyebrow {{font: 600 12px Inter, sans-serif; letter-spacing: .12em; text-transform: uppercase; color: #8B1E3F;}}
.lede {{font-size: 1.1rem; color: #4a4a4a; max-width: 640px; line-height: 1.6;}}
.stats {{display: flex; flex-wrap: wrap; gap: 2.5rem; border-top: 1px solid #e3ddd5; border-bottom: 1px solid #e3ddd5; padding: 1.1rem 0; margin: 1.5rem 0 1rem;}}
.stat b {{display: block; font: 600 1.6rem 'Source Serif 4', serif; color: #1f1f1f;}}
.stat span {{font-size: .85rem; color: #6b6b6b;}}
.section {{font: 600 12px Inter, sans-serif; letter-spacing: .12em; text-transform: uppercase; color: #6b6b6b;
          border-top: 1px solid #e3ddd5; padding-top: 1.5rem; margin-top: 2.5rem;}}
.pill {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;}}
.pos {{background: #e6f2ea; color: #1e6b3d;}} .neg {{background: #f6e4e6; color: #8B1E3F;}} .mid {{background: #efece6; color: #5b5348;}}
.review {{border-left: 3px solid #c9c1b6; padding: .25rem 0 .25rem 1rem; font: 1.05rem/1.65 'Source Serif 4', serif; color: #2b2b2b;}}
.small {{font-size: .8rem; color: #6b6b6b; text-transform: uppercase; letter-spacing: .08em;}}
footer {{visibility: hidden;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return SentimentPredictor(str(CKPT), device="cpu")


@st.cache_data
def load_samples(_predictor):
    df = pd.read_csv(SAMPLES).reset_index(drop=True)
    df["model_pos"] = [p["positive_prob"] for p in _predictor.predict(df["review"].tolist())]
    df["correct"] = (df["model_pos"] > 0.5).astype(int) == df["label"]
    return df


def verdict(pos):
    if 0.4 <= pos <= 0.6:
        return "Uncertain", "mid"
    return ("Positive", "pos") if pos > 0.5 else ("Negative", "neg")


def pill(pos):
    text, cls = verdict(pos)
    return f'<span class="pill {cls}">{text}</span>'


def confidence(pos):
    return f"{max(pos, 1 - pos):.0%} confident"


def section(n, title):
    st.markdown(f'<div class="section">{n:02d} &middot; {title}</div>', unsafe_allow_html=True)


def ticker(df):
    def track(label, direction):
        cards = "".join(
            f'<div class="card"><div class="top"><span class="{"pos" if label else "neg"}">'
            f'{"Positive" if label else "Negative"} review</span>'
            f'<span class="{"" if r.correct else "miss"}">Model: {"correct" if r.correct else "missed"}</span></div>'
            f'<p>{html.escape(r.review[:220])}…</p></div>'
            for r in df[df["label"] == label].itertuples())
        return f'<div class="row"><div class="track {direction}">{cards}{cards}</div></div>'

    return f"""<style>
    {FONTS}
    body {{margin: 0; background: #FAF8F5;}}
    .row {{overflow: hidden; padding: 6px 0;}}
    .track {{display: flex; gap: 14px; width: max-content; animation: scroll 140s linear infinite;}}
    .track.right {{animation-direction: reverse;}}
    .row:hover .track {{animation-play-state: paused;}}
    @keyframes scroll {{from {{transform: translateX(0);}} to {{transform: translateX(-50%);}}}}
    @media (prefers-reduced-motion: reduce) {{.track {{animation: none;}}}}
    .card {{width: 290px; flex: none; background: #fff; border: 1px solid #e3ddd5; border-radius: 8px; padding: 14px 16px;}}
    .top {{display: flex; justify-content: space-between; font: 600 11px Inter, sans-serif;
           letter-spacing: .08em; text-transform: uppercase; color: #8a8278; margin-bottom: 8px;}}
    .pos {{color: #1e6b3d;}} .neg {{color: #8B1E3F;}} .miss {{color: #b3541e;}}
    p {{margin: 0; height: 92px; overflow: hidden; font: 14.5px/1.55 'Source Serif 4', Georgia, serif; color: #2b2b2b;}}
    </style>{track(1, "left")}{track(0, "right")}"""


# Masthead
st.markdown('<div class="eyebrow">Udacity AI Engineer Nanodegree</div>', unsafe_allow_html=True)
st.title("SentimentScope")
st.markdown('<p class="lede">A small transformer, trained from scratch on IMDB reviews, '
            'that reads a review and decides whether the writer liked the film.</p>', unsafe_allow_html=True)

if not CKPT.exists():
    st.error("No trained model found. Run notebooks/SentimentScope.ipynb, then add models/sentimentscope.pt.")
    st.stop()

predictor = load_predictor()
m = json.loads(METRICS.read_text()) if METRICS.exists() else None
acc = f"{m['test_acc']:.1f}%" if m else "n/a"
params = f"{m['params'] / 1e6:.1f}M" if m else "n/a"
st.markdown(f"""<div class="stats">
  <div class="stat"><b>{acc}</b><span>accuracy on 25,000 unseen reviews</span></div>
  <div class="stat"><b>22,500</b><span>reviews used for training</span></div>
  <div class="stat"><b>{params}</b><span>model parameters</span></div>
  <div class="stat"><b>~100 words</b><span>read from each review</span></div>
</div>""", unsafe_allow_html=True)

if SAMPLES.exists():
    samples = load_samples(predictor)

    section(1, "Reviews in the wild")
    st.caption("Real reviews from the IMDB test set. Hover over a row to pause it.")
    components.html(ticker(samples), height=330)

    section(2, "Your read against the model")
    st.caption("Read the review and decide how the writer felt. Then see how you and the model compare.")
    ss.setdefault("score", {"you": 0, "model": 0, "rounds": 0})
    if "pick" not in ss:
        ss.pick, ss.guess = random.randrange(len(samples)), None

    row = samples.iloc[ss.pick]
    truth, model_guess = int(row["label"]), int(row["model_pos"] > 0.5)
    st.markdown(f'<div class="review">{html.escape(row["review"])}</div>', unsafe_allow_html=True)
    st.write("")

    if ss.guess is None:
        g1, g2, _ = st.columns([1, 1, 3])
        for col, label, value in [(g1, "Positive", 1), (g2, "Negative", 0)]:
            if col.button(label, use_container_width=True):
                ss.guess = value
                ss.score["rounds"] += 1
                ss.score["you"] += value == truth
                ss.score["model"] += model_guess == truth
                st.rerun()
    else:
        r1, r2, r3 = st.columns(3)
        r1.markdown(f'<div class="small">The writer</div>{pill(float(truth))}', unsafe_allow_html=True)
        r2.markdown(f'<div class="small">You</div>{"Correct" if ss.guess == truth else "Incorrect"}',
                    unsafe_allow_html=True)
        r3.markdown(f'<div class="small">The model</div>{pill(row["model_pos"])} {confidence(row["model_pos"])}, '
                    f'{"correct" if model_guess == truth else "incorrect"}', unsafe_allow_html=True)
        st.write("")
        if st.button("Next review"):
            ss.pick, ss.guess = random.randrange(len(samples)), None
            st.rerun()

    s = ss.score
    if s["rounds"]:
        st.caption(f"Score after {s['rounds']} round{'s' if s['rounds'] > 1 else ''}: "
                   f"you {s['you']}, model {s['model']}.")

section(3, "Classify a review")
st.caption("Start from an example or paste your own.")
for col, (name, text) in zip(st.columns([1, 1, 1, 3])[:3], EXAMPLES.items()):
    col.button(name, use_container_width=True, on_click=lambda t=text: ss.update(own=t))
review = st.text_area("Review", key="own", height=140, label_visibility="collapsed",
                      placeholder="Paste a movie review here")
if st.button("Classify", type="primary") and review.strip():
    pos = predictor.predict(review)[0]["positive_prob"]
    st.markdown(f"### {verdict(pos)[0]}")
    st.progress(pos, text=f"{confidence(pos)} · {pos:.0%} positive")

section(4, "Classify in bulk")
st.caption("Upload a CSV with a column named review, or paste one review per line.")
c1, c2 = st.columns([3, 1])
up = c1.file_uploader("CSV file", type="csv", label_visibility="collapsed")
c2.download_button("Blank template", "review\nGreat movie!\nNot for me.\n", "template.csv", "text/csv",
                   use_container_width=True)
pasted = st.text_area("Reviews", height=120, label_visibility="collapsed", placeholder="One review per line")
texts = pd.read_csv(up)["review"].astype(str).tolist() if up else [l for l in pasted.splitlines() if l.strip()]

if st.button("Classify all", type="primary") and texts:
    res = predictor.predict(texts)
    df = pd.DataFrame({"Review": [r["review"] for r in res],
                       "Sentiment": [verdict(r["positive_prob"])[0] for r in res],
                       "Confidence": [f"{max(r['positive_prob'], r['negative_prob']):.0%}" for r in res]})
    liked = sum(r["positive_prob"] > 0.5 for r in res)
    st.markdown(f"**{liked} of {len(res)}** reviews are positive.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download results", df.to_csv(index=False), "sentimentscope_results.csv", "text/csv")

section(5, "About the model")
st.write("The model is a four-layer transformer built in PyTorch and trained on labeled IMDB reviews. "
         "It breaks each review into word pieces, reads up to 128 of them (roughly the first 100 words), "
         "averages what it learned across the text, and outputs the chance that the review is positive. "
         "Long reviews are judged by their opening, so a review that turns around at the end can fool it.")
if m:
    hist = pd.DataFrame(m["history"]).set_index("epoch")
    h1, h2 = st.columns(2)
    h1.caption("Training loss by epoch")
    h1.line_chart(hist["train_loss"], height=200)
    h2.caption("Validation accuracy by epoch (%)")
    h2.line_chart(hist["val_acc"], height=200)

st.markdown(f'<div class="section">Built by Kalestahr &middot; <a href="{REPO_URL}">Source on GitHub</a></div>',
            unsafe_allow_html=True)
