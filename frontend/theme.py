from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ignis-bg: #080b12;
  --ignis-panel: #111826;
  --ignis-panel-2: #0d1420;
  --ignis-line: rgba(148, 163, 184, 0.16);
  --ignis-text: #e9eef7;
  --ignis-muted: #94a3b8;
  --ignis-ember: #ff6b2c;
  --ignis-amber: #ffb020;
  --ignis-gold: #ffd166;
  --ignis-good: #34d399;
  --ignis-bad: #fb7185;
  --ignis-warn: #fbbf24;
  --ignis-info: #60a5fa;
  --ignis-radius: 16px;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(255, 107, 44, 0.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(96, 165, 250, 0.10), transparent 55%),
    radial-gradient(700px 400px at 50% 120%, rgba(255, 176, 32, 0.08), transparent 60%),
    var(--ignis-bg);
  color: var(--ignis-text);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b111c 0%, #080b12 100%);
  border-right: 1px solid var(--ignis-line);
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.1rem;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMarkdown {
  color: var(--ignis-text);
}

[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
}

#MainMenu, footer {
  visibility: hidden;
}

.block-container {
  padding-top: 1.6rem;
  padding-bottom: 4rem;
  max-width: 1220px;
}

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: -0.02em;
}

code, pre, .ignis-mono {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}

.ignis-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.2rem 0 0.9rem 0;
}

.ignis-sidebar-mark {
  width: 42px;
  height: 42px;
  border-radius: 13px;
  display: grid;
  place-items: center;
  font-size: 1.25rem;
  background: linear-gradient(135deg, #ff6b2c 0%, #ffb020 55%, #ffd166 100%);
  color: #14100a;
  box-shadow: 0 0 26px rgba(255, 107, 44, 0.42);
  font-weight: 700;
}

.ignis-sidebar-name {
  font-weight: 700;
  font-size: 1.15rem;
  letter-spacing: 0.02em;
}

.ignis-sidebar-tag {
  font-size: 0.7rem;
  color: var(--ignis-muted);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.ignis-hero {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--ignis-line);
  border-radius: 24px;
  padding: 2.1rem 2.2rem 2rem 2.2rem;
  background:
    radial-gradient(600px 240px at 12% 0%, rgba(255, 107, 44, 0.20), transparent 60%),
    radial-gradient(520px 240px at 100% 10%, rgba(96, 165, 250, 0.14), transparent 55%),
    linear-gradient(135deg, rgba(17, 24, 38, 0.94), rgba(9, 13, 21, 0.96));
  box-shadow: 0 30px 60px -40px rgba(0, 0, 0, 0.9);
  margin-bottom: 1.25rem;
}

.ignis-hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.06) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(600px 300px at 20% 0%, black, transparent 75%);
  pointer-events: none;
}

.ignis-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ignis-gold);
  margin-bottom: 0.75rem;
}

.ignis-kicker::before {
  content: "";
  width: 22px;
  height: 2px;
  background: linear-gradient(90deg, var(--ignis-ember), var(--ignis-gold));
  border-radius: 2px;
}

.ignis-title {
  font-size: clamp(1.7rem, 3.2vw, 2.7rem);
  line-height: 1.08;
  font-weight: 700;
  margin: 0 0 0.55rem 0;
  background: linear-gradient(120deg, #ffffff 0%, #ffe6d5 42%, #ffb020 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.ignis-subtitle {
  color: var(--ignis-muted);
  font-size: 1rem;
  line-height: 1.6;
  max-width: 70ch;
  margin: 0;
}

.ignis-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 1.1rem;
}

.ignis-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.24rem 0.62rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  border: 1px solid var(--ignis-line);
  background: rgba(148, 163, 184, 0.10);
  color: #cbd5e1;
  white-space: nowrap;
}

.ignis-badge--good {
  color: #a7f3d0;
  border-color: rgba(52, 211, 153, 0.35);
  background: rgba(52, 211, 153, 0.12);
}

.ignis-badge--bad {
  color: #fecdd3;
  border-color: rgba(251, 113, 133, 0.38);
  background: rgba(251, 113, 133, 0.14);
}

.ignis-badge--warn {
  color: #fde68a;
  border-color: rgba(251, 191, 36, 0.4);
  background: rgba(251, 191, 36, 0.13);
}

.ignis-badge--info {
  color: #bfdbfe;
  border-color: rgba(96, 165, 250, 0.4);
  background: rgba(96, 165, 250, 0.13);
}

.ignis-badge--ember {
  color: #ffd9c2;
  border-color: rgba(255, 107, 44, 0.45);
  background: rgba(255, 107, 44, 0.15);
}

.ignis-badge--muted {
  color: var(--ignis-muted);
}

.ignis-card {
  border: 1px solid var(--ignis-line);
  border-radius: var(--ignis-radius);
  padding: 1rem 1.1rem;
  background:
    linear-gradient(160deg, rgba(17, 24, 38, 0.92), rgba(11, 17, 28, 0.94));
  box-shadow: 0 18px 34px -28px rgba(0, 0, 0, 0.95);
  height: 100%;
}

.ignis-card__label {
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ignis-muted);
  margin-bottom: 0.35rem;
}

.ignis-card__value {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

.ignis-card__value--ember {
  background: linear-gradient(120deg, #ffd9c2, #ff6b2c);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.ignis-card__sub {
  font-size: 0.78rem;
  color: var(--ignis-muted);
  margin-top: 0.4rem;
  line-height: 1.45;
}

.ignis-callout {
  border: 1px solid var(--ignis-line);
  border-left: 3px solid var(--ignis-ember);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  background: rgba(255, 107, 44, 0.06);
  margin: 0.6rem 0 1rem 0;
}

.ignis-callout--warn {
  border-left-color: var(--ignis-warn);
  background: rgba(251, 191, 36, 0.07);
}

.ignis-callout--bad {
  border-left-color: var(--ignis-bad);
  background: rgba(251, 113, 133, 0.08);
}

.ignis-callout--good {
  border-left-color: var(--ignis-good);
  background: rgba(52, 211, 153, 0.07);
}

.ignis-callout__title {
  font-weight: 700;
  font-size: 0.86rem;
  margin-bottom: 0.35rem;
  letter-spacing: 0.02em;
}

.ignis-callout ul {
  margin: 0.2rem 0 0 1.1rem;
  padding: 0;
}

.ignis-callout li {
  font-size: 0.83rem;
  color: #cbd5e1;
  line-height: 1.5;
  margin-bottom: 0.18rem;
}

.ignis-answer {
  border: 1px solid var(--ignis-line);
  border-radius: 18px;
  padding: 1.35rem 1.5rem;
  background: linear-gradient(165deg, rgba(17, 24, 38, 0.9), rgba(9, 13, 21, 0.96));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.ignis-answer p {
  font-size: 0.97rem;
  line-height: 1.72;
}

.ignis-answer h1, .ignis-answer h2, .ignis-answer h3 {
  margin-top: 1.1rem;
}

.ignis-answer a {
  color: var(--ignis-gold);
  text-decoration: underline;
  text-decoration-color: rgba(255, 209, 102, 0.35);
  text-underline-offset: 3px;
}

.ignis-answer a:hover {
  color: #fff1c9;
}

.ignis-source {
  display: block;
  border: 1px solid var(--ignis-line);
  border-radius: 12px;
  padding: 0.7rem 0.85rem;
  margin-bottom: 0.5rem;
  background: rgba(15, 22, 34, 0.75);
  transition: border-color 0.18s ease, transform 0.18s ease;
}

.ignis-source:hover {
  border-color: rgba(255, 107, 44, 0.5);
  transform: translateY(-1px);
}

.ignis-source__title {
  font-size: 0.88rem;
  font-weight: 600;
  color: #f1f5f9;
  line-height: 1.35;
}

.ignis-source__meta {
  font-size: 0.72rem;
  color: var(--ignis-muted);
  margin-top: 0.28rem;
  word-break: break-all;
}

.ignis-source a {
  color: var(--ignis-gold);
  text-decoration: none;
}

.ignis-step {
  border: 1px solid var(--ignis-line);
  border-radius: 13px;
  padding: 0.75rem 0.95rem;
  margin-bottom: 0.5rem;
  background: rgba(15, 22, 34, 0.72);
}

.ignis-step__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.ignis-step__agent {
  font-weight: 600;
  font-size: 0.9rem;
}

.ignis-step__meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  font-size: 0.72rem;
  color: var(--ignis-muted);
}

.ignis-step__body {
  font-size: 0.8rem;
  color: #cbd5e1;
  line-height: 1.5;
  margin-top: 0.5rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.ignis-record {
  border: 1px solid var(--ignis-line);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  margin-bottom: 0.7rem;
  background: linear-gradient(165deg, rgba(17, 24, 38, 0.85), rgba(10, 15, 24, 0.9));
}

.ignis-record__title {
  font-size: 0.98rem;
  font-weight: 600;
  line-height: 1.35;
  color: #f8fafc;
}

.ignis-record__title a {
  color: #f8fafc;
  text-decoration: none;
}

.ignis-record__title a:hover {
  color: var(--ignis-gold);
}

.ignis-record__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.45rem 0 0.2rem 0;
}

.ignis-record__abstract {
  font-size: 0.82rem;
  line-height: 1.55;
  color: #b8c4d6;
  margin-top: 0.5rem;
}

.ignis-passage {
  border-left: 2px solid rgba(255, 107, 44, 0.55);
  border-radius: 4px 12px 12px 4px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.6rem;
  background: rgba(15, 22, 34, 0.72);
  font-size: 0.84rem;
  line-height: 1.6;
  color: #cbd5e1;
}

.ignis-passage__head {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  flex-wrap: wrap;
  font-size: 0.74rem;
  color: var(--ignis-muted);
  margin-bottom: 0.4rem;
}

.ignis-muted {
  color: var(--ignis-muted);
  font-size: 0.84rem;
}

.ignis-mono-block {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.76rem;
  line-height: 1.55;
  color: #c7d2fe;
  background: rgba(8, 12, 20, 0.85);
  border: 1px solid var(--ignis-line);
  border-radius: 10px;
  padding: 0.7rem 0.85rem;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 320px;
  overflow: auto;
}

.ignis-empty {
  border: 1px dashed rgba(148, 163, 184, 0.28);
  border-radius: 14px;
  padding: 1.1rem;
  color: var(--ignis-muted);
  font-size: 0.85rem;
  text-align: center;
  background: rgba(15, 22, 34, 0.4);
}

.ignis-section-title {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin: 0.2rem 0 0.55rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.ignis-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--ignis-line), transparent);
  margin: 1.1rem 0;
}

.ignis-metric-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.4rem 0 0.8rem 0;
}

.ignis-graph-note {
  font-size: 0.74rem;
  color: var(--ignis-muted);
  margin-top: 0.4rem;
}

div[data-testid="stButton"] > button {
  border-radius: 11px;
  border: 1px solid rgba(255, 107, 44, 0.45);
  background: linear-gradient(135deg, rgba(255, 107, 44, 0.95), rgba(255, 176, 32, 0.95));
  color: #16100a;
  font-weight: 700;
  letter-spacing: 0.01em;
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
  box-shadow: 0 12px 26px -18px rgba(255, 107, 44, 0.9);
}

div[data-testid="stButton"] > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.06);
  box-shadow: 0 18px 34px -18px rgba(255, 107, 44, 0.95);
  border-color: rgba(255, 209, 102, 0.8);
  color: #16100a;
}

div[data-testid="stButton"] > button:focus {
  color: #16100a;
}

div[data-testid="stDownloadButton"] > button,
div[data-testid="stButton"] > button[kind="secondary"] {
  background: rgba(148, 163, 184, 0.10);
  border: 1px solid var(--ignis-line);
  color: var(--ignis-text);
  font-weight: 600;
  box-shadow: none;
}

div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: rgba(255, 107, 44, 0.45);
  color: #fff;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
  background: rgba(11, 17, 28, 0.85) !important;
  border-color: var(--ignis-line) !important;
  border-radius: 11px !important;
  color: var(--ignis-text) !important;
}

textarea, input {
  color: var(--ignis-text) !important;
}

div[data-testid="stExpander"] {
  border: 1px solid var(--ignis-line);
  border-radius: 12px;
  background: rgba(15, 22, 34, 0.55);
  overflow: hidden;
}

div[data-testid="stExpander"] summary {
  font-size: 0.85rem;
  font-weight: 600;
}

div[data-testid="stTabs"] button {
  font-weight: 600;
  color: var(--ignis-muted);
}

div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--ignis-gold);
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, var(--ignis-ember), var(--ignis-gold));
}

div[data-testid="stMetric"] {
  border: 1px solid var(--ignis-line);
  border-radius: 13px;
  padding: 0.7rem 0.85rem;
  background: rgba(15, 22, 34, 0.65);
}

div[data-testid="stMetricLabel"] {
  color: var(--ignis-muted);
}

div[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, var(--ignis-ember), var(--ignis-gold));
}

hr {
  border-color: var(--ignis-line);
}

[data-testid="stFileUploader"] section {
  border-radius: 13px;
  border-color: var(--ignis-line);
  background: rgba(15, 22, 34, 0.5);
}

[data-testid="stFileUploader"] section:hover {
  border-color: rgba(255, 107, 44, 0.45);
}

[data-testid="stAlert"] {
  border-radius: 12px;
}

[data-testid="stCodeBlock"] {
  border-radius: 12px;
}

::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.25);
  border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 107, 44, 0.5);
}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="ignis-hero">
          <div class="ignis-kicker">{eyebrow}</div>
          <div class="ignis-title">{title}</div>
          <p class="ignis-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, badge_html: str = "") -> None:
    st.markdown(
        f'<div class="ignis-section-title">{title} {badge_html}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="ignis-divider"></div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="ignis-empty">{message}</div>', unsafe_allow_html=True)


def callout(title: str, items: list[str], variant: str = "") -> None:
    if not items:
        return
    list_items = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="ignis-callout {('ignis-callout--' + variant) if variant else ''}">
          <div class="ignis-callout__title">{title}</div>
          <ul>{list_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
