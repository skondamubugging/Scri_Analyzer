# 📈 NSE/BSE Quantitative Script Analyzer & Momentum Burst Engine

A multi-agent Streamlit dashboard for Indian equities (NSE/BSE) that combines:

- **Mark Minervini's Trend Template** — Stage 2 structural uptrend filter + VCP squeeze detection
- **Pradeep Bonde's (Stockbee) Momentum Burst Engine** — Day-1 breakout detection out of tight consolidation
- **News Sentiment Agent** — Google News RSS + VADER sentiment scoring
- **Decision Matrix Agent** — fuses technicals + sentiment into a single call: `STRONG BUY`, `SPECULATIVE BUY`, `HOLD`, `TAKE PROFIT / SELL`, or `AVOID`
- **Stage 2 Entry Screener** — a dedicated ranked view of scripts that are structurally in a Minervini Stage 2 uptrend **and** confirmed by RSI(14), MACD, and ADX(14) momentum readings **and** not contradicted by bearish news — surfacing stocks *building* toward a high-probability entry even before a Bonde burst day fires, ranked by a 0–100 confluence score

> ⚠️ **Disclaimer:** This is an educational/analytical tool. Signals are generated from historical price, volume, and public news headlines using fixed, disclosed rule sets. Nothing here is investment advice — verify independently and consult a registered advisor before trading.

---

## 1. Repository structure

```
.
├── app.py                        # Main Streamlit application (entry point)
├── requirements.txt              # Pinned Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml               # Theme + server settings
└── .github/
    └── workflows/
        └── ci.yml                # CI: installs deps + syntax-checks app.py on push/PR
```

---

## 2. Push this project to GitHub

If you're starting fresh (from the folder containing these files):

```bash
git init
git add .
git commit -m "Initial commit: NSE/BSE Momentum & Trend Scanner"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

If you already have a GitHub repo created, just clone it first, copy `app.py`,
`requirements.txt`, `.gitignore`, and the `.streamlit/` / `.github/` folders
into it, then run the `add` / `commit` / `push` steps above.

---

## 3. Run it locally (clone from GitHub)

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Streamlit will start a local server (usually at `http://localhost:8501`) and
open it in your browser automatically.

---

## 4. Deploy for free with Streamlit Community Cloud (recommended)

This is the simplest way to get a shareable, always-on link straight from
your GitHub repo — no server management required.

1. Push your code to a **public** (or Streamlit-Cloud-connected private) GitHub repo, as in Step 2.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**. Streamlit Cloud will read `requirements.txt` automatically and install everything.
6. Your app will be live at a URL like:
   `https://<your-app-name>.streamlit.app`

Any time you `git push` new commits to `main`, Streamlit Cloud **automatically redeploys** the app — that's the whole CI/CD loop, no extra configuration needed.

> 💡 If the app needs any secrets in the future (e.g. a paid news API key), add them under **App settings → Secrets** in Streamlit Cloud rather than committing them to the repo. Read them in code with `st.secrets["KEY_NAME"]`.

---

## 5. Alternative deployment options

### A. Render.com / Railway.app (always-on, free-tier available)
1. Connect your GitHub repo.
2. Set the **build command**: `pip install -r requirements.txt`
3. Set the **start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy — both platforms auto-redeploy on every push to `main`.

### B. Docker (self-hosted / any cloud VM)
Add this `Dockerfile` to the repo root if you want a containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t nse-bse-scanner .
docker run -p 8501:8501 nse-bse-scanner
```

### C. GitHub Codespaces (zero local setup)
1. On your repo page, click **Code → Codespaces → Create codespace on main**.
2. Once the Codespace opens, run in its terminal:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501
   ```
3. Codespaces will prompt you to open the forwarded port in the browser.

---

## 6. Continuous Integration (already included)

`.github/workflows/ci.yml` runs automatically on every push and pull request to `main`:
- Installs all dependencies from `requirements.txt`
- Byte-compiles `app.py` to catch syntax errors
- Parses the file with `ast` as a smoke test

This won't catch runtime/network issues (e.g. a bad ticker or a Yahoo Finance outage), but it guarantees nothing broken ever reaches `main` silently. You can extend this workflow later with `pytest` unit tests on the agent classes (`TechnicalDataAgent`, `BondeMomentumAgent` logic, `DecisionMatrixAgent`) using synthetic OHLCV data — the classes are pure functions/methods and don't require network access to test.

---

## 7. Usage notes

- Enter comma-separated NSE/BSE trading symbols in the sidebar **without** the exchange suffix (e.g. `TATAMOTORS, RELIANCE, TCS`) — `.NS` or `.BO` is appended automatically based on the exchange you select.
- The scanner table is color-coded: green = Strong/Speculative Buy, amber = Hold, red = Take Profit/Sell.
- Expand any ticker in the "Per-Stock Deep Dive" section for a candlestick chart, the full Minervini/Bonde checklist, and the news headlines driving the sentiment score.
- Yahoo Finance data and Google News RSS are both free/unauthenticated sources — expect occasional rate-limiting or missing headlines for thinly-covered small caps; the app degrades gracefully (shows "Neutral"/"No data") rather than crashing.
