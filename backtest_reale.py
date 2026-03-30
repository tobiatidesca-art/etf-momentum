#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Reale ETF Momentum 2015-2026
"""

import argparse
import json
import math
import sys
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import yfinance as yf

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Backtest ETF Momentum')
parser.add_argument('--n_titoli',  type=int,   default=4,     help='Numero titoli per mese (1-4)')
parser.add_argument('--capitale',  type=float, default=10000, help='Capitale per titolo (EUR)')
parser.add_argument('--costo',     type=float, default=0.0,   help='Costo ingresso %% per trade (es. 0.1)')
args = parser.parse_args()

N_TITOLI            = max(1, min(4, args.n_titoli))
CAPITALE_PER_TITOLO = args.capitale
COSTO_INGRESSO_PCT  = args.costo

# ── Global constants ──────────────────────────────────────────────────────────
DATA_INIZIO   = date(2000, 1, 1)
_oggi         = date.today()
DATA_FINE     = date(_oggi.year, _oggi.month, 1)
DATA_OGGI     = _oggi
LOOKBACK_MESI = 3

# ── ETF Universe ──────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    'SMH':  ['NVDA', 'TSM', 'AVGO', 'ASML', 'AMD', 'QCOM', 'INTC', 'MU', 'AMAT', 'LRCX'],
    'SOXX': ['NVDA', 'AVGO', 'AMD',  'QCOM', 'TXN', 'AMAT', 'KLAC', 'LRCX', 'MU', 'ON'],
    'XLK':  ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'ACN',  'CSCO', 'IBM', 'ADBE'],
    'QQQ':  ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'COST', 'NFLX'],
    'ITA':  ['RTX', 'LMT', 'BA',   'NOC',  'GD',  'LHX',  'HII',  'TDG',  'AXON', 'CACI'],
    'XLE':  ['XOM', 'CVX', 'COP',  'EOG',  'SLB', 'MPC',  'PSX',  'VLO',  'HAL'],
}

# ── Download ──────────────────────────────────────────────────────────────────
def download_all_data():
    all_tickers = set(['SPY'])
    for etf, holdings in ETF_UNIVERSE.items():
        all_tickers.add(etf)
        all_tickers.update(holdings)

    start_dl = DATA_INIZIO - relativedelta(months=LOOKBACK_MESI + 1)
    end_dl   = DATA_OGGI + timedelta(days=5)

    print(f"Download {len(all_tickers)} ticker da {start_dl} a {end_dl}...")
    raw = yf.download(
        list(all_tickers),
        start=start_dl.strftime('%Y-%m-%d'),
        end=end_dl.strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw
    print(f"Dati scaricati: {data.shape[0]} righe x {data.shape[1]} colonne")
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────
def prezzo_alla_data(data, ticker, target_date):
    """Restituisce il prezzo di chiusura al giorno piu vicino disponibile."""
    if ticker not in data.columns:
        return None
    series = data[ticker].dropna()
    if series.empty:
        return None
    target = pd.Timestamp(target_date)
    idx = series.index.searchsorted(target)
    candidates = []
    for offset in range(-10, 11):
        i = idx + offset
        if 0 <= i < len(series):
            candidates.append((abs((series.index[i] - target).days), series.iloc[i]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return float(candidates[0][1])


def ritorno_pct(prezzo_inizio, prezzo_fine):
    if prezzo_inizio is None or prezzo_fine is None or prezzo_inizio == 0:
        return None
    return (prezzo_fine - prezzo_inizio) / prezzo_inizio * 100


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(data):
    risultati    = []
    capitale     = float(N_TITOLI * CAPITALE_PER_TITOLO)
    cap_iniziale = capitale

    cur = DATA_INIZIO
    while cur < DATA_FINE:
        d_acquisto     = cur
        d_vendita      = cur + relativedelta(months=1)
        d_lookback     = cur - relativedelta(months=LOOKBACK_MESI)

        # ── 1. Miglior ETF singolo (3 mesi di lookback) ──────────────────
        etf_ritorni = {}
        for etf in ETF_UNIVERSE:
            r = ritorno_pct(
                prezzo_alla_data(data, etf, d_lookback),
                prezzo_alla_data(data, etf, d_acquisto),
            )
            if r is not None:
                etf_ritorni[etf] = r
        if not etf_ritorni:
            cur = d_vendita
            continue
        miglior_etf     = max(etf_ritorni, key=etf_ritorni.get)
        miglior_etf_ret = etf_ritorni[miglior_etf]

        # ── 2. Top N titoli dalle holdings dell'ETF vincitore ─────────────
        stock_mom = {}
        for s in ETF_UNIVERSE[miglior_etf]:
            r = ritorno_pct(
                prezzo_alla_data(data, s, d_lookback),
                prezzo_alla_data(data, s, d_acquisto),
            )
            if r is not None:
                stock_mom[s] = r
        if len(stock_mom) < 2:
            cur = d_vendita
            continue
        top_stocks = sorted(stock_mom, key=stock_mom.get, reverse=True)[:N_TITOLI]

        # Compute returns for ALL stocks (for JS dynamic N selection)
        stocks_all = []
        for s in sorted(stock_mom, key=stock_mom.get, reverse=True):
            p_acq = prezzo_alla_data(data, s, d_acquisto)
            p_vnd = prezzo_alla_data(data, s, d_vendita)
            if p_acq and p_vnd and p_acq > 0:
                r_lordo = (p_vnd - p_acq) / p_acq * 100
                stocks_all.append({
                    'ticker':    s,
                    'mom_3m':    round(stock_mom[s], 2),
                    'ret_lordo': round(r_lordo, 2),
                    'p_acq':     round(p_acq, 4),
                    'p_vnd':     round(p_vnd, 4),
                })

        # ── 3. Rendimento del mese tenuto ─────────────────────────────────
        cap_per_titolo  = capitale / N_TITOLI   # portafoglio intero diviso N
        dettaglio       = []
        rendimenti_netti = []

        for s in top_stocks:
            p_acq = prezzo_alla_data(data, s, d_acquisto)
            p_vnd = prezzo_alla_data(data, s, d_vendita)
            if p_acq is None or p_vnd is None:
                continue
            r_lordo = ritorno_pct(p_acq, p_vnd)
            if r_lordo is None:
                continue

            r_netto      = ((1 + r_lordo / 100) * (1 - COSTO_INGRESSO_PCT / 100) - 1) * 100
            costo_eur    = cap_per_titolo * COSTO_INGRESSO_PCT / 100
            guadagno_eur = cap_per_titolo * r_netto / 100

            dettaglio.append({
                'ticker':            s,
                'prezzo_acquisto':   round(p_acq, 4),
                'prezzo_vendita':    round(p_vnd, 4),
                'ritorno_lordo_pct': round(r_lordo, 2),
                'ritorno_netto_pct': round(r_netto, 2),
                'costo_eur':         round(costo_eur, 2),
                'guadagno_eur':      round(guadagno_eur, 2),
                'momentum_3m_pct':   round(stock_mom.get(s, 0), 2),
            })
            rendimenti_netti.append(r_netto)

        if not rendimenti_netti:
            cur = d_vendita
            continue

        # Ordina per rendimento netto decrescente
        dettaglio.sort(key=lambda x: x['ritorno_netto_pct'], reverse=True)

        rit_portfolio  = sum(rendimenti_netti) / len(rendimenti_netti)
        capitale_fine  = capitale * (1 + rit_portfolio / 100)

        risultati.append({
            'data':             d_acquisto.strftime('%Y-%m-%d'),
            'etf':              miglior_etf,
            'etf_ritorno_3m':   round(miglior_etf_ret, 2),
            'top_stocks':       [d['ticker'] for d in dettaglio],
            'capitale_inizio':  round(capitale, 2),
            'ritorno_mese_pct': round(rit_portfolio, 2),
            'capitale_fine':    round(capitale_fine, 2),
            'dettaglio':        dettaglio,
            'tutti_etf':        {k: round(v, 2) for k, v in etf_ritorni.items()},
            'stocks_all':       stocks_all,
        })

        capitale = capitale_fine
        cur      = d_vendita

    # statistiche globali
    cap_iniziale   = CAPITALE_PER_TITOLO * N_TITOLI
    cap_finale     = risultati[-1]['capitale_fine'] if risultati else cap_iniziale
    guadagno       = cap_finale - cap_iniziale
    rendimento_tot = (cap_finale / cap_iniziale - 1) * 100 if cap_iniziale else 0

    rits          = [r['ritorno_mese_pct'] for r in risultati]
    n_mesi        = len(rits)
    mesi_positivi = sum(1 for r in rits if r > 0)
    win_rate      = mesi_positivi / n_mesi * 100 if n_mesi else 0
    rit_medio     = sum(rits) / n_mesi if n_mesi else 0
    rits_sorted   = sorted(rits)
    rit_mediano   = rits_sorted[n_mesi // 2] if n_mesi else 0
    best_month    = max(rits) if rits else 0
    worst_month   = min(rits) if rits else 0

    if n_mesi > 1:
        mean_r = rit_medio
        std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rits) / (n_mesi - 1))
        sharpe = (mean_r / std_r * math.sqrt(12)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    stats = {
        'cap_iniziale':        round(cap_iniziale, 2),
        'cap_finale':          round(cap_finale, 2),
        'guadagno':            round(guadagno, 2),
        'rendimento_tot':      round(rendimento_tot, 2),
        'n_mesi':              n_mesi,
        'mesi_positivi':       mesi_positivi,
        'win_rate':            round(win_rate, 2),
        'rit_medio':           round(rit_medio, 4),
        'rit_mediano':         round(rit_mediano, 4),
        'best_month':          round(best_month, 4),
        'worst_month':         round(worst_month, 4),
        'sharpe':              round(sharpe, 4),
        'n_titoli':            N_TITOLI,
        'capitale_per_titolo': CAPITALE_PER_TITOLO,
        'costo_pct':           COSTO_INGRESSO_PCT,
    }
    return risultati, stats


# ── Posizione aperta ──────────────────────────────────────────────────────────
def calcola_posizione_aperta(data, risultati):
    """Calcola la posizione del mese corrente (non ancora chiusa)."""
    if not risultati:
        return None

    d_acquisto = DATA_FINE
    d_lookback = d_acquisto - relativedelta(months=LOOKBACK_MESI)
    capitale_corrente = risultati[-1]['capitale_fine']

    # ── 1. Miglior ETF singolo ────────────────────────────────────────────
    etf_ritorni = {}
    for etf in ETF_UNIVERSE:
        r = ritorno_pct(
            prezzo_alla_data(data, etf, d_lookback),
            prezzo_alla_data(data, etf, d_acquisto),
        )
        if r is not None:
            etf_ritorni[etf] = r
    if not etf_ritorni:
        return None
    miglior_etf = max(etf_ritorni, key=etf_ritorni.get)

    # ── 2. Top N titoli dalle holdings dell'ETF ───────────────────────────
    stock_mom = {}
    for s in ETF_UNIVERSE[miglior_etf]:
        r = ritorno_pct(
            prezzo_alla_data(data, s, d_lookback),
            prezzo_alla_data(data, s, d_acquisto),
        )
        if r is not None:
            stock_mom[s] = r
    if len(stock_mom) < 2:
        return None
    top_stocks = sorted(stock_mom, key=stock_mom.get, reverse=True)[:N_TITOLI]

    # Compute returns for ALL stocks using today's price (for JS dynamic N selection)
    stocks_all_pa = []
    for s in sorted(stock_mom, key=stock_mom.get, reverse=True):
        p_acq  = prezzo_alla_data(data, s, d_acquisto)
        p_oggi = prezzo_alla_data(data, s, DATA_OGGI)
        if p_acq and p_oggi and p_acq > 0:
            r_lordo = (p_oggi - p_acq) / p_acq * 100
            stocks_all_pa.append({
                'ticker':      s,
                'mom_3m':      round(stock_mom[s], 2),
                'ret_lordo':   round(r_lordo, 2),
                'p_acq':       round(p_acq, 4),
                'p_oggi':      round(p_oggi, 4),
            })

    # ── 3. Valore attuale (prezzi di oggi) ────────────────────────────────
    cap_per_titolo   = capitale_corrente / N_TITOLI
    dettaglio        = []
    rendimenti_netti = []

    for s in top_stocks:
        p_acq  = prezzo_alla_data(data, s, d_acquisto)
        p_oggi = prezzo_alla_data(data, s, DATA_OGGI)
        if p_acq is None or p_oggi is None:
            continue
        r_lordo = ritorno_pct(p_acq, p_oggi)
        if r_lordo is None:
            continue

        r_netto      = ((1 + r_lordo / 100) * (1 - COSTO_INGRESSO_PCT / 100) - 1) * 100
        costo_eur    = cap_per_titolo * COSTO_INGRESSO_PCT / 100
        guadagno_eur = cap_per_titolo * r_netto / 100

        dettaglio.append({
            'ticker':            s,
            'prezzo_acquisto':   round(p_acq,  4),
            'prezzo_oggi':       round(p_oggi, 4),
            'ritorno_lordo_pct': round(r_lordo, 2),
            'ritorno_netto_pct': round(r_netto, 2),
            'costo_eur':         round(costo_eur, 2),
            'guadagno_eur':      round(guadagno_eur, 2),
            'momentum_3m_pct':   round(stock_mom.get(s, 0), 2),
        })
        rendimenti_netti.append(r_netto)

    if not rendimenti_netti:
        return None

    dettaglio.sort(key=lambda x: x['ritorno_netto_pct'], reverse=True)

    rit_attuale    = sum(rendimenti_netti) / len(rendimenti_netti)
    valore_attuale = capitale_corrente * (1 + rit_attuale / 100)

    return {
        'data_acquisto':     d_acquisto.strftime('%Y-%m-%d'),
        'data_oggi':         DATA_OGGI.strftime('%Y-%m-%d'),
        'etf':               miglior_etf,
        'top_stocks':        [d['ticker'] for d in dettaglio],
        'capitale_investito':round(capitale_corrente, 2),
        'ritorno_parziale':  round(rit_attuale, 2),
        'valore_attuale':    round(valore_attuale, 2),
        'dettaglio':         dettaglio,
        'tutti_etf':         {k: round(v, 2) for k, v in etf_ritorni.items()},
        'stocks_all':        stocks_all_pa,
    }


# ── Drawdown ──────────────────────────────────────────────────────────────────
def calcola_drawdown(equity_vals):
    """Calcola drawdown percentuale punto per punto e massimo drawdown."""
    drawdown_list = []
    peak = equity_vals[0] if equity_vals else 1.0
    for v in equity_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak != 0 else 0.0
        drawdown_list.append(round(dd, 4))
    max_dd = min(drawdown_list) if drawdown_list else 0.0
    return drawdown_list, round(max_dd, 4)


# ── S&P 500 normalizzato ──────────────────────────────────────────────────────
def get_sp500_normalizzato(data, cap_labs, cap_iniziale):
    spy_start = prezzo_alla_data(data, 'SPY', DATA_INIZIO)
    if spy_start is None or spy_start == 0:
        return [None] * len(cap_labs)

    result = []
    for label in cap_labs:
        try:
            if len(label) == 7:  # YYYY-MM
                year, month = int(label[:4]), int(label[5:7])
                d = date(year, month, 1)
            else:  # YYYY-MM-DD
                parts = label.split('-')
                d = date(int(parts[0]), int(parts[1]), int(parts[2]))
            spy_price = prezzo_alla_data(data, 'SPY', d)
            if spy_price is not None:
                result.append(round(cap_iniziale * spy_price / spy_start, 4))
            else:
                result.append(None)
        except Exception:
            result.append(None)
    return result


# ── Ticker charts ─────────────────────────────────────────────────────────────
def build_ticker_charts(data, risultati, posizione_aperta):
    charts = {}

    def extract_series(ticker, d_from, d_to):
        if ticker not in data.columns:
            return [], []
        series = data[ticker].dropna()
        mask   = (series.index >= pd.Timestamp(d_from)) & (series.index <= pd.Timestamp(d_to))
        sub    = series[mask]
        dates  = [d.strftime('%Y-%m-%d') for d in sub.index]
        prices = [round(float(v), 4) for v in sub.values]
        return dates, prices

    for mese in risultati:
        d_acquisto = date.fromisoformat(mese['data'])
        d_vendita  = date.fromisoformat(mese['data']) + relativedelta(months=1)
        d_from     = d_acquisto - timedelta(days=10)
        d_to       = d_acquisto + relativedelta(months=1) + timedelta(days=10)

        for stock in mese['dettaglio']:
            ticker = stock['ticker']
            key    = f"{ticker}-{mese['data'][:7]}"
            dates, prices = extract_series(ticker, d_from, d_to)
            if not dates:
                continue
            charts[key] = {
                'ticker':            ticker,
                'month':             mese['data'][:7],
                'dates':             dates,
                'prices':            prices,
                'entry_date':        d_acquisto.strftime('%Y-%m-%d'),
                'entry_price':       stock['prezzo_acquisto'],
                'exit_date':         d_vendita.strftime('%Y-%m-%d'),
                'exit_price':        stock['prezzo_vendita'],
                'ritorno_netto_pct': stock['ritorno_netto_pct'],
            }

    if posizione_aperta:
        d_acquisto = date.fromisoformat(posizione_aperta['data_acquisto'])
        d_from     = d_acquisto - timedelta(days=10)
        d_to       = DATA_OGGI + timedelta(days=10)

        for stock in posizione_aperta['dettaglio']:
            ticker = stock['ticker']
            key    = f"{ticker}-{posizione_aperta['data_acquisto'][:7]}"
            dates, prices = extract_series(ticker, d_from, d_to)
            if not dates:
                continue
            charts[key] = {
                'ticker':            ticker,
                'month':             posizione_aperta['data_acquisto'][:7],
                'dates':             dates,
                'prices':            prices,
                'entry_date':        posizione_aperta['data_acquisto'],
                'entry_price':       stock['prezzo_acquisto'],
                'exit_date':         DATA_OGGI.isoformat(),
                'exit_price':        stock['prezzo_oggi'],
                'ritorno_netto_pct': stock['ritorno_netto_pct'],
            }

    return charts


# ── Performance matrix ────────────────────────────────────────────────────────
def build_perf_matrix(risultati):
    matrix = {}
    for mese in risultati:
        d = date.fromisoformat(mese['data'])
        y, m = d.year, d.month
        if y not in matrix:
            matrix[y] = {}
        matrix[y][m] = mese['ritorno_mese_pct']

    annual_returns = {}
    for y, months_dict in matrix.items():
        prod = 1.0
        for v in months_dict.values():
            prod *= (1 + v / 100)
        annual_returns[y] = round((prod - 1) * 100, 2)

    return matrix, annual_returns


# ── HTML Generator ────────────────────────────────────────────────────────────
def _cell_color(v):
    """Genera inline style per colore cella performance."""
    if v is None:
        return 'background:transparent; color:rgba(139,148,158,0.3);'
    if abs(v) < 0.05:
        return 'background:rgba(48,54,61,0.5); color:var(--muted);'
    intensity = min(abs(v) / 15, 1.0)
    sat  = int(40 + intensity * 55)
    lig  = int(45 - intensity * 15)
    hue  = 142 if v > 0 else 0
    return f'background:hsla({hue},{sat}%,{lig}%,0.85); color:#fff;'


def _build_perf_matrix_html(perf_matrix, annual_returns):
    MESI_NOMI = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
                 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    years = sorted(perf_matrix.keys())

    header = '<tr><th>Anno</th>'
    for mn in MESI_NOMI:
        header += f'<th>{mn}</th>'
    header += '<th>Anno</th></tr>'

    rows = ''
    for y in years:
        rows += f'<tr><td>{y}</td>'
        for m in range(1, 13):
            v = perf_matrix[y].get(m)
            if v is None:
                rows += '<td class="pm-empty">&#x2014;</td>'
            else:
                style = _cell_color(v)
                sign  = '+' if v >= 0 else ''
                rows += f'<td style="{style}">{sign}{v:.1f}%</td>'
        ar = annual_returns.get(y)
        if ar is None:
            rows += '<td class="pm-empty pm-annual">&#x2014;</td>'
        else:
            style = _cell_color(ar)
            sign  = '+' if ar >= 0 else ''
            rows += f'<td class="pm-annual" style="{style}">{sign}{ar:.1f}%</td>'
        rows += '</tr>'

    return (
        '<div class="matrix-wrap">'
        '<h2>Performance mensile</h2>'
        '<table class="perf-matrix">'
        f'<thead>{header}</thead>'
        f'<tbody>{rows}</tbody>'
        '</table></div>'
    )


def genera_html(risultati, stats, cap_labs, cap_vals, sp500_vals, drawdown_vals, max_dd,
                ticker_charts, perf_matrix, annual_returns, posizione_aperta=None,
                mesi_raw_js=None, sp500_norm_js=None, pos_raw_js=None):

    def fmt_pct(v, decimals=2):
        if v is None:
            return '&#x2014;'
        sign = '+' if v >= 0 else ''
        return f"{sign}{v:.{decimals}f}%"

    def fmt_eur(v):
        if v is None:
            return '&#x2014;'
        sign = '+' if v >= 0 else ''
        return f"{sign}{v:,.2f} &#x20AC;"

    def color_cls(v):
        if v is None or v == 0:
            return 'neutral'
        return 'pos' if v > 0 else 'neg'

    # ── table rows ─────────────────────────────────────────────────────────────
    table_rows_html = ''
    for mese in risultati:
        d_fmt = mese['data'][:7]
        r     = mese['ritorno_mese_pct']
        c_cls = color_cls(r)

        badges = ''
        for d in mese['dettaglio']:
            key = f"{d['ticker']}-{d_fmt}"
            dc  = color_cls(d['ritorno_netto_pct'])
            badges += (
                f'<span class="tbadge {dc}" onclick="openTickerChart(\'{key}\')">'
                f'{d["ticker"]}</span> '
            )

        det_rows = ''
        for d in mese['dettaglio']:
            dc = color_cls(d['ritorno_netto_pct'])
            det_rows += (
                f'<tr>'
                f'<td>{d["ticker"]}</td>'
                f'<td>{d["prezzo_acquisto"]:.4f}</td>'
                f'<td>{d["prezzo_vendita"]:.4f}</td>'
                f'<td class="{color_cls(d["ritorno_lordo_pct"])}">{fmt_pct(d["ritorno_lordo_pct"])}</td>'
                f'<td>{d["costo_eur"]:.2f} &#x20AC;</td>'
                f'<td class="{dc}">{fmt_pct(d["ritorno_netto_pct"])}</td>'
                f'<td>{fmt_pct(d["momentum_3m_pct"])}</td>'
                f'<td class="{dc}">{fmt_eur(d["guadagno_eur"])}</td>'
                f'</tr>'
            )

        table_rows_html += (
            f'<tr class="main-row" onclick="toggleDet(this)">'
            f'<td>{d_fmt}</td>'
            f'<td>{badges}</td>'
            f'<td class="{c_cls}">{fmt_pct(r)}</td>'
            f'<td>{mese["capitale_fine"]:,.2f} &#x20AC;</td>'
            f'</tr>'
            f'<tr class="det-row" style="display:none">'
            f'<td colspan="4">'
            f'<table class="det-table">'
            f'<thead><tr>'
            f'<th>Ticker</th><th>P.Acquisto</th><th>P.Vendita</th>'
            f'<th>Rit.lordo</th><th>Costo EUR</th><th>Rit.netto</th>'
            f'<th>Momentum 3m</th><th>P&amp;L EUR</th>'
            f'</tr></thead>'
            f'<tbody>{det_rows}</tbody>'
            f'</table></td></tr>'
        )

    # ── posizione aperta ────────────────────────────────────────────────────────
    pos_html = ''
    if posizione_aperta:
        pos_det_rows = ''
        for d in posizione_aperta['dettaglio']:
            dc = color_cls(d['ritorno_netto_pct'])
            pos_det_rows += (
                f'<tr>'
                f'<td>{d["ticker"]}</td>'
                f'<td>{d["prezzo_acquisto"]:.4f}</td>'
                f'<td>{d["prezzo_oggi"]:.4f}</td>'
                f'<td class="{color_cls(d["ritorno_lordo_pct"])}">{fmt_pct(d["ritorno_lordo_pct"])}</td>'
                f'<td>{d["costo_eur"]:.2f} &#x20AC;</td>'
                f'<td class="{dc}">{fmt_pct(d["ritorno_netto_pct"])}</td>'
                f'<td>{fmt_pct(d["momentum_3m_pct"])}</td>'
                f'<td class="{dc}">{fmt_eur(d["guadagno_eur"])}</td>'
                f'</tr>'
            )

        g_cls = color_cls(posizione_aperta['valore_attuale'] - posizione_aperta['capitale_investito'])
        r_cls = color_cls(posizione_aperta['ritorno_parziale'])
        pos_html = (
            '<div class="pos-aperta">'
            '<div class="pos-header">'
            f'<span class="pos-title">Posizione aperta &#x2014; {posizione_aperta["data_acquisto"][:7]}</span>'
            f'<span class="pos-meta">Aggiornato al {posizione_aperta["data_oggi"]}</span>'
            '</div>'
            '<div class="pos-stats">'
            f'<div class="pos-stat"><label>Rendimento medio</label>'
            f'<span class="{r_cls}">{fmt_pct(posizione_aperta["ritorno_parziale"])}</span></div>'
            f'<div class="pos-stat"><label>Guadagno</label>'
            f'<span class="{g_cls}">{fmt_eur(posizione_aperta["valore_attuale"] - posizione_aperta["capitale_investito"])}</span></div>'
            f'<div class="pos-stat"><label>Valore attuale</label>'
            f'<span>{posizione_aperta["valore_attuale"]:,.2f} &#x20AC;</span></div>'
            '</div>'
            '<table class="det-table" style="margin-top:14px">'
            '<thead><tr>'
            '<th>Ticker</th><th>P.Acquisto</th><th>P.Oggi</th>'
            '<th>Rit.lordo</th><th>Costo EUR</th><th>Rit.netto</th>'
            '<th>Momentum 3m</th><th>P&amp;L EUR</th>'
            '</tr></thead>'
            f'<tbody>{pos_det_rows}</tbody>'
            '</table></div>'
        )

    # ── nota metodologica ────────────────────────────────────────────────────────
    nota_costo = ''
    if COSTO_INGRESSO_PCT > 0:
        nota_costo = (
            f' Ad ogni ingresso viene applicato un costo dello '
            f'<strong>{COSTO_INGRESSO_PCT}%</strong> sul capitale investito per titolo.'
        )
    nota_html = (
        '<div class="nota">'
        f'<strong>Nota metodologica:</strong> A ogni inizio mese vengono selezionati i top-{N_TITOLI} ETF '
        f'per momentum a 3 mesi (ritorno dal prezzo di {LOOKBACK_MESI} mesi prima al primo del mese). '
        f'Capitale per ETF: {CAPITALE_PER_TITOLO:,.0f} &#x20AC;.{nota_costo} '
        'I rendimenti sono calcolati su prezzi di chiusura aggiustati (split &amp; dividendi).'
        '</div>'
    )

    # ── performance matrix ───────────────────────────────────────────────────────
    perf_matrix_html = _build_perf_matrix_html(perf_matrix, annual_returns)

    # ── JS data ────────────────────────────────────────────────────────────────
    cap_labs_js   = json.dumps(cap_labs)
    cap_vals_js   = json.dumps(cap_vals)
    sp500_js      = json.dumps(sp500_vals)
    dd_vals_js    = json.dumps(drawdown_vals)
    cap_iniz_js   = json.dumps(stats['cap_iniziale'])
    tc_js         = json.dumps(ticker_charts)
    costo_js      = json.dumps(COSTO_INGRESSO_PCT)
    mesi_raw_js_s = json.dumps(mesi_raw_js if mesi_raw_js is not None else [])
    sp500_norm_js_s = json.dumps(sp500_norm_js)
    pos_raw_js_s  = json.dumps(pos_raw_js)
    data_fine_lab = cap_labs[-1] if cap_labs else ''

    # ── stat card colors ───────────────────────────────────────────────────────
    g_cls  = 'pos' if stats['guadagno'] >= 0 else 'neg'
    rt_cls = 'pos' if stats['rendimento_tot'] >= 0 else 'neg'

    # ── assemble HTML ─────────────────────────────────────────────────────────
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="it">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Backtest ETF Momentum 2015-2026</title>\n'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>\n'
        '<style>\n'
        ':root {\n'
        '  --bg:     #0d1117;\n'
        '  --card:   #161b22;\n'
        '  --card2:  #1c2330;\n'
        '  --border: #30363d;\n'
        '  --text:   #e6edf3;\n'
        '  --muted:  #8b949e;\n'
        '  --blue:   #58a6ff;\n'
        '  --green:  #3fb950;\n'
        '  --red:    #f85149;\n'
        '  --yellow: #d29922;\n'
        '  --purple: #bc8cff;\n'
        '}\n'
        '*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n'
        'body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;font-size:0.88rem;line-height:1.5}\n'
        'header{background:var(--card);border-bottom:1px solid var(--border);padding:22px 40px 18px}\n'
        '.h-title{font-size:1.4rem;font-weight:700;color:var(--blue)}\n'
        '.h-sub{color:var(--muted);font-size:0.82rem;margin-top:4px}\n'
        '.h-meta{color:var(--muted);font-size:0.75rem;margin-top:6px}\n'
        '.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;padding:24px 40px;}\n'
        '.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px}\n'
        '.stat-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}\n'
        '.stat-val{font-size:1.2rem;font-weight:700}\n'
        '.pos{color:var(--green)}\n'
        '.neg{color:var(--red)}\n'
        '.neutral{color:var(--muted)}\n'
        '.blue{color:var(--blue)}\n'
        '.yellow{color:var(--yellow)}\n'
        '.pos-aperta{margin:0 40px 24px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}\n'
        '.pos-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}\n'
        '.pos-title{font-weight:700;color:var(--blue)}\n'
        '.pos-meta{font-size:0.75rem;color:var(--muted)}\n'
        '.pos-stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:4px}\n'
        '.pos-stat label{font-size:0.72rem;color:var(--muted);display:block}\n'
        '.pos-stat span{font-weight:700;font-size:0.95rem}\n'
        '.nota{margin:0 40px 20px;padding:12px 16px;background:var(--card2);border:1px solid var(--border);border-radius:10px;font-size:0.78rem;color:var(--muted)}\n'
        '.nota strong{color:var(--text)}\n'
        '.chart-wrap{margin:0 40px 0;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}\n'
        '.chart-wrap-top{border-radius:12px 12px 0 0 !important}\n'
        '.chart-title{font-size:0.82rem;color:var(--muted);margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.4px}\n'
        '.zoom-btns{display:flex;gap:6px;margin-bottom:14px}\n'
        '.zoom-btn{background:var(--card2);border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:0.75rem}\n'
        '.zoom-btn:hover,.zoom-btn.active{background:var(--blue);color:#fff;border-color:var(--blue)}\n'
        '.dd-chart-wrap{margin:-18px 40px 24px;background:var(--card);border:1px solid var(--border);border-top:none;border-radius:0 0 12px 12px;padding:0 20px 16px}\n'
        '.dd-chart-wrap h3{font-size:0.78rem;color:var(--muted);padding-top:10px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px}\n'
        '.matrix-wrap{margin:0 40px 28px;overflow-x:auto}\n'
        '.matrix-wrap h2{font-size:0.95rem;color:var(--blue);margin-bottom:12px}\n'
        '.perf-matrix{border-collapse:collapse;font-size:0.78rem;min-width:100%}\n'
        '.perf-matrix th{padding:6px 10px;color:var(--muted);text-align:center;font-size:0.72rem;text-transform:uppercase;white-space:nowrap;border-bottom:2px solid var(--border)}\n'
        '.perf-matrix th:first-child{text-align:left}\n'
        '.perf-matrix td{padding:5px 8px;text-align:center;border:1px solid rgba(48,54,61,0.4);min-width:54px;font-weight:600}\n'
        '.perf-matrix td:first-child{text-align:left;color:var(--muted);font-weight:400}\n'
        '.pm-empty{color:rgba(139,148,158,0.3) !important;background:transparent !important}\n'
        '.pm-annual{border-left:2px solid var(--border) !important;font-weight:700 !important}\n'
        '.table-wrap{margin:0 40px 40px;overflow-x:auto}\n'
        '.main-table{width:100%;border-collapse:collapse;font-size:0.82rem}\n'
        '.main-table th{padding:10px 12px;text-align:left;color:var(--muted);font-size:0.72rem;text-transform:uppercase;border-bottom:2px solid var(--border);white-space:nowrap}\n'
        '.main-table td{padding:9px 12px;border-bottom:1px solid rgba(48,54,61,0.5)}\n'
        '.main-row{cursor:pointer}\n'
        '.main-row:hover{background:rgba(88,166,255,0.04)}\n'
        '.det-row td{padding:0;background:var(--card2)}\n'
        '.det-table{width:100%;border-collapse:collapse;font-size:0.78rem}\n'
        '.det-table th{padding:7px 10px;color:var(--muted);font-size:0.7rem;text-transform:uppercase;border-bottom:1px solid var(--border);background:var(--card)}\n'
        '.det-table td{padding:6px 10px;border-bottom:1px solid rgba(48,54,61,0.4)}\n'
        '.tbadge{display:inline-block;padding:2px 8px;border-radius:5px;font-size:0.72rem;font-weight:600;margin:1px;cursor:pointer !important}\n'
        '.tbadge:hover{background:rgba(88,166,255,0.12) !important}\n'
        '.tbadge.pos{background:rgba(63,185,80,0.15);color:var(--green)}\n'
        '.tbadge.neg{background:rgba(248,81,73,0.15);color:var(--red)}\n'
        '.tbadge.neutral{background:rgba(139,148,158,0.12);color:var(--muted)}\n'
        '#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:1000;align-items:center;justify-content:center}\n'
        '.modal-box{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;width:min(820px,94vw);max-height:90vh;overflow-y:auto}\n'
        '.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}\n'
        '.modal-title{font-size:1.1rem;font-weight:700;color:var(--blue)}\n'
        '.modal-close{background:none;border:none;color:var(--muted);font-size:1.4rem;cursor:pointer;padding:0 6px}\n'
        '.modal-close:hover{color:var(--text)}\n'
        '.modal-stats{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:0.82rem}\n'
        '.modal-stat{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:8px 14px}\n'
        '.modal-stat span{font-weight:700}\n'
        'canvas{display:block;width:100% !important}\n'
        '/* Controls panel */\n'
        '.controls-panel{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-end;padding:16px 40px;background:var(--card2);border-bottom:1px solid var(--border);}\n'
        '.ctrl-group{display:flex;flex-direction:column;gap:6px;min-width:160px;}\n'
        '.ctrl-group label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.4px;}\n'
        '.ctrl-row{display:flex;align-items:center;gap:8px;}\n'
        '.ctrl-row input[type=range]{width:110px;accent-color:var(--blue);}\n'
        '.ctrl-num{font-size:0.88rem;font-weight:700;color:var(--blue);min-width:28px;}\n'
        '.ctrl-input{background:var(--card);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;padding:5px 10px;width:120px;}\n'
        '.ctrl-input:focus{outline:none;border-color:var(--blue);}\n'
        '.ctrl-note{font-size:0.7rem;color:var(--muted);margin-top:2px;}\n'
        '.inner-table{width:100%;border-collapse:collapse;font-size:0.78rem;}\n'
        '.inner-table th{padding:7px 10px;color:var(--muted);font-size:0.7rem;text-transform:uppercase;border-bottom:1px solid var(--border);background:var(--card);}\n'
        '.inner-table td{padding:6px 10px;border-bottom:1px solid rgba(48,54,61,0.4);}\n'
        '.etf-pill{display:inline-block;padding:2px 8px;border-radius:5px;font-size:0.75rem;font-weight:600;background:rgba(88,166,255,0.15);color:var(--blue);}\n'
        '.tr-main{cursor:pointer;}\n'
        '.tr-main:hover{background:rgba(88,166,255,0.04);}\n'
        'header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px}\n'
        '.h-actions{display:flex;flex-direction:column;align-items:flex-end;gap:8px;flex-shrink:0}\n'
        '.h-btns{display:flex;gap:10px}\n'
        '.act-btn{border:none;border-radius:8px;padding:9px 18px;font-size:0.84rem;font-weight:600;cursor:pointer;transition:opacity 0.15s;white-space:nowrap}\n'
        '.act-btn:hover{opacity:0.82}\n'
        '.act-btn:disabled{opacity:0.45;cursor:not-allowed}\n'
        '.tg-btn{background:#229ED9;color:#fff}\n'
        '.upd-btn{background:#3fb950;color:#0d1117}\n'
        '.act-status{font-size:0.72rem;color:var(--muted);min-height:1em;text-align:right}\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '\n'
        '<header>\n'
        '  <div class="h-title">Backtest ETF Momentum 2015-2026</div>\n'
        f'  <div class="h-sub">Top-{N_TITOLI} ETF per momentum 3m &nbsp;|&nbsp; '
        f'Capitale: {CAPITALE_PER_TITOLO:,.0f} &#x20AC; per titolo &nbsp;|&nbsp; '
        f'Costo ingresso: {COSTO_INGRESSO_PCT}%</div>\n'
        f'  <div class="h-meta">Aggiornato al {DATA_OGGI.strftime("%d/%m/%Y")}</div>\n'
        '</header>\n'
        '\n'
        '<div class="controls-panel">\n'
        '  <div class="ctrl-group">\n'
        '    <label>Titoli per mese</label>\n'
        '    <div class="ctrl-row">\n'
        f'      <input type="range" id="ctrl-n" min="1" max="4" value="{N_TITOLI}" step="1" oninput="onCtrl()">\n'
        f'      <span class="ctrl-num" id="ctrl-n-lbl">{N_TITOLI}</span>\n'
        '    </div>\n'
        '    <div class="ctrl-note">1 = solo il migliore</div>\n'
        '  </div>\n'
        '  <div class="ctrl-group">\n'
        '    <label>Capitale per titolo (&euro;)</label>\n'
        f'    <input type="number" class="ctrl-input" id="ctrl-cap" value="{CAPITALE_PER_TITOLO:.0f}" min="100" step="1000" oninput="onCtrl()">\n'
        '  </div>\n'
        '  <div class="ctrl-group">\n'
        '    <label>Commissioni ingresso (%)</label>\n'
        f'    <input type="number" class="ctrl-input" id="ctrl-costo" value="{COSTO_INGRESSO_PCT}" min="0" max="5" step="0.05" oninput="onCtrl()">\n'
        '  </div>\n'
        '  <div class="ctrl-group">\n'
        '    <label>Data inizio backtest</label>\n'
        f'    <input type="month" class="ctrl-input" id="ctrl-date-start" oninput="onCtrl()">\n'
        '  </div>\n'
        '  <div class="ctrl-group">\n'
        '    <label>Data fine backtest</label>\n'
        f'    <input type="month" class="ctrl-input" id="ctrl-date-end" oninput="onCtrl()">\n'
        '  </div>\n'
        '</div>\n'
        '\n'
        '<div class="stats-grid">\n'
        f'  <div class="stat-card"><div class="stat-label">Capitale iniziale</div>'
        f'<div class="stat-val blue" id="stat-cap-iniz">{stats["cap_iniziale"]:,.0f} &#x20AC;</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Valore oggi</div>'
        f'<div class="stat-val pos" id="stat-valore">{stats["cap_finale"]:,.0f} &#x20AC;</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Guadagno netto</div>'
        f'<div class="stat-val {g_cls}" id="stat-guadagno">{fmt_eur(stats["guadagno"])}</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Rendimento totale</div>'
        f'<div class="stat-val {rt_cls}" id="stat-rend-tot">{fmt_pct(stats["rendimento_tot"])}</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Win rate</div>'
        f'<div class="stat-val pos" id="stat-winrate">{stats["win_rate"]:.1f}%</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Rit. medio/mese</div>'
        f'<div class="stat-val yellow" id="stat-rit-medio">{fmt_pct(stats["rit_medio"])}</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Miglior mese</div>'
        f'<div class="stat-val pos" id="stat-best">{fmt_pct(stats["best_month"])}</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Peggior mese</div>'
        f'<div class="stat-val neg" id="stat-worst">{fmt_pct(stats["worst_month"])}</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Max Drawdown</div>'
        f'<div class="stat-val neg" id="stat-maxdd">{max_dd:.2f}%</div></div>\n'
        f'  <div class="stat-card"><div class="stat-label">Sharpe</div>'
        f'<div class="stat-val blue" id="stat-sharpe">{stats["sharpe"]:.2f}</div></div>\n'
        '</div>\n'
        '\n'
        + '<div id="pos-aperta-container">' + pos_html + '</div>\n'
        + nota_html + '\n'
        '\n'
        '<div class="chart-wrap chart-wrap-top" style="margin-bottom:0">\n'
        '  <div class="chart-title">Curva del capitale</div>\n'
        '  <div class="zoom-btns">\n'
        '    <button class="zoom-btn" onclick="setRange(6,this)">6M</button>\n'
        '    <button class="zoom-btn" onclick="setRange(12,this)">1A</button>\n'
        '    <button class="zoom-btn" onclick="setRange(36,this)">3A</button>\n'
        '    <button class="zoom-btn" onclick="setRange(60,this)">5A</button>\n'
        '    <button class="zoom-btn active" onclick="setRange(0,this)">Tutto</button>\n'
        '  </div>\n'
        '  <div style="position:relative;height:420px">\n'
        '    <canvas id="mainChart"></canvas>\n'
        '  </div>\n'
        '</div>\n'
        '\n'
        '<div class="dd-chart-wrap">\n'
        '  <h3>Drawdown %</h3>\n'
        '  <div style="position:relative;height:120px">\n'
        '    <canvas id="ddChart"></canvas>\n'
        '  </div>\n'
        '</div>\n'
        '\n'
        + '<div id="matrix-container">' + perf_matrix_html + '</div>\n'
        '\n'
        '<div class="table-wrap">\n'
        '  <table class="main-table">\n'
        '    <thead>\n'
        '      <tr>\n'
        '        <th>Mese</th>\n'
        '        <th>ETF</th>\n'
        '        <th>Momentum ETF</th>\n'
        '        <th>Titoli selezionati</th>\n'
        '        <th>Rendimento</th>\n'
        '        <th>Capitale fine mese</th>\n'
        '      </tr>\n'
        '    </thead>\n'
        '    <tbody id="tbody-main">\n'
        + table_rows_html
        + '    </tbody>\n'
        '  </table>\n'
        '</div>\n'
        '\n'
        '<div id="modal">\n'
        '  <div class="modal-box">\n'
        '    <div class="modal-header">\n'
        '      <span class="modal-title" id="modal-title">Grafico</span>\n'
        '      <button class="modal-close" onclick="closeModal()">&#x2715;</button>\n'
        '    </div>\n'
        '    <canvas id="modalChart" height="120"></canvas>\n'
        '    <div class="modal-stats" id="modal-stats"></div>\n'
        '  </div>\n'
        '</div>\n'
        '\n'
        '<script>\n'
        f'const ALL_LABELS    = {cap_labs_js};\n'
        f'const ALL_VALS      = {cap_vals_js};\n'
        f'const SP500_VALS    = {sp500_js};\n'
        f'const DD_VALS       = {dd_vals_js};\n'
        f'const CAP_INIZ      = {cap_iniz_js};\n'
        f'const TICKER_CHARTS = {tc_js};\n'
        f'const COSTO_PCT     = {costo_js};\n'
        f'const MESI_RAW      = {mesi_raw_js_s};\n'
        f'const SP500_NORM    = {sp500_norm_js_s};\n'
        f'const POS_RAW       = {pos_raw_js_s};\n'
        f'const PARAMS_DEFAULT = {{n: {N_TITOLI}, cap: {CAPITALE_PER_TITOLO:.0f}, costo: {COSTO_INGRESSO_PCT}}};\n'
        f'const DATA_FINE_LAB = {json.dumps(data_fine_lab)};\n'
        '\n'
        '// Main chart\n'
        'const mainCtx = document.getElementById("mainChart").getContext("2d");\n'
        'const mainChart = new Chart(mainCtx, {\n'
        '  type: "line",\n'
        '  data: {\n'
        '    labels: ALL_LABELS,\n'
        '    datasets: [\n'
        '      {\n'
        '        label: "Portafoglio",\n'
        '        data: ALL_VALS,\n'
        '        borderColor: "#58a6ff",\n'
        '        backgroundColor: "rgba(88,166,255,0.08)",\n'
        '        borderWidth: 2,\n'
        '        fill: true,\n'
        '        tension: 0.3,\n'
        '        pointRadius: 0,\n'
        '        pointHoverRadius: 5,\n'
        '      },\n'
        '      {\n'
        '        label: "Baseline",\n'
        '        data: ALL_LABELS.map(() => CAP_INIZ),\n'
        '        borderColor: "rgba(139,148,158,0.4)",\n'
        '        borderWidth: 1,\n'
        '        borderDash: [6,4],\n'
        '        fill: false,\n'
        '        tension: 0,\n'
        '        pointRadius: 0,\n'
        '      },\n'
        '      {\n'
        '        label: "S&P 500",\n'
        '        data: SP500_VALS,\n'
        '        borderColor: "#d29922",\n'
        '        backgroundColor: "transparent",\n'
        '        borderWidth: 2,\n'
        '        fill: false,\n'
        '        tension: 0.3,\n'
        '        pointRadius: 0,\n'
        '        pointHoverRadius: 5,\n'
        '      },\n'
        '    ]\n'
        '  },\n'
        '  options: {\n'
        '    responsive: true,\n'
        '    maintainAspectRatio: false,\n'
        '    interaction: { mode: "index", intersect: false },\n'
        '    plugins: {\n'
        '      legend: { labels: { color: "#8b949e", font: { size: 11 } } },\n'
        '      tooltip: {\n'
        '        backgroundColor: "#161b22",\n'
        '        borderColor: "#30363d",\n'
        '        borderWidth: 1,\n'
        '        titleColor: "#e6edf3",\n'
        '        bodyColor: "#8b949e",\n'
        '        callbacks: {\n'
        '          label(ctx) {\n'
        '            const v = ctx.parsed.y;\n'
        '            if (v === null || v === undefined) return null;\n'
        '            return ctx.dataset.label + ": " + v.toFixed(2) + " \u20AC";\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '    },\n'
        '    scales: {\n'
        '      x: { ticks: { color: "#8b949e", maxTicksLimit: 14, font: { size: 10 } }, grid: { color: "rgba(48,54,61,0.5)" } },\n'
        '      y: { ticks: { color: "#8b949e", font: { size: 10 }, callback: v => v.toLocaleString("it") + " \u20AC" }, grid: { color: "rgba(48,54,61,0.5)" } }\n'
        '    }\n'
        '  }\n'
        '});\n'
        '\n'
        '// Drawdown chart\n'
        'const ddCtx = document.getElementById("ddChart").getContext("2d");\n'
        'const ddChart = new Chart(ddCtx, {\n'
        '  type: "line",\n'
        '  data: {\n'
        '    labels: ALL_LABELS,\n'
        '    datasets: [{\n'
        '      label: "Drawdown",\n'
        '      data: DD_VALS,\n'
        '      borderColor: "#ff4d6d",\n'
        '      backgroundColor: "rgba(255,77,109,0.25)",\n'
        '      borderWidth: 1.5,\n'
        '      fill: true,\n'
        '      tension: 0.2,\n'
        '      pointRadius: 0,\n'
        '    }]\n'
        '  },\n'
        '  options: {\n'
        '    responsive: true,\n'
        '    maintainAspectRatio: false,\n'
        '    plugins: {\n'
        '      legend: { display: false },\n'
        '      tooltip: {\n'
        '        backgroundColor: "#161b22",\n'
        '        borderColor: "#30363d",\n'
        '        borderWidth: 1,\n'
        '        titleColor: "#e6edf3",\n'
        '        bodyColor: "#8b949e",\n'
        '        callbacks: {\n'
        '          label(ctx) { return "Drawdown: " + ctx.parsed.y.toFixed(2) + "%"; }\n'
        '        }\n'
        '      }\n'
        '    },\n'
        '    scales: {\n'
        '      x: { ticks: { color: "#8b949e", maxTicksLimit: 14, font: { size: 10 } }, grid: { color: "rgba(48,54,61,0.5)" } },\n'
        '      y: {\n'
        '        max: 0,\n'
        '        ticks: { color: "#8b949e", font: { size: 10 }, callback: v => v.toFixed(1) + "%" },\n'
        '        grid: { color: "rgba(48,54,61,0.5)" }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '});\n'
        '\n'
        '// Zoom\n'
        'function setRange(months, btn) {\n'
        '  currentRange = months;\n'
        '  if (btn) {\n'
        '    document.querySelectorAll(".zoom-btn").forEach(b => b.classList.remove("active"));\n'
        '    btn.classList.add("active");\n'
        '  }\n'
        '  let labels, vals, sp5, ddv;\n'
        '  if (months === 0) {\n'
        '    labels = ALL_LABELS_CUR; vals = ALL_VALS_CUR; sp5 = SP500_VALS_CUR; ddv = DD_VALS_CUR;\n'
        '  } else {\n'
        '    const n = Math.min(months + 1, ALL_LABELS_CUR.length);\n'
        '    labels = ALL_LABELS_CUR.slice(-n); vals = ALL_VALS_CUR.slice(-n);\n'
        '    sp5 = SP500_VALS_CUR ? SP500_VALS_CUR.slice(-n) : null; ddv = DD_VALS_CUR.slice(-n);\n'
        '  }\n'
        '  mainChart.data.labels = labels;\n'
        '  mainChart.data.datasets[0].data = vals;\n'
        '  mainChart.data.datasets[1].data = labels.map(() => CAP_INIZ);\n'
        '  mainChart.data.datasets[2].data = sp5;\n'
        '  mainChart.update();\n'
        '  ddChart.data.labels = labels;\n'
        '  ddChart.data.datasets[0].data = ddv;\n'
        '  ddChart.update();\n'
        '}\n'
        '\n'
        '// Toggle detail (legacy rows)\n'
        'function toggleDet(tr) {\n'
        '  const next = tr.nextElementSibling;\n'
        '  if (!next || !next.classList.contains("det-row")) return;\n'
        '  next.style.display = next.style.display === "none" ? "table-row" : "none";\n'
        '}\n'
        '\n'
        '// ── Live recalculation engine ────────────────────────────────────────────\n'
        '\n'
        'function calcDrawdown(vals) {\n'
        '  let peak = vals[0];\n'
        '  return vals.map(v => {\n'
        '    if (v > peak) peak = v;\n'
        '    return peak > 0 ? Math.round((v - peak) / peak * 10000) / 100 : 0;\n'
        '  });\n'
        '}\n'
        '\n'
        'function sharpeJS(rets) {\n'
        '  if (rets.length < 2) return 0;\n'
        '  const mean = rets.reduce((a,b)=>a+b,0)/rets.length;\n'
        '  const variance = rets.reduce((a,b)=>a+(b-mean)**2,0)/(rets.length-1);\n'
        '  const std = Math.sqrt(variance);\n'
        '  return std > 0 ? Math.round(mean/std*Math.sqrt(12)*100)/100 : 0;\n'
        '}\n'
        '\n'
        'function cellColorStyle(v) {\n'
        '  if (v === null || v === undefined) return "";\n'
        '  const abs = Math.abs(v);\n'
        '  const intensity = Math.min(abs / 15, 1);\n'
        '  const sat = Math.round(40 + intensity * 55);\n'
        '  const lgt = Math.round(45 - intensity * 15);\n'
        '  const h = v >= 0 ? 142 : 0;\n'
        '  return `background:hsla(${h},${sat}%,${lgt}%,0.85);color:#fff;`;\n'
        '}\n'
        '\n'
        'function fmtPct(v, dec=2) {\n'
        '  if (v === null || v === undefined) return "\\u2014";\n'
        '  return (v>=0?"+":"")+v.toFixed(dec)+"%";\n'
        '}\n'
        'function fmtEur(v) {\n'
        '  if (v === null || v === undefined) return "\\u2014";\n'
        '  return (v>=0?"+":"")+v.toLocaleString("it-IT",{minimumFractionDigits:2,maximumFractionDigits:2})+" \\u20AC";\n'
        '}\n'
        '\n'
        'function recalcola() {\n'
        '  const n_titoli  = parseInt(document.getElementById("ctrl-n").value) || 4;\n'
        '  const cap_t     = parseFloat(document.getElementById("ctrl-cap").value) || 10000;\n'
        '  const costo_pct = parseFloat(document.getElementById("ctrl-costo").value) || 0;\n'
        '  document.getElementById("ctrl-n-lbl").textContent = n_titoli;\n'
        '  // Filtro date\n'
        '  const ds = document.getElementById("ctrl-date-start").value;  // "YYYY-MM"\n'
        '  const de = document.getElementById("ctrl-date-end").value;\n'
        '  const dateStart = ds ? ds + "-01" : "0000-01-01";\n'
        '  const dateEnd   = de ? de + "-31" : "9999-12-31";\n'
        '\n'
        '  let capitale      = n_titoli * cap_t;\n'
        '  const cap_iniziale = capitale;\n'
        '  const eq_labs     = [];\n'
        '  const eq_vals     = [];\n'
        '  const mesi_out    = [];\n'
        '\n'
        '  for (const m of MESI_RAW) {\n'
        '    if (m.data < dateStart || m.data > dateEnd) continue;\n'
        '    const top = m.stocks_all.slice(0, n_titoli);\n'
        '    if (top.length === 0) continue;\n'
        '    const cap_per_t = capitale / n_titoli;\n'
        '    const stocks_calc = top.map(s => {\n'
        '      const r_netto = ((1 + s.ret_lordo/100) * (1 - costo_pct/100) - 1) * 100;\n'
        '      return {\n'
        '        ticker:      s.ticker,\n'
        '        mom_3m:      s.mom_3m,\n'
        '        ret_lordo:   s.ret_lordo,\n'
        '        ret_netto:   Math.round(r_netto*100)/100,\n'
        '        costo_eur:   Math.round(cap_per_t * costo_pct/100 * 100)/100,\n'
        '        guadagno_eur:Math.round(cap_per_t * r_netto/100 * 100)/100,\n'
        '        p_acq:       s.p_acq,\n'
        '        p_vnd:       s.p_vnd,\n'
        '      };\n'
        '    });\n'
        '    const rit_medio = stocks_calc.reduce((a,s)=>a+s.ret_netto,0) / stocks_calc.length;\n'
        '    const cap_fine  = capitale * (1 + rit_medio/100);\n'
        '    eq_labs.push(m.data.slice(0,7));\n'
        '    eq_vals.push(Math.round(capitale*100)/100);\n'
        '    mesi_out.push({\n'
        '      data: m.data, etf: m.etf, etf_ret_3m: m.etf_ret_3m,\n'
        '      tutti_etf: m.tutti_etf,\n'
        '      cap_inizio: Math.round(capitale*100)/100,\n'
        '      cap_fine:   Math.round(cap_fine*100)/100,\n'
        '      rit_medio:  Math.round(rit_medio*100)/100,\n'
        '      stocks: stocks_calc.sort((a,b)=>b.ret_netto-a.ret_netto),\n'
        '    });\n'
        '    capitale = cap_fine;\n'
        '  }\n'
        '\n'
        '  // Last closed month endpoint\n'
        '  eq_labs.push(DATA_FINE_LAB);\n'
        '  eq_vals.push(Math.round(capitale*100)/100);\n'
        '\n'
        '  // Open position\n'
        '  let pos_aperta = null;\n'
        '  if (POS_RAW && POS_RAW.stocks_all) {\n'
        '    const pos_top = POS_RAW.stocks_all.slice(0, n_titoli);\n'
        '    const pos_cap_t = capitale / n_titoli;\n'
        '    const pos_stocks = pos_top.map(s => {\n'
        '      const r_netto = ((1 + s.ret_lordo/100) * (1 - costo_pct/100) - 1) * 100;\n'
        '      return {\n'
        '        ticker: s.ticker, mom_3m: s.mom_3m,\n'
        '        ret_lordo: s.ret_lordo, ret_netto: Math.round(r_netto*100)/100,\n'
        '        costo_eur: Math.round(pos_cap_t * costo_pct/100*100)/100,\n'
        '        guadagno_eur: Math.round(pos_cap_t * r_netto/100*100)/100,\n'
        '        p_acq: s.p_acq, p_oggi: s.p_oggi,\n'
        '      };\n'
        '    });\n'
        '    const pos_rit = pos_stocks.reduce((a,s)=>a+s.ret_netto,0)/pos_stocks.length;\n'
        '    const pos_val = capitale * (1 + pos_rit/100);\n'
        '    pos_aperta = {\n'
        '      data_acquisto: POS_RAW.data_acquisto,\n'
        '      data_oggi: POS_RAW.data_oggi,\n'
        '      etf: POS_RAW.etf,\n'
        '      tutti_etf: POS_RAW.tutti_etf,\n'
        '      capitale_investito: Math.round(capitale*100)/100,\n'
        '      ritorno_parziale: Math.round(pos_rit*100)/100,\n'
        '      valore_attuale: Math.round(pos_val*100)/100,\n'
        '      stocks: pos_stocks.sort((a,b)=>b.ret_netto-a.ret_netto),\n'
        '    };\n'
        '    eq_labs.push(POS_RAW.data_oggi);\n'
        '    eq_vals.push(Math.round(pos_val*100)/100);\n'
        '  }\n'
        '\n'
        '  // Drawdown\n'
        '  const dd_vals = calcDrawdown(eq_vals);\n'
        '  const max_dd  = Math.min(...dd_vals);\n'
        '\n'
        '  // SP500 renormalized\n'
        '  const sp5 = SP500_NORM\n'
        '    ? SP500_NORM.map(v => v !== null ? Math.round(v * cap_iniziale * 100)/100 : null)\n'
        '    : null;\n'
        '\n'
        '  // Stats\n'
        '  const rits        = mesi_out.map(m=>m.rit_medio);\n'
        '  const n_mesi      = rits.length;\n'
        '  const mesi_pos    = rits.filter(r=>r>0).length;\n'
        '  const win_rate    = n_mesi ? Math.round(mesi_pos/n_mesi*1000)/10 : 0;\n'
        '  const rit_medio   = n_mesi ? Math.round(rits.reduce((a,b)=>a+b,0)/n_mesi*100)/100 : 0;\n'
        '  const best        = n_mesi ? Math.max(...rits) : 0;\n'
        '  const worst       = n_mesi ? Math.min(...rits) : 0;\n'
        '  const sharpe      = sharpeJS(rits);\n'
        '  const valore_oggi = pos_aperta ? pos_aperta.valore_attuale : Math.round(capitale*100)/100;\n'
        '  const rend_tot    = Math.round((valore_oggi/cap_iniziale-1)*10000)/100;\n'
        '\n'
        '  // Update stat cards\n'
        '  const $ = id => document.getElementById(id);\n'
        '  const setVal = (id, html, color) => {\n'
        '    const el = $(id);\n'
        '    if (!el) return;\n'
        '    el.innerHTML = html;\n'
        '    if (color) el.style.color = color;\n'
        '  };\n'
        '  setVal("stat-cap-iniz",  cap_iniziale.toLocaleString("it-IT",{maximumFractionDigits:0})+" \\u20AC", "#58a6ff");\n'
        '  setVal("stat-valore",    valore_oggi.toLocaleString("it-IT",{maximumFractionDigits:0})+" \\u20AC", "#00c897");\n'
        '  const guad = valore_oggi - cap_iniziale;\n'
        '  setVal("stat-guadagno",  fmtEur(guad), guad>=0?"#00c897":"#ff4d6d");\n'
        '  setVal("stat-rend-tot",  fmtPct(rend_tot,1), rend_tot>=0?"#00c897":"#ff4d6d");\n'
        '  setVal("stat-winrate",   win_rate+"%", "#00c897");\n'
        '  setVal("stat-rit-medio", fmtPct(rit_medio), rit_medio>=0?"#f0c040":"#ff4d6d");\n'
        '  setVal("stat-best",      "+"+best.toFixed(2)+"%", "#00c897");\n'
        '  setVal("stat-worst",     worst.toFixed(2)+"%", "#ff4d6d");\n'
        '  setVal("stat-maxdd",     max_dd.toFixed(2)+"%", "#ff4d6d");\n'
        '  setVal("stat-sharpe",    sharpe.toFixed(2), "#58a6ff");\n'
        '\n'
        '  // Update charts\n'
        '  ALL_LABELS_CUR = eq_labs;\n'
        '  ALL_VALS_CUR   = eq_vals;\n'
        '  SP500_VALS_CUR = sp5;\n'
        '  DD_VALS_CUR    = dd_vals;\n'
        '  setRange(currentRange, null);\n'
        '\n'
        '  // Rebuild pos aperta\n'
        '  rebuildPosAperta(pos_aperta);\n'
        '\n'
        '  // Rebuild table\n'
        '  rebuildTable(mesi_out, n_titoli, cap_t, costo_pct);\n'
        '\n'
        '  // Rebuild matrix\n'
        '  rebuildMatrix(mesi_out);\n'
        '}\n'
        '\n'
        '// Store current chart data\n'
        'let ALL_LABELS_CUR = ALL_LABELS.slice();\n'
        'let ALL_VALS_CUR   = ALL_VALS.slice();\n'
        'let SP500_VALS_CUR = SP500_VALS ? SP500_VALS.slice() : null;\n'
        'let DD_VALS_CUR    = DD_VALS.slice();\n'
        'let currentRange   = 0;\n'
        '\n'
        'function rebuildPosAperta(pa) {\n'
        '  const el = document.getElementById("pos-aperta-container");\n'
        '  if (!el) return;\n'
        '  if (!pa) { el.innerHTML = ""; return; }\n'
        '  const rc = pa.ritorno_parziale >= 0 ? "#00c897" : "#ff4d6d";\n'
        '  const etf_str = Object.entries(pa.tutti_etf)\n'
        '    .sort((a,b)=>b[1]-a[1])\n'
        '    .map(([k,v]) => k===pa.etf\n'
        '      ? `<b style="color:#f0c040">${k} ${v>=0?"+":""}${v}%</b>`\n'
        '      : `${k} ${v>=0?"+":""}${v}%`)\n'
        '    .join(" | ");\n'
        '  let badges = pa.stocks.map(s => {\n'
        '    const c = s.ret_netto>=0?"#00c897":"#ff4d6d";\n'
        '    const key = `${s.ticker}-${pa.data_acquisto.slice(0,7)}`;\n'
        '    return `<span class="tbadge" onclick="openTickerChart(\'${key}\')" style="border:1px solid ${c};cursor:pointer">\n'
        '      ${s.ticker} <span style="color:${c}">${s.ret_netto>=0?"+":""}${s.ret_netto}%</span>\n'
        '    </span>`;\n'
        '  }).join(" ");\n'
        '  let det_rows = pa.stocks.map(s => {\n'
        '    const c = s.ret_netto>=0?"#00c897":"#ff4d6d";\n'
        '    return `<tr>\n'
        '      <td><b>${s.ticker}</b></td>\n'
        '      <td>${s.p_acq}</td><td>${s.p_oggi}</td>\n'
        '      <td style="color:${s.ret_lordo>=0?"#00c897":"#ff4d6d"}">${fmtPct(s.ret_lordo)}</td>\n'
        '      <td>${s.costo_eur.toFixed(2)} \\u20AC</td>\n'
        '      <td style="color:${c};font-weight:700">${fmtPct(s.ret_netto)}</td>\n'
        '      <td>${fmtPct(s.mom_3m)}</td>\n'
        '      <td style="color:${c}">${fmtEur(s.guadagno_eur)}</td>\n'
        '    </tr>`;\n'
        '  }).join("");\n'
        '  el.innerHTML = `<div style="margin:0 40px 28px;padding:16px 20px;\n'
        '    background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.3);border-radius:12px">\n'
        '    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;margin-bottom:10px">\n'
        '      <span style="color:#58a6ff;font-weight:700">&#128308; Posizione aperta &mdash; ${pa.data_acquisto.slice(0,7)}</span>\n'
        '      <span style="color:var(--muted);font-size:0.8rem">Aggiornato al ${pa.data_oggi}</span>\n'
        '    </div>\n'
        '    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px">\n'
        '      <div><div style="font-size:0.7rem;color:var(--muted)">RENDIMENTO</div>\n'
        '        <div style="color:${rc};font-weight:700">${fmtPct(pa.ritorno_parziale)}</div></div>\n'
        '      <div><div style="font-size:0.7rem;color:var(--muted)">GUADAGNO</div>\n'
        '        <div style="color:${rc};font-weight:700">${fmtEur(pa.valore_attuale - pa.capitale_investito)}</div></div>\n'
        '      <div><div style="font-size:0.7rem;color:var(--muted)">VALORE ATTUALE</div>\n'
        '        <div style="font-weight:700">${pa.valore_attuale.toLocaleString("it-IT",{maximumFractionDigits:0})} \\u20AC</div></div>\n'
        '    </div>\n'
        '    <div style="font-size:0.75rem;color:var(--muted);margin-bottom:10px">${etf_str}</div>\n'
        '    <div style="margin-bottom:12px">${badges}</div>\n'
        '    <table class="inner-table"><thead><tr>\n'
        '      <th>Ticker</th><th>P.Acquisto</th><th>P.Oggi</th><th>Rit.Lordo</th>\n'
        '      <th>Costo \\u20AC</th><th>Rit.Netto</th><th>Mom.3m</th><th>P&amp;L \\u20AC</th>\n'
        '    </tr></thead><tbody>${det_rows}</tbody></table>\n'
        '  </div>`;\n'
        '}\n'
        '\n'
        'function rebuildTable(mesi_out, n_titoli, cap_t, costo_pct) {\n'
        '  const tbody = document.getElementById("tbody-main");\n'
        '  if (!tbody) return;\n'
        '  let html = "";\n'
        '  for (let i = mesi_out.length-1; i >= 0; i--) {\n'
        '    const m = mesi_out[i];\n'
        '    const rc = m.rit_medio >= 0 ? "#00c897" : "#ff4d6d";\n'
        '    const etf_str = Object.entries(m.tutti_etf)\n'
        '      .sort((a,b)=>b[1]-a[1])\n'
        '      .map(([k,v]) => k===m.etf\n'
        '        ? `<b style="color:#f0c040">${k} ${v>=0?"+":""}${v}%</b>`\n'
        '        : `${k} ${v>=0?"+":""}${v}%`)\n'
        '      .join(" | ");\n'
        '    const badges = m.stocks.map(s => {\n'
        '      const c = s.ret_netto>=0?"#00c897":"#ff4d6d";\n'
        '      const key = `${s.ticker}-${m.data.slice(0,7)}`;\n'
        '      return `<span class="tbadge" onclick="openTickerChart(\'${key}\')" style="border:1px solid ${c};cursor:pointer"\n'
        '        title="${s.ticker}: acq ${s.p_acq} vnd ${s.p_vnd} | netto ${s.ret_netto>=0?"+":""}${s.ret_netto}% | mom ${s.mom_3m>=0?"+":""}${s.mom_3m}%">\n'
        '        ${s.ticker} <span style="color:${c}">${s.ret_netto>=0?"+":""}${s.ret_netto}%</span>\n'
        '      </span>`;\n'
        '    }).join("");\n'
        '    const det_rows = m.stocks.map(s => {\n'
        '      const c = s.ret_netto>=0?"#00c897":"#ff4d6d";\n'
        '      return `<tr>\n'
        '        <td><b>${s.ticker}</b></td>\n'
        '        <td>${s.p_acq}</td><td>${s.p_vnd}</td>\n'
        '        <td style="color:${s.ret_lordo>=0?"#00c897":"#ff4d6d"}">${fmtPct(s.ret_lordo)}</td>\n'
        '        <td>${s.costo_eur.toFixed(2)} \\u20AC</td>\n'
        '        <td style="color:${c};font-weight:700">${fmtPct(s.ret_netto)}</td>\n'
        '        <td>${fmtPct(s.mom_3m)}</td>\n'
        '        <td style="color:${c}">${fmtEur(s.guadagno_eur)}</td>\n'
        '      </tr>`;\n'
        '    }).join("");\n'
        '    html += `\n'
        '      <tr class="tr-main" onclick="toggleDetById(\'det-${i}\')">\n'
        '        <td>${m.data.slice(0,7)}</td>\n'
        '        <td><span class="etf-pill">${m.etf}</span></td>\n'
        '        <td style="font-size:0.75rem;color:var(--muted)">${etf_str}</td>\n'
        '        <td>${badges}</td>\n'
        '        <td style="color:${rc};font-weight:700">${fmtPct(m.rit_medio)}</td>\n'
        '        <td style="font-weight:600">${m.cap_fine.toLocaleString("it-IT",{maximumFractionDigits:0})} \\u20AC</td>\n'
        '      </tr>\n'
        '      <tr id="det-${i}" style="display:none">\n'
        '        <td colspan="6">\n'
        '          <table class="inner-table"><thead><tr>\n'
        '            <th>Ticker</th><th>P.Acquisto</th><th>P.Vendita</th><th>Rit.Lordo</th>\n'
        '            <th>Costo \\u20AC</th><th>Rit.Netto</th><th>Mom.3m</th><th>P&amp;L \\u20AC</th>\n'
        '          </tr></thead><tbody>${det_rows}</tbody></table>\n'
        '        </td>\n'
        '      </tr>`;\n'
        '  }\n'
        '  tbody.innerHTML = html;\n'
        '}\n'
        '\n'
        'function toggleDetById(id) {\n'
        '  const el = document.getElementById(id);\n'
        '  if (el) el.style.display = el.style.display === "none" ? "table-row" : "none";\n'
        '}\n'
        '\n'
        'function rebuildMatrix(mesi_out) {\n'
        '  const el = document.getElementById("matrix-container");\n'
        '  if (!el) return;\n'
        '  const by_year = {};\n'
        '  for (const m of mesi_out) {\n'
        '    const y = parseInt(m.data.slice(0,4));\n'
        '    const mo = parseInt(m.data.slice(5,7));\n'
        '    if (!by_year[y]) by_year[y] = {};\n'
        '    by_year[y][mo] = m.rit_medio;\n'
        '  }\n'
        '  const years = Object.keys(by_year).map(Number).sort();\n'
        '  const MESI_IT = ["","Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"];\n'
        '  let hdr = "<tr><th>Anno</th>" + MESI_IT.slice(1).map(m=>`<th>${m}</th>`).join("") + "<th>Anno</th></tr>";\n'
        '  let rows = "";\n'
        '  for (const y of years) {\n'
        '    let ann = 1;\n'
        '    let row = `<tr><td>${y}</td>`;\n'
        '    for (let mo=1; mo<=12; mo++) {\n'
        '      const v = by_year[y][mo];\n'
        '      if (v === undefined) {\n'
        '        row += \'<td class="pm-empty">\\u2014</td>\';\n'
        '      } else {\n'
        '        ann *= (1 + v/100);\n'
        '        row += `<td style="${cellColorStyle(v)}">${fmtPct(v,1)}</td>`;\n'
        '      }\n'
        '    }\n'
        '    const ann_pct = Math.round((ann-1)*10000)/100;\n'
        '    row += `<td class="pm-annual" style="${cellColorStyle(ann_pct)}">${fmtPct(ann_pct,1)}</td></tr>`;\n'
        '    rows += row;\n'
        '  }\n'
        '  el.innerHTML = `<div class="matrix-wrap"><h2>Performance mensile</h2>\n'
        '    <table class="perf-matrix"><thead>${hdr}</thead><tbody>${rows}</tbody></table>\n'
        '  </div>`;\n'
        '}\n'
        '\n'
        'let _ctrlTimer = null;\n'
        'function onCtrl() {\n'
        '  clearTimeout(_ctrlTimer);\n'
        '  _ctrlTimer = setTimeout(recalcola, 300);\n'
        '}\n'
        '\n'
        '// Inizializza date picker con min/max dai dati disponibili\n'
        '(function() {\n'
        '  const dates = MESI_RAW.map(m => m.data.slice(0,7)).filter(Boolean);\n'
        '  if (!dates.length) return;\n'
        '  const minM = dates[0];\n'
        '  const maxM = dates[dates.length-1];\n'
        '  const ds = document.getElementById("ctrl-date-start");\n'
        '  const de = document.getElementById("ctrl-date-end");\n'
        '  ds.min = minM; ds.max = maxM; ds.value = minM;\n'
        '  de.min = minM; de.max = maxM; de.value = maxM;\n'
        '})();\n'
        '// Run initial calculation\n'
        'recalcola();\n'
        '\n'
        '// Modal ticker chart\n'
        'let modalChartInstance = null;\n'
        '\n'
        '// Plugin canvas custom per frecce ingresso/uscita\n'
        'const tradeArrowsPlugin = {\n'
        '  id: "tradeArrows",\n'
        '  afterDatasetsDraw(chart, _args, opts) {\n'
        '    if (!opts || opts.entryIdx === undefined) return;\n'
        '    const { ctx, scales } = chart;\n'
        '    function drawArrow(idx, price, up, color) {\n'
        '      if (idx < 0 || price == null) return;\n'
        '      const x = scales.x.getPixelForValue(idx);\n'
        '      const y = scales.y.getPixelForValue(price);\n'
        '      const s = 13;\n'
        '      ctx.save();\n'
        '      ctx.fillStyle = color;\n'
        '      ctx.strokeStyle = "rgba(255,255,255,0.7)";\n'
        '      ctx.lineWidth = 1.5;\n'
        '      ctx.shadowColor = color;\n'
        '      ctx.shadowBlur = 6;\n'
        '      ctx.beginPath();\n'
        '      if (up) {\n'
        '        ctx.moveTo(x,       y - s);\n'
        '        ctx.lineTo(x - s,   y + s * 0.5);\n'
        '        ctx.lineTo(x + s,   y + s * 0.5);\n'
        '      } else {\n'
        '        ctx.moveTo(x,       y + s);\n'
        '        ctx.lineTo(x - s,   y - s * 0.5);\n'
        '        ctx.lineTo(x + s,   y - s * 0.5);\n'
        '      }\n'
        '      ctx.closePath();\n'
        '      ctx.fill();\n'
        '      ctx.stroke();\n'
        '      ctx.restore();\n'
        '    }\n'
        '    drawArrow(opts.entryIdx, opts.entryPrice, true,  "#00c897");\n'
        '    drawArrow(opts.exitIdx,  opts.exitPrice,  false, "#ff4d6d");\n'
        '  }\n'
        '};\n'
        'Chart.register(tradeArrowsPlugin);\n'
        '\n'
        'function openTickerChart(key) {\n'
        '  const tc = TICKER_CHARTS[key];\n'
        '  if (!tc) return;\n'
        '  document.getElementById("modal-title").textContent = tc.ticker + "  \u2014  " + tc.month;\n'
        '  // Trova indici con >= / <= per gestire giorni non di borsa\n'
        '  let entryIdx = tc.dates.findIndex(d => d >= tc.entry_date);\n'
        '  let exitIdx  = -1;\n'
        '  for (let i = tc.dates.length - 1; i >= 0; i--) {\n'
        '    if (tc.dates[i] <= tc.exit_date) { exitIdx = i; break; }\n'
        '  }\n'
        '  // Prezzi effettivi alle date trovate\n'
        '  const entryPx = entryIdx >= 0 ? tc.prices[entryIdx] : tc.entry_price;\n'
        '  const exitPx  = exitIdx  >= 0 ? tc.prices[exitIdx]  : tc.exit_price;\n'
        '  if (modalChartInstance) { modalChartInstance.destroy(); modalChartInstance = null; }\n'
        '  const mCtx = document.getElementById("modalChart").getContext("2d");\n'
        '  modalChartInstance = new Chart(mCtx, {\n'
        '    type: "line",\n'
        '    data: {\n'
        '      labels: tc.dates,\n'
        '      datasets: [{\n'
        '        label: tc.ticker,\n'
        '        data: tc.prices,\n'
        '        borderColor: "#58a6ff",\n'
        '        backgroundColor: "rgba(88,166,255,0.06)",\n'
        '        fill: true, tension: 0.2, borderWidth: 2, pointRadius: 0,\n'
        '      }]\n'
        '    },\n'
        '    options: {\n'
        '      responsive: true,\n'
        '      maintainAspectRatio: false,\n'
        '      plugins: {\n'
        '        legend: { display: false },\n'
        '        tradeArrows: { entryIdx: entryIdx, entryPrice: entryPx * 0.984,\n'
        '                       exitIdx:  exitIdx,  exitPrice:  exitPx  * 1.016 },\n'
        '        tooltip: {\n'
        '          backgroundColor: "#161b22", borderColor: "#30363d", borderWidth: 1,\n'
        '          titleColor: "#e6edf3", bodyColor: "#8b949e",\n'
        '          callbacks: { label: c => " $" + c.raw.toFixed(2) }\n'
        '        }\n'
        '      },\n'
        '      scales: {\n'
        '        x: { ticks: { color: "#8b949e", maxTicksLimit: 10, font: { size: 10 } }, grid: { color: "rgba(48,54,61,0.5)" } },\n'
        '        y: { ticks: { color: "#8b949e", font: { size: 10 }, callback: v => "$"+v.toFixed(2) }, grid: { color: "rgba(48,54,61,0.5)" } }\n'
        '      }\n'
        '    }\n'
        '  });\n'
        '  const entryDateLbl = entryIdx >= 0 ? tc.dates[entryIdx] : tc.entry_date;\n'
        '  const exitDateLbl  = exitIdx  >= 0 ? tc.dates[exitIdx]  : tc.exit_date;\n'
        '  const lordo = (exitPx && entryPx) ? ((exitPx - entryPx) / entryPx * 100).toFixed(2) : "—";\n'
        '  const lc = parseFloat(lordo) >= 0 ? "#00c897" : "#ff4d6d";\n'
        '  document.getElementById("modal-stats").innerHTML =\n'
        '    `<div class="modal-stat">&#9650; Ingresso <span>${entryDateLbl}</span> @ <span>$${entryPx != null ? entryPx.toFixed(2) : "—"}</span></div>`\n'
        '   +`<div class="modal-stat">&#9660; Uscita <span>${exitDateLbl}</span> @ <span>$${exitPx != null ? exitPx.toFixed(2) : "—"}</span></div>`\n'
        '   +`<div class="modal-stat">Rit. lordo <span style="color:${lc};font-weight:700">${lordo >= 0 ? "+" : ""}${lordo}%</span></div>`;\n'
        '  document.getElementById("modal").style.display = "flex";\n'
        '}\n'
        '\n'
        'function closeModal() {\n'
        '  document.getElementById("modal").style.display = "none";\n'
        '  if (modalChartInstance) { modalChartInstance.destroy(); modalChartInstance = null; }\n'
        '}\n'
        '\n'
        'document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });\n'
        'document.getElementById("modal").addEventListener("click", e => { if (e.target.id === "modal") closeModal(); });\n'
        '\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )

    import os as _os; _os.makedirs('docs', exist_ok=True)
    output_path = _os.path.join('docs', 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report HTML salvato: {output_path}")
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data = download_all_data()
    risultati, stats = run_backtest(data)

    if not risultati:
        print("Nessun risultato di backtest. Uscita.")
        sys.exit(1)

    pos_aperta = calcola_posizione_aperta(data, risultati)

    # equity curve
    equity_vals = [r['capitale_inizio'] for r in risultati]
    equity_vals.append(risultati[-1]['capitale_fine'])
    if pos_aperta:
        equity_vals.append(pos_aperta['valore_attuale'])

    drawdown_vals, max_dd = calcola_drawdown(equity_vals)

    cap_labs = [r['data'][:7] for r in risultati]
    cap_labs.append(DATA_FINE.strftime('%Y-%m'))
    if pos_aperta:
        cap_labs.append(DATA_OGGI.strftime('%Y-%m-%d'))

    cap_vals = [r['capitale_inizio'] for r in risultati]
    cap_vals.append(risultati[-1]['capitale_fine'])
    if pos_aperta:
        cap_vals.append(pos_aperta['valore_attuale'])

    sp500_vals    = get_sp500_normalizzato(data, cap_labs, stats['cap_iniziale'])
    ticker_charts = build_ticker_charts(data, risultati, pos_aperta)
    perf_matrix, annual_returns = build_perf_matrix(risultati)

    with open('backtest_risultati.json', 'w', encoding='utf-8') as f:
        json.dump(
            {'stats': stats, 'mesi': risultati, 'posizione_aperta': pos_aperta},
            f, ensure_ascii=False, indent=2
        )
    print("JSON salvato: backtest_risultati.json")

    # Build MESI_RAW for JS
    mesi_raw_js = []
    for r in risultati:
        mesi_raw_js.append({
            'data':        r['data'],
            'etf':         r['etf'],
            'etf_ret_3m':  r['etf_ritorno_3m'],
            'tutti_etf':   r['tutti_etf'],
            'stocks_all':  r.get('stocks_all', []),
        })

    # Build SP500_NORM (normalized to 1.0)
    sp500_norm_js = None
    if sp500_vals and stats['cap_iniziale'] > 0:
        sp500_norm_js = [
            round(v / stats['cap_iniziale'], 6) if v is not None else None
            for v in sp500_vals
        ]

    # Build POS_RAW
    pos_raw_js = None
    if pos_aperta:
        pos_raw_js = {
            'data_acquisto': pos_aperta['data_acquisto'],
            'data_oggi':     pos_aperta['data_oggi'],
            'etf':           pos_aperta.get('etf', ''),
            'tutti_etf':     pos_aperta.get('tutti_etf', {}),
            'stocks_all':    pos_aperta.get('stocks_all', []),
        }

    genera_html(risultati, stats, cap_labs, cap_vals, sp500_vals,
                drawdown_vals, max_dd, ticker_charts, perf_matrix,
                annual_returns, pos_aperta,
                mesi_raw_js=mesi_raw_js, sp500_norm_js=sp500_norm_js, pos_raw_js=pos_raw_js)

    print('GitHub Actions: report salvato in docs/')
