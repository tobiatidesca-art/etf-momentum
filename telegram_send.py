#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invia su Telegram:
  1. Messaggio testo con posizione aperta
  2. Immagine PNG con tabella performance mensile ultimi 5 anni
Legge TELEGRAM_TOKEN e TELEGRAM_CHAT_IDS da variabili d'ambiente (GitHub Secrets).
"""

import io
import json
import os
import ssl
import urllib.request
from datetime import date

GITHUB_PAGES_URL = 'https://tobiatidesca-art.github.io/etf-momentum/'
MESI_IT = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
           'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get_config():
    token = os.environ.get('TELEGRAM_TOKEN', '')
    ids_raw = os.environ.get('TELEGRAM_CHAT_IDS', '')
    chat_ids = [int(x.strip()) for x in ids_raw.split(',') if x.strip()]
    return token, chat_ids


def _send_text(token, chat_id, text):
    data = json.dumps({'chat_id': chat_id, 'text': text}).encode('utf-8')
    req = urllib.request.Request(
        f'https://api.telegram.org/bot{token}/sendMessage',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode())


def _send_photo(token, chat_id, img_bytes):
    boundary = b'----TGBoundary'
    body  = b'--' + boundary + b'\r\n'
    body += b'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
    body += str(chat_id).encode() + b'\r\n'
    body += b'--' + boundary + b'\r\n'
    body += b'Content-Disposition: form-data; name="photo"; filename="tabella.png"\r\n'
    body += b'Content-Type: image/png\r\n\r\n'
    body += img_bytes + b'\r\n'
    body += b'--' + boundary + b'--\r\n'
    req = urllib.request.Request(
        f'https://api.telegram.org/bot{token}/sendPhoto',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary.decode()}'},
    )
    with urllib.request.urlopen(req, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode())


def build_testo(pa):
    if not pa:
        return 'Nessuna posizione aperta al momento.\n\n' + GITHUB_PAGES_URL

    guadagno  = pa['valore_attuale'] - pa['capitale_investito']
    rend      = pa['ritorno_parziale']
    rend_icon = '📈' if rend >= 0 else '📉'
    etf       = pa['etf']

    titoli_lines = []
    for t in pa['dettaglio']:
        r    = t['ritorno_netto_pct']
        icon = '🟢' if r >= 0 else '🔴'
        titoli_lines.append(
            f"  {icon} {t['ticker']}   {r:+.2f}%"
            f"   ${t['prezzo_acquisto']:.2f} → ${t['prezzo_oggi']:.2f}"
        )

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 ETF MOMENTUM — {pa['data_acquisto'][:7]}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"🏷  ETF selezionato:  {etf}\n"
        f"📅  Acquisto:         {pa['data_acquisto']}\n"
        f"🕐  Aggiornato al:    {pa['data_oggi']}\n"
        f"\n"
        f"💰  Investito:   {pa['capitale_investito']:>14,.0f} €\n"
        f"💵  Valore oggi: {pa['valore_attuale']:>14,.0f} €\n"
        f"{rend_icon}  Rendimento: {rend:>+13.2f}%\n"
        f"✅  Guadagno:    {guadagno:>+14,.0f} €\n"
        f"\n"
        f"📌 Titoli in portafoglio\n"
        f"   (selezionati da {etf}, non l'ETF stesso)\n"
        f"\n"
        + '\n'.join(titoli_lines)
        + f"\n\n🌐 Report completo: {GITHUB_PAGES_URL}"
    )


def build_tabella_png(mesi, oggi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    anno_fine   = oggi.year
    anno_inizio = anno_fine - 4
    anni        = list(range(anno_inizio, anno_fine + 1))

    perf = {}
    for m in mesi:
        try:
            dt = date.fromisoformat(m['data'])
            perf[(dt.year, dt.month)] = m['ritorno_mese_pct']
        except Exception:
            pass

    col_labels = MESI_IT + ['Anno']
    n_cols = len(col_labels)

    cell_text   = []
    cell_colors = []

    for anno in anni:
        row_vals  = []
        row_cols  = []
        vals_anno = []
        for mese in range(1, 13):
            if date(anno, mese, 1) >= date(oggi.year, oggi.month, 1):
                row_vals.append('')
                row_cols.append('#1c2330')
            else:
                v = perf.get((anno, mese))
                if v is None:
                    row_vals.append('—')
                    row_cols.append('#1c2330')
                else:
                    vals_anno.append(v)
                    row_vals.append(f'{v:+.1f}%')
                    intensity = min(abs(v) / 20, 1.0)
                    if v >= 0:
                        row_cols.append((0.15 * (1 - intensity * 0.6),
                                         0.55,
                                         0.25 * (1 - intensity * 0.6),
                                         0.65))
                    else:
                        row_cols.append((0.85,
                                         0.15 * (1 - intensity * 0.7),
                                         0.15 * (1 - intensity * 0.7),
                                         0.65))

        if vals_anno:
            ann = 1.0
            for v in vals_anno:
                ann *= (1 + v / 100)
            ann = (ann - 1) * 100
            row_vals.append(f'{ann:+.1f}%')
            row_cols.append((0.1, 0.6, 0.2, 0.85) if ann >= 0 else (0.7, 0.1, 0.1, 0.85))
        else:
            row_vals.append('—')
            row_cols.append('#1c2330')

        cell_text.append(row_vals)
        cell_colors.append(row_cols)

    fig_w = n_cols * 0.78 + 0.8
    fig_h = len(anni) * 0.52 + 1.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    ax.set_title('Performance mensile — ultimi 5 anni',
                 color='#58a6ff', fontsize=12, fontweight='bold',
                 pad=14, loc='left')

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=[str(a) for a in anni],
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#30363d')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#8b949e', fontweight='bold', fontsize=8.5)
        elif col == -1:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#58a6ff', fontweight='bold')
        else:
            cell.set_text_props(color='#e6edf3', fontsize=8.5)
        if col == n_cols - 1 and row > 0:
            cell.set_text_props(fontweight='bold', fontsize=9)

    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    token, chat_ids = _get_config()
    if not token or not chat_ids:
        print('TELEGRAM_TOKEN o TELEGRAM_CHAT_IDS mancanti — skip invio.')
        return

    with open('backtest_risultati.json', encoding='utf-8') as f:
        d = json.load(f)

    pa   = d.get('posizione_aperta')
    mesi = d.get('mesi', [])
    oggi = date.today()

    testo   = build_testo(pa)
    img_png = build_tabella_png(mesi, oggi)

    for chat_id in chat_ids:
        print(f'Invio a chat_id {chat_id}...')
        _send_text(token, chat_id, testo)
        _send_photo(token, chat_id, img_png)
        print(f'  OK')

    print('Invio Telegram completato.')


if __name__ == '__main__':
    main()
