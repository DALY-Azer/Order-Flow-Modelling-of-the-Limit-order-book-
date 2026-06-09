"""
Génère tous les graphes du notebook `notebook.ipynb` et les enregistre dans le
dossier `figures/` (un PNG par figure).

Reprend à l'identique le pipeline du notebook :
  1. Chargement trades + book
  2. Restriction à la période du book + horodatage
  3. Classification du flux d'ordres (6 types : market / limit / cancellation x buy/sell)
  4. Découpage en événements selon le pas de prix
  5. Agrégation par événement
  6-8. Figures (log-returns, ACF, volumes, profil intraday)

Usage :  python generate_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend non interactif (pas de fenêtre)
import matplotlib.pyplot as plt
from numba import njit

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# 1. Chargement des données
# ============================================================
print("1. Chargement des données ...")
df = pd.read_csv("TOTF_trade_2014_2017.csv.gz")
book = pd.read_csv("TOTF_book_2014-04-13_to_2014-10-12.csv.gz")
print("   trades :", df.shape)
print("   book   :", book.shape)


# ============================================================
# 2. Restriction à la période du book + horodatage
# ============================================================
df["date"] = pd.to_datetime(df["date"])
trades = df[df["date"] <= "2014-10-12"].copy()
trades["dt"] = pd.to_datetime(trades["date"].astype(str) + " " + trades["time"].astype(str))
book["dt"] = pd.to_datetime(book["date"].astype(str) + " " + book["time"].astype(str))
trades = trades.sort_values("dt").reset_index(drop=True)
book = book.sort_values("dt").reset_index(drop=True)
print("   trades (6 mois) :", trades.shape)


# ============================================================
# 3.1 Market orders (Lee-Ready)
# ============================================================
print("3.1 Market orders ...")
trades_book = pd.merge_asof(
    trades, book[["dt", "bid_1", "ask_1"]], on="dt", direction="backward"
)
valid = (trades_book["bid_1"] > 0) & (trades_book["ask_1"] > 0)
trades_book = trades_book[valid].reset_index(drop=True)

sign = np.sign(trades_book["trade.price"].diff()).replace(0, np.nan).ffill()
trades_book["sign"] = sign
midpoint = (trades_book["bid_1"] + trades_book["ask_1"]) / 2
trades_book["order_type"] = np.select(
    [
        trades_book["trade.price"] > midpoint,
        trades_book["trade.price"] < midpoint,
        (trades_book["trade.price"] == midpoint) & (trades_book["sign"] > 0),
        (trades_book["trade.price"] == midpoint) & (trades_book["sign"] < 0),
    ],
    ["market_buy", "market_sell", "market_buy", "market_sell"],
    default="market_buy",
)
market_events = trades_book[["dt", "trade.price", "trade.volume", "order_type"]].rename(
    columns={"trade.price": "price", "trade.volume": "volume"}
)


# ============================================================
# 3.2 Limit orders & cancellations (diff aligné sur le prix)
# ============================================================
print("3.2 Limit orders & cancellations ...")
TICK = 0.005
SENT = -2_000_000_000

bid_p = ["bid_1", "bid_2", "bid_3", "bid_4", "bid_5"]
bid_q = ["bidQ_1", "bidQ_2", "bidQ_3", "bidQ_4", "bidQ_5"]
ask_p = ["ask_1", "ask_2", "ask_3", "ask_4", "ask_5"]
ask_q = ["askQ_1", "askQ_2", "askQ_3", "askQ_4", "askQ_5"]

def _to_arrays(p_cols, q_cols):
    raw = book[p_cols].to_numpy(float)
    P = np.round(raw / TICK)
    P = np.where(np.isnan(raw) | (raw <= 0), SENT, P).astype(np.int32)
    Q = np.nan_to_num(book[q_cols].to_numpy(float))
    return P, Q

Pb, Qb = _to_arrays(bid_p, bid_q)
Pa, Qa = _to_arrays(ask_p, ask_q)
dt_int = book["dt"].to_numpy().astype("datetime64[ns]").astype("int64")

chg = np.empty(len(book), bool); chg[0] = True
chg[1:] = (np.any(Pb[1:] != Pb[:-1], 1) | np.any(Qb[1:] != Qb[:-1], 1)
           | np.any(Pa[1:] != Pa[:-1], 1) | np.any(Qa[1:] != Qa[:-1], 1))
ix = np.flatnonzero(chg)
Pb, Qb, Pa, Qa, dt2 = Pb[ix], Qb[ix], Pa[ix], Qa[ix], dt_int[ix]

_TRI = np.tril(np.ones((5, 5), bool), -1)

def _side_events(P, Q, chunk=2_000_000):
    R, T, D = [], [], []
    n = len(P) - 1
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Pc, Qc = P[s + 1:e + 1], Q[s + 1:e + 1]
        Pp, Qp = P[s:e], Q[s:e]
        eqcc = (Pc[:, :, None] == Pc[:, None, :]); depth_c = (eqcc * Qc[:, None, :]).sum(2)
        eqcp = (Pc[:, :, None] == Pp[:, None, :]); depth_pc = (eqcp * Qp[:, None, :]).sum(2)
        delta_c = depth_c - depth_pc
        first_c = ~(eqcc & _TRI[None]).any(2)
        keep_c = (Pc != SENT) & first_c & (delta_c != 0)
        eqpc = (Pp[:, :, None] == Pc[:, None, :]); present = eqpc.any(2)
        eqpp = (Pp[:, :, None] == Pp[:, None, :]); depth_p = (eqpp * Qp[:, None, :]).sum(2)
        first_p = ~(eqpp & _TRI[None]).any(2)
        keep_g = (Pp != SENT) & (~present) & first_p & (depth_p > 0)
        r, j = np.nonzero(keep_c)
        rg, k = np.nonzero(keep_g)
        R.append(np.concatenate([r, rg]) + s + 1)
        T.append(np.concatenate([Pc[r, j], Pp[rg, k]]))
        D.append(np.concatenate([delta_c[r, j], -depth_p[rg, k]]))
    R = np.concatenate(R); T = np.concatenate(T); D = np.concatenate(D)
    return dt2[R], T, D

bdt, btk, bd = _side_events(Pb, Qb)
adt, atk, ad = _side_events(Pa, Qa)

tr_dt = trades["dt"].to_numpy().astype("datetime64[ns]").astype("int64")
tr_tk = np.round(trades["trade.price"].to_numpy() / TICK).astype(np.int64)
traded_set = set(zip(tr_dt.tolist(), tr_tk.tolist()))

def _build(dts, tks, dlt, side):
    out = []
    add = dlt > 0
    out.append(pd.DataFrame({
        "dt": dts[add].astype("datetime64[ns]"), "price": tks[add] * TICK,
        "volume": dlt[add], "order_type": f"limit_order_{side}"}))
    dr = np.flatnonzero(dlt < 0)
    is_exec = np.fromiter(((int(dts[i]), int(tks[i])) in traded_set for i in dr), bool, len(dr))
    cx = dr[~is_exec]
    out.append(pd.DataFrame({
        "dt": dts[cx].astype("datetime64[ns]"), "price": tks[cx] * TICK,
        "volume": -dlt[cx], "order_type": f"cancellation_{side}"}))
    return out

book_events = pd.concat(_build(bdt, btk, bd, "buy") + _build(adt, atk, ad, "sell"),
                        ignore_index=True)


# ============================================================
# 3.3 Flux d'ordres combiné
# ============================================================
all_events = pd.concat([market_events, book_events], ignore_index=True)
all_events = all_events.sort_values("dt").reset_index(drop=True)
all_events = pd.merge_asof(
    all_events, book[["dt", "bid_1", "ask_1"]], on="dt", direction="backward"
)
all_events["mid"] = (all_events["bid_1"] + all_events["ask_1"]) / 2
all_events.loc[(all_events["bid_1"] <= 0) | (all_events["ask_1"] <= 0), "mid"] = np.nan
all_events["mid"] = all_events["mid"].ffill().bfill()
print("   Total événements :", len(all_events))


# ============================================================
# 4. Découpage en événements selon le pas de prix
# ============================================================
@njit
def build_events(prices, threshold):
    n = len(prices)
    event_id = np.zeros(n, dtype=np.int64)
    current_event = 0
    ref_price = prices[0]
    for i in range(1, n):
        if abs(prices[i] - ref_price) > threshold:
            current_event += 1
            ref_price = prices[i]
        event_id[i] = current_event
    return event_id

mid = all_events["mid"].to_numpy()
all_events["event_id"] = build_events(mid, 0.01)
print("   Nombre d'événements (pas de prix) :", all_events["event_id"].nunique())


# ============================================================
# 5. Agrégation par événement
# ============================================================
counts = (
    all_events.pivot_table(index="event_id", columns="order_type",
                           values="volume", aggfunc="size", fill_value=0)
    .add_prefix("n_")
)
volumes = (
    all_events.pivot_table(index="event_id", columns="order_type",
                           values="volume", aggfunc="sum", fill_value=0)
    .add_prefix("vol_")
)
events = (
    all_events.groupby("event_id")
    .agg(
        start_dt=("dt", "first"), end_dt=("dt", "last"),
        n_orders=("order_type", "size"),
        start_mid=("mid", "first"), end_mid=("mid", "last"),
        bid_1=("bid_1", "last"), ask_1=("ask_1", "last"),
    )
    .join(counts).join(volumes).reset_index()
)


# ============================================================
# 6. Log-returns et autocorrélation
# ============================================================
print("Figures ...")

# --- Figure 1 : log-returns (cellule 19) ---
price = events["start_mid"].to_numpy()
log_ret = np.diff(np.log(price))
log_ret = log_ret[np.isfinite(log_ret)]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(log_ret, lw=0.4)
ax[0].set_title("Log-returns par événement")
ax[0].set_xlabel("événement")
ax[0].set_ylabel(r"$r_t$")
ax[1].hist(log_ret, bins=200)
ax[1].set_title("Distribution des log-returns")
ax[1].set_xlabel(r"$r_t$")
ax[1].set_yscale("log")
fig.tight_layout()
save(fig, "01_log_returns.png")

# --- Figure 2 : ACF (cellule 20) ---
def acf(x, nlags):
    x = x - x.mean()
    var = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / var for k in range(1, nlags + 1)])

nlags = 50
lags = np.arange(nlags + 1)
acf_r = acf(log_ret, nlags)
acf_abs = acf(np.abs(log_ret), nlags)
conf = 1.96 / np.sqrt(len(log_ret))

fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, vals, title in (
    (ax[0], acf_r, "ACF des log-returns $r_t$"),
    (ax[1], acf_abs, "ACF des |log-returns| $|r_t|$"),
):
    a.bar(lags, vals, width=0.6)
    a.axhline(0, color="k", lw=0.8)
    a.axhline(conf, color="r", ls="--", lw=0.8)
    a.axhline(-conf, color="r", ls="--", lw=0.8)
    a.set_title(title)
    a.set_xlabel("lag (événements)")
ax[0].set_ylabel("autocorrélation")
fig.tight_layout()
save(fig, "02_acf.png")


# ============================================================
# 7. Volumes par type d'ordre
# ============================================================
order_types = ["market_buy", "market_sell",
               "limit_order_buy", "limit_order_sell",
               "cancellation_buy", "cancellation_sell"]
colors = dict(zip(order_types, plt.cm.tab10.colors))

# --- Figure 3 : volume par ordre individuel (cellule 22) ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
tot = all_events.groupby("order_type")["volume"].sum().reindex(order_types).fillna(0)
ax[0].barh(order_types, tot.values, color=[colors[t] for t in order_types])
ax[0].set_xlabel("volume total")
ax[0].set_title("7.1 — Volume total par type (par ordre)")
ax[0].invert_yaxis()
for ot in order_types:
    v = np.sort(all_events.loc[all_events["order_type"] == ot, "volume"].to_numpy())
    v = v[v > 0]
    if len(v) == 0:
        continue
    ccdf = 1 - np.arange(len(v)) / len(v)
    ax[1].loglog(v, ccdf, label=ot, color=colors[ot], lw=1.2)
ax[1].set_xlabel("volume v")
ax[1].set_ylabel(r"$P(V > v)$")
ax[1].set_title("7.1 — CCDF des volumes par ordre (log-log)")
ax[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "03_volumes_par_ordre.png")

# --- Figure 4 : volume agrégé par événement (cellule 23) ---
vol_cols = ["vol_" + t for t in order_types]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
tot_e = events[vol_cols].sum()
ax[0].barh(order_types, tot_e.values, color=[colors[t] for t in order_types])
ax[0].set_xlabel("volume total")
ax[0].set_title("7.2 — Volume total par type (par événement)")
ax[0].invert_yaxis()
for ot in order_types:
    v = np.sort(events["vol_" + ot].to_numpy())
    v = v[v > 0]
    if len(v) == 0:
        continue
    ccdf = 1 - np.arange(len(v)) / len(v)
    ax[1].loglog(v, ccdf, label=ot, color=colors[ot], lw=1.2)
ax[1].set_xlabel("volume par événement v")
ax[1].set_ylabel(r"$P(V > v)$")
ax[1].set_title("7.2 — CCDF des volumes par événement (log-log)")
ax[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "04_volumes_par_evenement.png")


# ============================================================
# 8. Profil intraday des flux
# ============================================================
BIN = "5min"
prof = all_events.copy()
prof["date"] = prof["dt"].dt.normalize()
prof["tod"] = prof["dt"].dt.floor(BIN).dt.time
profile = (
    prof.groupby(["date", "tod", "order_type"])["volume"].sum()
        .groupby(level=["tod", "order_type"]).mean()
        .unstack("order_type")
        .sort_index()
)
profile_z = (profile - profile.mean()) / profile.std()

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(profile_z))
for ot in order_types:
    if ot in profile_z:
        ax.plot(x, profile_z[ot], lw=1, label=ot, color=colors[ot])
ticks = np.arange(0, len(profile_z), max(1, len(profile_z) // 8))
ax.set_xticks(ticks)
ax.set_xticklabels([str(profile_z.index[i])[:5] for i in ticks])
ax.set_xlabel("Time of Day")
ax.set_ylabel("Normalized Daily Volume")
ax.set_title("Profil intraday des flux (forme en U)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
save(fig, "05_profil_intraday.png")

print("\nTerminé : 5 figures enregistrées dans", FIG_DIR + "/")
