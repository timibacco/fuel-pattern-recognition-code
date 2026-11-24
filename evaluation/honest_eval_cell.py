# =====================================================================
# HONEST EVALUATION CELL -- paste into ONE new Colab cell and run it.
# Needs only: the CSV (loaded below if 'df' is not already in memory).
#
# What it fixes compared with the earlier cells:
# 1. Purged split: inside every landmark block the first 75% of rows
# train, then GAP rows are thrown away, and only the last 25% test.
# Windows are built separately inside each part, so no row is shared.
# 2. The autoencoder is RE-PRETRAINED here WITHOUT any test rows, and the
# scaler is fit on those pretraining rows only (no test statistics).
# 3. No checkpoint picking on the test set: epochs are fixed in advance
# and the LAST epoch is reported.
# 4. Five seeds, mean +/- sd, plus paired SSL-vs-no-SSL differences.
# 5. Baselines: majority class and a random forest on window statistics.
# Do NOT change the epoch counts after you see the results.
# =====================================================================
import json, collections
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# ------------------------------ config -------------------------------
FEATURES = ['rpm', 'vehicle_speed_kmph', 'maf_g_s']
WINDOW, LABELED_STRIDE, SSL_STRIDE = 20, 3, 10
GAP = 20 # rows dropped before the test part (>= WINDOW)
TEST_FRAC = 0.25
SSL_EPOCHS, FT_EPOCHS = 50, 50
SEEDS = [0, 1, 2, 3, 4]
VAL_SESSION = '6' # held out for SSL validation, as in your notebook
CSV_PATH = '/content/drive/MyDrive/Shared fyp/merged_feature_table.csv'

collapse_map = {
    "Opa-dam junction": "O1: Opa-dam Junction",
    "2nd turn/ctw": "O2a: 2nd Turn-ctw",
    "3rd bump": "O2b: 3rd Bump",
    "Banking area": "O3: Banking Area",
    "Bus stop": "O4: Bus Stop",
    "FRSC/bus stop": "I1: FRSC-Bus Stop",
    "Religion Ground": "I2: Religion Ground",
    "Education": "I3: Education-Bump",
    "Bump": "I3: Education-Bump",
    "Adesakin footpath/maintenance": "I4: Adesakin Footpath",
}

# ----------------------------- data prep -----------------------------
try:
    _src = df
except NameError:
    _src = pd.read_csv(CSV_PATH, dtype={'session_id': str})
d = _src.reset_index(drop=True).copy()
d['session_id'] = d['session_id'].astype(str).str.strip()
d['segment_label'] = d['segment_label_fine'].map(collapse_map)
_unmapped = set(d['segment_label_fine'].dropna().unique()) - set(collapse_map)
if _unmapped:
    print('NOTE: fine labels not in collapse_map (left unlabeled):', sorted(_unmapped))
print('sessions in data:', sorted(d['session_id'].unique()))

dt = d.groupby('session_id')['time_s'].diff()
if (dt.dropna() <= 0).any():
    print("WARNING: time_s is not strictly increasing inside some session; check row order.")
med_dt = float(dt.median())
print(f"median sample interval = {med_dt:.3f} s -> a window of {WINDOW} rows is about {WINDOW*med_dt:.1f} s "
     f"(use this number for [W] in the paper)")
print("NaNs per feature:", d[FEATURES].isna().sum().to_dict())

# contiguous labeled blocks (new block on label change OR a time jump, e.g. GPS dropout)
lab = d[d['segment_label'].notna()].copy()
changed = (lab[['session_id', 'segment_label']] != lab[['session_id', 'segment_label']].shift()).any(axis=1)
jump = lab.groupby('session_id')['time_s'].diff() > 5 * med_dt
lab['block'] = (changed | jump.fillna(False)).cumsum()

classes = sorted(lab['segment_label'].unique())
K = len(classes)
cid = {c: i for i, c in enumerate(classes)}


def windows_from(feats, window, stride):
    n = len(feats)
    if n < window:
       return np.empty((0, window, feats.shape[1]))
    w = np.stack([feats[s:s + window] for s in range(0, n - window + 1, stride)])
    return w[np.isfinite(w).all(axis=(1, 2))]


Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
test_rows = []
for b, blk in lab.groupby('block'):
    n = len(blk)
    cut = int(n * (1 - TEST_FRAC))
    tr = blk.iloc[:max(cut - GAP, 0)] # train part ends GAP rows before the test part
    te = blk.iloc[cut:] # test part
    y = cid[blk['segment_label'].iloc[0]]
    wtr = windows_from(tr[FEATURES].values, WINDOW, LABELED_STRIDE)
    wte = windows_from(te[FEATURES].values, WINDOW, LABELED_STRIDE)
    if len(wtr):
       Xtr_l.append(wtr); ytr_l += [y] * len(wtr)
    if len(wte):
       Xte_l.append(wte); yte_l += [y] * len(wte)
    test_rows += list(te.index)

Xtr_raw = np.concatenate(Xtr_l); ytr = np.array(ytr_l)
Xte_raw = np.concatenate(Xte_l); yte = np.array(yte_l)
test_rows = np.array(test_rows)

print(f"\nlabeled blocks: {lab['block'].nunique()} train windows: {len(ytr)} test windows: {len(yte)}")
print(f"{'class':28s} train test")
for c in classes:
    a, b = int((ytr == cid[c]).sum()), int((yte == cid[c]).sum())
    flag = " <-- EMPTY" if a == 0 or b == 0 else ""
    print(f"{c:28s} {a:5d} {b:5d}{flag}")

# unlabeled windows for SSL: every session except the validation one,
# skipping any window that touches a TEST row
ssl_tr, ssl_val = [], []
for sid, g in d.groupby('session_id', sort=False):
    idx, feats = g.index.values, g[FEATURES].values
    for s in range(0, len(g) - WINDOW + 1, SSL_STRIDE):
       w = feats[s:s + WINDOW]
       if not np.isfinite(w).all():
           continue
       if sid == VAL_SESSION:
           ssl_val.append(w)
       elif not np.isin(idx[s:s + WINDOW], test_rows).any():
           ssl_tr.append(w)
if not ssl_val:
    raise ValueError(f"No windows for VAL_SESSION={VAL_SESSION!r}; set it to one of the session ids printed above.")
ssl_tr, ssl_val = np.stack(ssl_tr), np.stack(ssl_val)
print(f"SSL pretraining windows: {len(ssl_tr)} (test rows excluded) SSL validation windows: {len(ssl_val)}")

mean = ssl_tr.reshape(-1, len(FEATURES)).mean(axis=0) # scaler fit on pretraining rows only
std = ssl_tr.reshape(-1, len(FEATURES)).std(axis=0) + 1e-6
sc = lambda X: (X - mean) / std
Xtr, Xte, ssl_tr_s, ssl_val_s = sc(Xtr_raw), sc(Xte_raw), sc(ssl_tr), sc(ssl_val)


def stat_feats(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :] - X[:, 0, :]], axis=1)


def metrics(y_true, y_pred):
    # macro-F1 averages only classes that HAVE test windows (a class with no test windows cannot be scored)
    present = sorted(set(np.asarray(y_true).tolist()))
    L = list(range(K))
    return dict(acc=accuracy_score(y_true, y_pred),
               macro_f1=f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0),
               recall=recall_score(y_true, y_pred, labels=L, average=None, zero_division=0).tolist(),
               precision=precision_score(y_true, y_pred, labels=L, average=None, zero_division=0).tolist(),
               cm=confusion_matrix(y_true, y_pred, labels=L).tolist())


# ==== MODELS (torch) =================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = lambda a, dt_=torch.float32: torch.tensor(a, dtype=dt_).to(device)


class AE(nn.Module):
    def __init__(self, n_features=3, hidden=32, latent=16):
       super().__init__()
       self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
       self.to_latent = nn.Linear(hidden, latent)
       self.from_latent = nn.Linear(latent, hidden)
       self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
       self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
       _, (h, _) = self.encoder(x)
       z = self.to_latent(h[-1])
       h0 = self.from_latent(z).unsqueeze(0)
       dec_in = h0.permute(1, 0, 2).repeat(1, x.size(1), 1)
       out, _ = self.decoder(dec_in, (h0, torch.zeros_like(h0)))
       return self.out(out)


class Clf(nn.Module):
    def __init__(self, n_features=3, hidden=32, latent=16, num_classes=9, dropout=0.3):
       super().__init__()
       self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
       self.to_latent = nn.Linear(hidden, latent)
       self.dropout = nn.Dropout(dropout)
       self.head = nn.Linear(latent, num_classes)

    def forward(self, x):
       _, (h, _) = self.encoder(x)
       return self.head(self.dropout(self.to_latent(h[-1])))


def pretrain(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ae = AE().to(device)
    opt, crit = torch.optim.Adam(ae.parameters(), lr=1e-3), nn.MSELoss()
    loader = DataLoader(TensorDataset(T(ssl_tr_s)), batch_size=64, shuffle=True)
    for _ in range(SSL_EPOCHS):
       ae.train()
       for (xb,) in loader:
           opt.zero_grad(); loss = crit(ae(xb), xb); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
       tr_mse = crit(ae(T(ssl_tr_s)), T(ssl_tr_s)).item()
       val_mse = crit(ae(T(ssl_val_s)), T(ssl_val_s)).item()
    return ae, tr_mse, val_mse


def finetune(seed, ae=None):
    torch.manual_seed(seed)
    clf = Clf(num_classes=K).to(device)
    if ae is not None:
       sd, pre = clf.state_dict(), ae.state_dict()
       for k in ['encoder.weight_ih_l0', 'encoder.weight_hh_l0', 'encoder.bias_ih_l0',
                'encoder.bias_hh_l0', 'to_latent.weight', 'to_latent.bias']:
           sd[k] = pre[k]
       clf.load_state_dict(sd)
    counts = np.bincount(ytr, minlength=K)
    w = np.sqrt(counts.sum() / (K * np.maximum(counts, 1)))
    crit = nn.CrossEntropyLoss(weight=T(w))
    opt = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(T(Xtr), T(ytr, torch.long)), batch_size=32, shuffle=True)
    for _ in range(FT_EPOCHS): # fixed epochs, LAST epoch is reported
       clf.train()
       for xb, yb in loader:
           opt.zero_grad(); crit(clf(xb), yb).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
       p_tr = clf(T(Xtr)).argmax(1).cpu().numpy()
       p_te = clf(T(Xte)).argmax(1).cpu().numpy()
    m = metrics(yte, p_te)
    m['train_acc'] = accuracy_score(ytr, p_tr)
    return m


# ------------------------------ run ----------------------------------
res = {'ssl': [], 'nossl': [], 'rf': [], 'ae_mse': []}
for sd in SEEDS:
    ae, tr_mse, val_mse = pretrain(sd)
    res['ae_mse'].append((tr_mse, val_mse))
    res['ssl'].append(finetune(sd, ae))
    res['nossl'].append(finetune(sd, None))
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=sd, n_jobs=-1)
    rf.fit(stat_feats(Xtr), ytr)
    m = metrics(yte, rf.predict(stat_feats(Xte)))
    m['train_acc'] = accuracy_score(ytr, rf.predict(stat_feats(Xtr)))
    res['rf'].append(m)
    print(f"seed {sd}: SSL acc={res['ssl'][-1]['acc']:.3f} F1={res['ssl'][-1]['macro_f1']:.3f} | "
         f"noSSL acc={res['nossl'][-1]['acc']:.3f} F1={res['nossl'][-1]['macro_f1']:.3f} | "
         f"RF acc={m['acc']:.3f} F1={m['macro_f1']:.3f} | AE val MSE={val_mse:.4f}")

maj = collections.Counter(yte.tolist()).most_common(1)[0][0]
maj_m = metrics(yte, np.full_like(yte, maj))
maj_m['train_acc'] = float((ytr == maj).mean())


def ms(name, key):
    v = np.array([r[key] for r in res[name]])
    return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}"


print("\n================ TABLE 1 (mean +/- sd over %d seeds, LAST epoch) ================" % len(SEEDS))
print(f"{'model':26s} {'train acc':16s} {'test acc':16s} {'test macro-F1':16s}")
print(f"{'Majority class':26s} {maj_m['train_acc']:<16.3f} {maj_m['acc']:<16.3f} {maj_m['macro_f1']:<16.3f}")
for name, label in [('rf', 'Random forest (stats)'), ('nossl', 'LSTM, no pretraining'), ('ssl', 'LSTM, SSL pretraining')]:
    print(f"{label:26s} {ms(name,'train_acc'):16s} {ms(name,'acc'):16s} {ms(name,'macro_f1'):16s}")

dacc = np.array([a['acc'] - b['acc'] for a, b in zip(res['ssl'], res['nossl'])])
df1 = np.array([a['macro_f1'] - b['macro_f1'] for a, b in zip(res['ssl'], res['nossl'])])
print(f"\nPaired SSL - noSSL: acc {dacc.mean():+.3f} (per seed {np.round(dacc,3).tolist()}), "
     f"macro-F1 {df1.mean():+.3f} (per seed {np.round(df1,3).tolist()}); SSL better in {(df1>0).sum()}/{len(SEEDS)} seeds by F1")
print(f"Autoencoder MSE (train, val) per seed: {[(round(a,4), round(b,4)) for a, b in res['ae_mse']]}")

print("\nPer-class recall / precision (mean over seeds); n = test windows (0 means the class cannot be scored):")
print(f"{'class':26s} {'n':>4s} | {'recall SSL':>10s} {'noSSL':>6s} {'RF':>6s} | {'prec SSL':>9s} {'noSSL':>6s} {'RF':>6s}")
for i, c in enumerate(classes):
    r = lambda n, k: np.mean([x[k][i] for x in res[n]])
    print(f"{c:26s} {int((yte == i).sum()):4d} | {r('ssl','recall'):10.2f} {r('nossl','recall'):6.2f} {r('rf','recall'):6.2f} | "
         f"{r('ssl','precision'):9.2f} {r('nossl','precision'):6.2f} {r('rf','precision'):6.2f}")

json.dump({'config': dict(WINDOW=WINDOW, LABELED_STRIDE=LABELED_STRIDE, SSL_STRIDE=SSL_STRIDE, GAP=GAP,
                        TEST_FRAC=TEST_FRAC, SSL_EPOCHS=SSL_EPOCHS, FT_EPOCHS=FT_EPOCHS, SEEDS=SEEDS,
                        median_dt_s=med_dt, classes=classes),
          'n_train': int(len(ytr)), 'n_test': int(len(yte)), 'majority': maj_m, 'results': res},
         open('honest_eval_results.json', 'w'), indent=1, default=float)
print("\nSaved honest_eval_results.json (upload it to me, or paste the TABLE 1 block above).")
