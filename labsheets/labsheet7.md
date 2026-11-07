# 🖥️ Computer Vision Lab 7 — Sequence Models: RNN, LSTM & GRU on Stock-Market Time Series

## 🎯 Objectives

In this lab, you will:

1. Download real **stock price data** and turn it into **supervised windows** — and see why you model **returns, not prices**.
2. Split **chronologically**, fit scalers on **train only**, and avoid the look-ahead leaks that plague financial ML.
3. Set **honest baselines** (zero-change, majority class, recent volatility) that any model must beat to be worth using.
4. Build a vanilla **RNN**, an **LSTM** and a **GRU** with one `SeqModel` class, and implement an **LSTM cell by hand**.
5. Watch the **vanishing gradient problem** happen on a controlled memory task.
6. Compare **LSTM variants** (stacked, bidirectional, pooling, forget-bias), and apply **gradient clipping**.
7. Evaluate with **walk-forward validation**, the way time series should be tested.
8. Run a **worked example** end to end on a real ticker, then complete a **multi-ticker exercise**.

> ⚠️ **Educational use only. Nothing in this lab is financial advice or a trading system.** Part of the point of the lab is to see how hard it is to extract a usable signal from market data, and how easy it is to fool yourself into thinking you have.

> 📌 **This lab continues from [Lab 6-2](./labsheet6-2.md)** and keeps its habits: one parameterized `train()` function, every experiment a *config change*, every run logged to the same `runs/` tree for overlay in TensorBoard, comparisons on the right metric, and a control for every claim.
>
> **Why new files instead of reusing Lab 6's `train.py`?** Lab 6's models take a flat `(batch, features)` input. Sequence models take `(batch, time, features)` and need gradient clipping, so this lab introduces `seq_dataset.py`, `seq_model.py` and `seq_train.py`, built in the same style. Nothing in Lab 6's files is modified.

## 📖 Background: RNN, LSTM and GRU in a Nutshell

The MLPs of Lab 6 take one fixed-size vector and have no notion of *order*. A time series is different: what happened yesterday should shape how you read today. **Recurrent networks** handle this by reading the sequence **one step at a time**, carrying a **hidden state** `h_t` (a running memory) from each step to the next, and reusing the **same weights** at every step.

**RNN (vanilla recurrent network).** The simplest version blends the new input with the previous memory:

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

The weakness is memory. Training unrolls the loop through time, so the gradient is multiplied by `W_h` (and the slope of `tanh`) once per step. Over many steps it either shrinks towards zero (**vanishing gradient**, the network forgets distant inputs) or blows up (**exploding gradient**). In practice a vanilla RNN remembers roughly the last 10 to 20 steps.

**LSTM (Long Short-Term Memory).** Adds a second, protected memory called the **cell state** `c_t`, a "conveyor belt" that information can ride along with little interference, controlled by three **gates** (sigmoid units that output values between 0 and 1, acting as dials):

```
forget gate f : how much of the old memory to keep
input gate  i : how much of the new candidate g to write
output gate o : how much of the memory to expose as h_t

c_t = f ⊙ c_{t-1} + i ⊙ g          (additive update: the gradient highway)
h_t = o ⊙ tanh(c_t)
```

Because the cell state is updated by *adding* rather than repeatedly squashing, gradients can travel across many steps, so the LSTM can learn long-range dependencies.

**GRU (Gated Recurrent Unit).** A streamlined LSTM with no separate cell state and only two gates: an **update gate** `z` (how much of the old state to keep versus replace) and a **reset gate** `r` (how much of the past to use when proposing the new state):

```
h_t = (1 − z) ⊙ n + z ⊙ h_{t-1},     n = tanh(W · x_t + r ⊙ (W_h · h_{t-1}))
```

It has about **¾ of the LSTM's parameters** and often performs about as well, so it is a common default when you want a gated model that trains faster.

| | RNN | LSTM | GRU |
|---|---|---|---|
| Memory carried | `h` | `h` and cell state `c` | `h` |
| Gates | none | 3 (forget, input, output) | 2 (update, reset) |
| Parameters (same width) | 1× | ~4× | ~3× |
| Long-range memory | weak | strong | strong |
| Typical use | teaching, very short sequences | long dependencies | LSTM alternative, lighter |

> 🔗 **Why this matters for stocks.** Market volatility *clusters*: a turbulent week tends to be followed by more turbulence, and the effect can fade slowly over weeks. A model that can hold on to recent turbulence has a chance of forecasting it, and the gated cells are built to do exactly that. Whether they actually beat a simple "recent volatility" rule on real data is what this lab measures. In PyTorch, these are `nn.RNN`, `nn.LSTM` and `nn.GRU`; Section 2 builds a configurable wrapper around them and an LSTM from scratch.

---

### The key idea of this lab

Stock **prices** look wonderfully predictable (tomorrow's price is almost exactly today's), but that is an illusion. Day-to-day **returns** are close to unpredictable. What *is* predictable is **volatility**, how large the moves will be: volatile days tend to follow volatile days (*volatility clustering*). So the lab studies three targets on the same data:

| Target | Question | Expected difficulty |
|---|---|---|
| **return** | What will tomorrow's log-return be? | Near-impossible; expect skill ≈ 0 |
| **direction** | Will tomorrow close up or down? | Near-impossible; expect ≈ majority-class accuracy |
| **volatility** | How large will the next 5 days' moves be? | Genuinely forecastable; the place recurrent models can help |

**Report what you measure.** A model that honestly shows no skill on returns is a *correct* result; one that shows 70% accuracy on direction almost certainly has a leak.

---

## 0️⃣ Prerequisites

- The **`cv-dl`** conda environment and its Jupyter kernel from Lab 6-1, plus two small additions:

```bash
conda activate cv-dl
pip install yfinance pandas
cd <your lab7 folder>
```

### Check you're ready

```python
import sys, torch, pandas
print("python:", sys.executable)          # must contain 'cv-dl'
print("torch :", torch.__version__, "| pandas:", pandas.__version__)
import yfinance
print("yfinance imported OK")
```

Everything runs on a CPU in seconds to a couple of minutes per run. **No internet or yfinance trouble?** `seq_dataset.py` below falls back to a synthetic stock generator, so you can still complete every section (the exercise asks you to say which data you used).

Start TensorBoard before training anything:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

or in a second terminal: `tensorboard --logdir runs`, then open `http://localhost:6006`.

---

## 1️⃣ Stock Data: Download, Returns, Windows and Leakage

### 1.1 Downloading prices

`yfinance` pulls daily data from Yahoo Finance (for research and education). The loader below downloads once, **caches to CSV** so later runs are instant and reproducible, and keeps only the two columns we need: `Close` (adjusted for splits and dividends via `auto_adjust=True`) and `Volume`.

> 💡 **Why cache?** Yahoo data is revised and rate-limited. A cached CSV means your results do not change under you between runs. Record the **ticker, start date and download date** in your report.

> ⚠️ **Always use adjusted prices.** An unadjusted series shows a fake 50% "crash" on the day of a 2-for-1 split, which a model will happily learn as a pattern.

Tickers to try: `AAPL`, `MSFT`, `^GSPC` (S&P 500 index), `^NSEI` (Nifty 50 index), `RELIANCE.NS`, `TCS.NS`. Indian NSE stocks use the `.NS` suffix.

### 1.2 Prices vs returns, and the persistence illusion

Let `P_t` be the closing price. The **log-return** is

```
r_t = log(P_t / P_{t-1})
```

Returns are roughly stationary (their mean and spread do not drift the way a price level does), which is what a neural network trained on one period and tested on another needs. Prices are not: a model scaled with training-period statistics sees test-period prices *outside the range it ever saw* (the same extrapolation problem as a trended series).

Try this once, because it is the most common beginner mistake in financial ML:

```python
import numpy as np
from seq_dataset import load_prices

df = load_prices("AAPL", start="2015-01-01")
close = df["Close"].to_numpy()

# "Forecast" tomorrow's PRICE as today's price. No learning at all.
mse = np.mean((close[1:] - close[:-1]) ** 2)
r2 = 1 - mse / np.var(close[1:])
print(f"R^2 of 'tomorrow = today' on PRICE LEVELS: {r2:.4f}")
```

You will get an R² very close to 1. That does *not* mean prices are predictable: the series is so smooth relative to its overall range that the trivial rule scores almost perfectly. **A forecast of a price level has to be judged against this trivial rule, not against zero.** Working with returns removes the illusion: the equivalent "always predict zero return" baseline has R² = 0 by definition, and anything a model adds is genuine.

### 1.3 Four leakage traps (all four appear in published, wrong results)

> ⚠️ **Trap 1: random splits.** Neighbouring windows overlap almost completely (29 of 30 days). Shuffle-and-split puts near-copies of validation windows in training. **Split the raw series chronologically first, then window each part separately.**

> ⚠️ **Trap 2: scaler leakage.** Standardisation has learned parameters, so they must be fitted on the **training segment only** (the Lab 4 / Lab 6 lesson again). Here we scale by the training standard deviation and do **not** subtract the mean, so that "zero return" stays exactly zero.

> ⚠️ **Trap 3: target look-ahead at split boundaries.** A volatility target uses the *next 5 days*. The last training day's target would peek 5 days into the validation period. We therefore compute targets **inside each segment** and drop the final rows that lack a full horizon.

> ⚠️ **Trap 4: adjusted prices and "today's close".** The close on day `t` is known only at the end of day `t`. A feature for day `t` may use day `t`'s close; the target must be strictly *after* it. Off-by-one errors here silently produce spectacular, fake accuracy.

### 1.4 `seq_dataset.py`

```python
"""seq_dataset.py - stock-market windows, a synthetic fallback, and a memory task."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------ generic
class TensorSeqDataset(Dataset):
    """Wraps pre-built tensors. features: (N, T, F); targets: (N,) or (N, 1)."""

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.input_size = features.shape[2]
        self.seq_len = features.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ------------------------------------------------------------- data loading
def make_synthetic_stock(n=2500, seed=0, start="2015-01-01"):
    """GARCH(1,1) returns: UNPREDICTABLE direction, PREDICTABLE volatility.

    Fallback when the download fails, and a ground-truth lab bench: by
    construction there is no return signal to find, but volatility clusters.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90        # long-run daily vol = 1%
    var = omega / (1 - alpha - beta)
    r = np.zeros(n)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    close = 100 * np.exp(np.cumsum(r))
    volume = 1e6 * (1 + 20 * np.abs(r)) * np.exp(0.2 * rng.standard_normal(n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def load_prices(ticker="AAPL", start="2015-01-01", end=None, cache_dir="./data"):
    """Daily adjusted Close + Volume. Downloads once, then reads the cached CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    safe = ticker.replace("^", "").replace(".", "_")
    path = os.path.join(cache_dir, f"{safe}_{start}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("empty download (bad ticker, or no network?)")
        if isinstance(raw.columns, pd.MultiIndex):       # newer yfinance versions
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close", "Volume"]].dropna()
        df.to_csv(path)                                  # cache REAL data only
        print(f"[load_prices] downloaded {ticker}: {len(df)} rows, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as exc:
        print(f"[load_prices] WARNING: download failed for {ticker} "
              f"({type(exc).__name__}: {exc}).")
        print("[load_prices] Using SYNTHETIC data instead - say so in your report.")
        return make_synthetic_stock(start=start)        # not cached on purpose


# ----------------------------------------------------- targets and windowing
def _segment_targets(r, target, h_vol):
    """Targets for ONE segment, computed inside it (no peeking across splits).

    r[t] is the log-return of day t. A window ending at day t predicts:
      return    : r[t+1]
      direction : 1 if r[t+1] > 0 else 0
      volatility: log of the REALIZED volatility of r[t+1 .. t+h_vol],
                  i.e. the root-mean-square of those returns.
    The last rows without a full horizon get NaN and are dropped later.

    Why RMS and not std? Realized volatility is conventionally computed
    WITHOUT subtracting the mean (the daily mean return is ~0, and
    demeaning a handful of samples only adds noise). It also stays well
    defined at h_vol = 1, where it reduces to log|r[t+1]|; std() of a
    single sample is exactly 0, which would make every target the
    constant log(1e-8) = -18.42.
    """
    n = len(r)
    y = np.full(n, np.nan)
    if target in ("return", "direction"):
        nxt = r[1:]
        y[:-1] = nxt if target == "return" else (nxt > 0).astype(float)
    elif target == "volatility":
        for t in range(n - h_vol):
            seg = r[t + 1: t + 1 + h_vol]
            y[t] = np.log(np.sqrt((seg ** 2).mean()) + 1e-8)
    else:
        raise ValueError(f"unknown target {target!r}")
    return y


def _windows(feats, y, window):
    X, Y = [], []
    for end in range(window - 1, len(y)):
        if np.isnan(y[end]):
            continue
        X.append(feats[end - window + 1: end + 1])
        Y.append(y[end])
    return np.stack(X).astype(np.float32), np.asarray(Y)


def build_stock_datasets(prices, target="volatility", window=30, h_vol=5,
                         fracs=(0.7, 0.15, 0.15)):
    """Chronological split FIRST -> targets inside each segment -> scaler on TRAIN
    only -> windows. Returns (train_ds, val_ds, test_ds, scaler).

    Features per day (F = 2): log-return and change in log-volume, each divided
    by its TRAIN-segment std (no mean subtraction) and clipped to +/-8 for the
    heavy tails typical of markets.
    """
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float).replace(0, np.nan)
    frame = pd.DataFrame({"ret": np.log(close).diff(), "dvol": np.log(volume).diff()})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(frame)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    segs = [frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]]

    ret_std = float(segs[0]["ret"].std())                # fitted on TRAIN only
    dvol_std = float(segs[0]["dvol"].std())
    scale = np.array([ret_std, dvol_std])

    raw_targets = [_segment_targets(s["ret"].to_numpy(), target, h_vol) for s in segs]

    t_mean, t_std = 0.0, 1.0
    if target == "return":
        t_std = ret_std                                  # keeps zero return at zero
    elif target == "volatility":
        tr = raw_targets[0][~np.isnan(raw_targets[0])]
        t_mean, t_std = float(tr.mean()), float(tr.std())

    datasets = []
    for seg, y in zip(segs, raw_targets):
        feats = np.clip(seg[["ret", "dvol"]].to_numpy() / scale, -8, 8)
        if target != "direction":
            y = (y - t_mean) / t_std
        X, Y = _windows(feats, y, window)
        if target == "direction":
            Y = torch.tensor(Y, dtype=torch.long)
        else:
            Y = torch.tensor(Y, dtype=torch.float32)[:, None]
        datasets.append(TensorSeqDataset(torch.tensor(X), Y))

    scaler = dict(target=target, window=window, h_vol=h_vol, ret_std=ret_std,
                  dvol_std=dvol_std, t_mean=t_mean, t_std=t_std)
    return (*datasets, scaler)


# ----------------------------------------------------------------- baselines
def baseline_scores(test_ds, scaler, train_ds=None):
    """The bar a model must clear. Regression numbers are MSE on the SCALED target.

    return    : zero_mse (always predict no change), prev_return_mse
    direction : majority_acc (always predict the train majority class), prev_sign_acc
    volatility: train_mean_mse (predict the average), recent_vol_mse (persistence:
                'the next 5 days will be as volatile as the last 5')
    """
    target, X, y = scaler["target"], test_ds.features, test_ds.targets
    out = {}
    if target == "return":
        out["zero_mse"] = (y ** 2).mean().item()
        out["prev_return_mse"] = ((X[:, -1, 0:1] - y) ** 2).mean().item()
    elif target == "direction":
        majority = int(train_ds.targets.float().mean() >= 0.5)
        out["majority_acc"] = (y == majority).float().mean().item()
        out["prev_sign_acc"] = ((X[:, -1, 0] > 0).long() == y).float().mean().item()
    else:
        h = scaler["h_vol"]
        recent = X[:, -h:, 0] * scaler["ret_std"]        # raw returns, last h days
        # RMS, to match the target definition in _segment_targets
        pred = (torch.log(recent.pow(2).mean(dim=1).sqrt() + 1e-8)
                - scaler["t_mean"]) / scaler["t_std"]
        out["train_mean_mse"] = (y ** 2).mean().item()   # predicting 0 on the scaled axis
        out["recent_vol_mse"] = ((pred[:, None] - y) ** 2).mean().item()
    return out


def skill(model_mse, baseline_mse):
    """Out-of-sample skill = 1 - MSE_model / MSE_baseline. 0 = no better, <0 = worse."""
    return 1.0 - model_mse / baseline_mse


# ------------------------------------------------------ walk-forward folds
def walk_forward(prices, n_folds=4, val_len=250, test_len=250):
    """Yield (fold, prices_up_to_fold_end, fracs) for expanding-window evaluation.

    Each fold trains on everything before its validation block, validates on the
    next `val_len` days and tests on the `test_len` days after that. Pass the
    yielded prices and fracs straight to build_stock_datasets(). (Boundaries are
    approximate because 1 row is lost to differencing.)
    """
    n = len(prices)
    for k in range(n_folds):
        end = n - (n_folds - 1 - k) * test_len
        fracs = ((end - val_len - test_len) / end, val_len / end, test_len / end)
        yield k, prices.iloc[:end], fracs


# ------------------------------------------------- memory task (Section 5.2)
def make_first_token_task(n, T, k=2, seed=0):
    """The label is hidden in the FIRST time step; the other T-1 steps are noise.

    Channel 0 is pure noise at every step. Channel 1 is zero except at t=0, where
    it encodes the class. To classify correctly the network must carry that one
    signal across T-1 steps of distraction. Chance accuracy = 1/k.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, k, (n,), generator=g)
    x = torch.randn(n, T, 2, generator=g)
    x[:, :, 1] = 0.0
    x[:, 0, 1] = labels.float() - (k - 1) / 2
    return TensorSeqDataset(x, labels)
```

### Task

1. Download `AAPL` and plot `Close` and the log-return series side by side. Which looks stationary?
2. Reproduce the **persistence illusion** from 1.2 for two tickers. Report the price-level R² and explain in two sentences why it is meaningless.
3. **Volatility clustering.** Compute the autocorrelation at lags 1 to 20 of (a) `r_t` and (b) `|r_t|` and plot both. Which one has visible structure, and what does that tell you about which target is forecastable?

   ```python
   def acf(x, k):
       return np.corrcoef(x[:-k], x[k:])[0, 1]
   ```
4. Build the three datasets with `build_stock_datasets(df, target=...)` for `return`, `direction` and `volatility`. Report the shapes of `train_ds.features` and `train_ds.targets` for each, and the number of windows in train, val and test.
5. **Demonstrate Trap 1.** After Section 3 gives you `train()`, train the same small model on the volatility task with (a) the chronological split and (b) a random 70/15/15 shuffle of *all* windows. Compare validation MSE and explain the gap.
6. Compute `baseline_scores` for each target. **Predict first**: will "recent volatility" beat "predict the average" for the volatility target, and by roughly how much? Then measure.

---

## 2️⃣ The Models: RNN, LSTM, GRU

### What a recurrent layer does

At each time step `t`, the layer combines the current input `x_t` with a **hidden state** `h_{t-1}` that summarises everything seen so far:

| Cell | State carried | Update (simplified) | Weight blocks |
|---|---|---|---|
| **RNN** | `h` | `h_t = tanh(W_x x_t + W_h h_{t-1} + b)` | 1 |
| **GRU** | `h` | reset gate `r`, update gate `z`; `h_t = (1−z)·n + z·h_{t-1}` | 3 |
| **LSTM** | `h` and cell state `c` | input `i`, forget `f`, output `o` gates and candidate `g`; `c_t = f·c_{t-1} + i·g`; `h_t = o·tanh(c_t)` | 4 |

The vanilla RNN **multiplies** its state through a squashing non-linearity at every step, so information and gradients decay (or explode) exponentially with distance. The LSTM's cell state is updated **additively** (`c_t = f·c_{t-1} + i·g`), giving gradients a path that is not squashed at every step. Section 5.2 measures this instead of asking you to take it on trust.

### Output shapes (they cause most bugs)

For `nn.LSTM(..., batch_first=True)` with input `(B, T, F)`:

| Output | Shape |
|---|---|
| `out` | `(B, T, H · D)` — hidden state at **every** step, last layer only |
| `h_n` | `(L · D, B, H)` — **final** hidden state of every layer and direction |
| `c_n` (LSTM only) | `(L · D, B, H)` |

where `H` is the hidden size, `L` the number of layers and `D` is 2 if bidirectional else 1. `batch_first=True` affects `out` but **not** `h_n`, which is always layers-first.

> ⚠️ **A classic bidirectional bug.** With `bidirectional=True`, `out[:, -1]` is **not** the "final" state of the backward direction: that pass starts at the last element and ends at the first, so at index `-1` it has seen a single input. Use `h_n` (forward final from `h_n[-2]`, backward final from `h_n[-1]`). The model below does this.

```python
"""seq_model.py - RNN / LSTM / GRU wrapped in one configurable class."""

import torch
from torch import nn


class SeqModel(nn.Module):
    """(B, T, F) -> (B, out_features).

    cell          : "rnn" | "lstm" | "gru"
    pooling       : "last" (final hidden state h_n) | "mean" (mean of out over time)
    out_features  : 1 for regression, num_classes for classification
    Returns raw outputs (logits for classification). Softmax belongs at display
    time, as in Lab 6-1 - never inside forward().
    """

    def __init__(self, cell="lstm", input_size=1, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.0, out_features=1,
                 pooling="last", nonlinearity="tanh"):
        super().__init__()
        classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        kwargs = dict(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
            # nn.* applies this BETWEEN layers only; with 1 layer PyTorch would
            # warn and ignore it, so we zero it ourselves.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if cell == "rnn":
            kwargs["nonlinearity"] = nonlinearity
        self.cell = cell
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.rnn = classes[cell](**kwargs)
        self.drop = nn.Dropout(dropout)          # dropout before the output layer
        width = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(width, out_features)

    def forward(self, x):
        out, hn = self.rnn(x)
        if isinstance(hn, tuple):                # LSTM returns (h_n, c_n)
            hn = hn[0]
        if self.pooling == "last":
            summary = torch.cat([hn[-2], hn[-1]], dim=1) if self.bidirectional else hn[-1]
        elif self.pooling == "mean":
            summary = out.mean(dim=1)
        else:
            raise ValueError(f"unknown pooling {self.pooling!r}")
        return self.head(self.drop(summary))


def init_forget_bias(model, value=1.0):
    """Set the LSTM forget-gate bias to `value` (a common trick: start by remembering).

    PyTorch packs LSTM gates in the order [input, forget, cell, output], so the
    forget gate is the second quarter of each bias vector.
    """
    H = model.rnn.hidden_size
    with torch.no_grad():
        for name, p in model.rnn.named_parameters():
            if name.startswith("bias_ih"):
                p[H:2 * H].fill_(value)
            elif name.startswith("bias_hh"):
                p[H:2 * H].zero_()     # keep the total forget bias equal to `value`


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- from scratch
class ManualLSTM(nn.Module):
    """A single-layer, unidirectional LSTM written out gate by gate.

    Parameter layout deliberately mirrors nn.LSTM (gate order i, f, g, o), so
    you can copy weights across and check the outputs agree.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.H = hidden_size
        k = 1.0 / hidden_size ** 0.5
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))

    def forward(self, x):                        # x: (B, T, F)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.H)
        c = x.new_zeros(B, self.H)
        outputs = []
        for t in range(T):
            gates = (x[:, t] @ self.weight_ih.T + self.bias_ih
                     + h @ self.weight_hh.T + self.bias_hh)
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g                    # additive cell-state update
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)
```

### Task

1. **Parameter counting.** For one layer, unidirectional, the count is `g · (H·I + H·H + 2H)` where `I` is the input size, `H` the hidden size and `g` is 1, 3 or 4 for RNN, GRU and LSTM. Verify against `count_params` for `H = 32, I = 2` (your stock input size). Then work out what bidirectional and a second layer do to the count.
2. **Verify your understanding of the gates.** Build `nn.LSTM(2, 16, batch_first=True)` and a `ManualLSTM(2, 16)`. Copy the four parameter tensors across (e.g. `manual.weight_ih.data.copy_(ref.weight_ih_l0.data)`), run both on the same random input, and confirm with `torch.allclose(..., atol=1e-6)` that outputs and final states match.
3. Force the forget gate `f` to all zeros inside `ManualLSTM`, then to all ones. In one sentence each, what does the cell state do?
4. Why does the vanilla RNN's repeated multiplication cause trouble? Answer with numbers: compute `0.9**100` and `1.1**100`.

---

## 3️⃣ `seq_train.py`: One Parameterized Training Function

Same idea as Lab 6-1: one function configured by a dict, logging to TensorBoard, early stopping, best-checkpointing. The new ingredients are **gradient clipping** and a **gradient-norm log**.

| Config key | Meaning | Default |
|---|---|---|
| `run_name` | log/checkpoint folder name | required |
| `task` | `"regression"` or `"classification"` | required |
| `cell`, `input_size`, `out_features` | model basics | required |
| `hidden_size`, `num_layers`, `bidirectional`, `dropout`, `pooling` | architecture | 64, 1, False, 0.0, `"last"` |
| `forget_bias` | LSTM forget-gate bias init (`None` = PyTorch default) | `None` |
| `optimizer`, `lr`, `weight_decay` | optimisation | `"adam"`, `1e-3`, 0.0 |
| `clip` | max gradient norm (`None` = no clipping) | `None` |
| `batch_size`, `epochs`, `patience`, `seed` | loop control | 64, 30, 5, 0 |

```python
"""seq_train.py"""

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from seq_model import SeqModel, init_forget_bias, count_params


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, task, device):
    """Returns (loss, metric): metric is accuracy for classification, MAE for regression.

    Calls model.eval() itself - exactly like Lab 6-1's evaluate().
    """
    model.eval()
    total_loss = metric = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        bs = len(y)
        total_loss += criterion(out, y).item() * bs
        if task == "classification":
            metric += (out.argmax(dim=1) == y).sum().item()
        else:
            metric += (out - y).abs().mean(dim=1).sum().item()
        n += bs
    return total_loss / n, metric / n


def build_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def train(config, train_ds, val_ds, test_ds=None, extras=None):
    """`extras` (e.g. the scaler dict) is stored in best.pt for inference."""
    torch.manual_seed(config.get("seed", 0))
    device = get_device()
    task = config["task"]

    model = SeqModel(
        cell=config["cell"], input_size=config["input_size"],
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 1),
        bidirectional=config.get("bidirectional", False),
        dropout=config.get("dropout", 0.0),
        out_features=config["out_features"],
        pooling=config.get("pooling", "last"),
    ).to(device)
    if config.get("forget_bias") is not None and config["cell"] == "lstm":
        init_forget_bias(model, config["forget_bias"])

    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    optimizer = build_optimizer(config.get("optimizer", "adam"), model.parameters(),
                                config.get("lr", 1e-3), config.get("weight_decay", 0.0))

    bs = config.get("batch_size", 64)
    # Shuffling is safe HERE: the split was already chronological, and each
    # window is an independent (input, target) pair.
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512) if test_ds is not None else None

    run_name = config["run_name"]
    writer = SummaryWriter(os.path.join("runs", run_name))
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    # float("inf") makes clip_grad_norm_ a pure measurement (no clipping)
    max_norm = config["clip"] if config.get("clip") else float("inf")
    patience = config.get("patience", 5)

    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, config.get("epochs", 30) + 1):
        model.train()
        run_loss = run_gn = n_batches = n_seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            run_gn += float(gn)
            n_batches += 1

        train_loss = run_loss / n_seen
        val_loss, val_metric = evaluate(model, val_loader, criterion, task, device)
        mean_gn = run_gn / n_batches

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/val", val_metric, epoch)
        writer.add_scalar("grad_norm/mean", mean_gn, epoch)
        writer.add_scalar("generalization_gap", val_loss - train_loss, epoch)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_metric=val_metric, grad_norm=mean_gn))

        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_loss": val_loss, "val_metric": val_metric,
                        "extras": extras}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    train_time = time.perf_counter() - t_start
    stopped_epoch = epoch

    # Report TEST performance of the BEST checkpoint, not of the last epoch.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    result = dict(run_name=run_name, best_epoch=best_epoch, stopped_epoch=stopped_epoch,
                  best_val_loss=best_val, val_metric=ckpt["val_metric"],
                  train_time=train_time, n_params=count_params(model), history=history)
    if test_loader is not None:
        result["test_loss"], result["test_metric"] = evaluate(
            model, test_loader, criterion, task, device)

    writer.add_hparams({k: str(v) for k, v in config.items()},
                       {"hparam/val_loss": best_val,
                        "hparam/test_metric": result.get("test_metric", float("nan"))})
    writer.close()
    return result
```

> 💡 **The `task` key vs the stock `target`.** In `train()`'s config, `task` means `"regression"` or `"classification"` (how to compute the loss). In `build_stock_datasets`, `target` means `"return"`, `"direction"` or `"volatility"` (what to predict). Return and volatility are regression; direction is classification.

> 💡 **Regression losses are on the scaled target.** `test_loss` is MSE in standardised units, so it is directly comparable with the numbers from `baseline_scores`, which are in the same units. Never compare against baselines computed on a different scale.

---

## 4️⃣ Worked Example: LSTM on Apple (AAPL), Three Targets

This is a complete, runnable example for you to study, run and then adapt. It trains one LSTM per target on `AAPL`, and compares each to its baseline.

```python
"""run_stock_example.py - one LSTM, three targets, honest baselines."""

from seq_dataset import load_prices, build_stock_datasets, baseline_scores, skill
from seq_train import train

TICKER, START, WINDOW = "AAPL", "2015-01-01", 30

prices = load_prices(TICKER, start=START)
print(prices.tail(3))
print(f"{len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")

# stock target -> (how train() computes loss, number of outputs)
settings = {
    "volatility": dict(task="regression", out_features=1),
    "return":     dict(task="regression", out_features=1),
    "direction":  dict(task="classification", out_features=2),
}

for target, kind in settings.items():
    train_ds, val_ds, test_ds, scaler = build_stock_datasets(
        prices, target=target, window=WINDOW)
    base = baseline_scores(test_ds, scaler, train_ds)

    config = dict(
        run_name=f"stock_{TICKER}_{target}_lstm", cell="lstm", input_size=2,
        hidden_size=32, num_layers=1, dropout=0.0, optimizer="adam", lr=1e-3,
        weight_decay=1e-4, clip=1.0, batch_size=64, epochs=40, patience=8,
        window=WINDOW, **kind,
    )
    res = train(config, train_ds, val_ds, test_ds, extras=scaler)

    print(f"\n=== {target} ===  best epoch {res['best_epoch']} / stopped {res['stopped_epoch']}")
    print("baselines:", {k: round(v, 4) for k, v in base.items()})
    if target == "direction":
        print(f"LSTM test accuracy: {res['test_metric']:.4f}")
    else:
        ref = base["recent_vol_mse"] if target == "volatility" else base["zero_mse"]
        print(f"LSTM test MSE: {res['test_loss']:.4f}   skill vs "
              f"{'recent-vol' if target == 'volatility' else 'zero'} baseline: "
              f"{skill(res['test_loss'], ref):+.3f}")
```

### What to look for (and what *not* to expect)

- **return:** the LSTM's skill against the zero-return baseline should hover around **0**, and may be slightly **negative**. That is the honest answer, not a bug. A model with no real signal, trained on noise, often *loses* to "predict nothing" because it adds variance.
- **direction:** test accuracy should sit close to the **majority-class** accuracy (stocks drift upward, so "up" is the majority and typically wins slightly more than half the days). Beating the majority baseline by a point or two is within noise; check with three seeds before believing it.
- **volatility:** this is where something real exists. But read the two baselines carefully before celebrating. On AAPL, `recent_vol_mse` measures around **1.5** against `train_mean_mse` around **1.0** — that is, the `h`-day persistence estimate is *worse* than simply predicting the average, because the RMS of 5 returns is a very noisy estimator of volatility. So a large "skill vs recent-vol" number mostly says the opponent is weak. **Always report skill against both baselines**, and treat `train_mean` as the one that really has to be beaten. Whichever way it comes out, **report it**, with three seeds.
- **Compare epochs:** if `best_epoch` is 1 or 2 on the return task, the model is overfitting almost immediately, which is another symptom of "nothing to learn here".
- **Test period caveat:** your test set is one specific stretch of calendar time (one market regime). A result on a calm year may not carry to a volatile one. Section 5.6 (walk-forward) addresses this.

### Try-it-yourself checklist (5 minutes)

1. Change `TICKER` to `"^NSEI"` or `"RELIANCE.NS"`. Do the three conclusions above survive?
2. Set `cell="gru"` and `cell="rnn"`. Does the ranking change for volatility? For return?
3. Change `WINDOW` to `10` and `60`. Does a longer window help volatility?
4. Run the whole script on the synthetic series by setting `TICKER = "NO_SUCH_TICKER"` (the loader will fall back and warn you). The synthetic series has **no return signal by construction** and **strong volatility clustering**. Do your three conclusions get *clearer*? Why does a ground-truth series help you judge your own pipeline?

---

## 5️⃣ Experiments

Every experiment is a config change. Give each run a new `run_name` and overlay them in TensorBoard. Use the **volatility** target unless stated otherwise, since it is the one with signal.

### 5.1 RNN vs LSTM vs GRU

Run `cell ∈ {"rnn", "lstm", "gru"}` for each of the three targets on one ticker, with **three seeds** each (`seed ∈ {0, 1, 2}`).

> ⚠️ **Compare against the baselines, not against each other.** Market data is noisy, so differences between cells are often smaller than the difference you get from changing only `seed`. A cell that cannot beat the simple baseline has learned nothing a one-line rule does not give you.

**Task**

1. One table: target, cell, parameters, best epoch, test metric (mean ± std over seeds), and the baseline rows.
2. Is any gap between two cells larger than the seed-to-seed spread? If not, say so; "no detectable difference" is a valid, important result.
3. Convert each regression result to **skill** with `skill(model_mse, baseline_mse)`. Report it for volatility against both baselines.

### 5.2 Watching gradients vanish

Real markets are too noisy to isolate the vanishing-gradient mechanism cleanly, so this experiment uses a controlled synthetic task. The label is hidden at `t = 0` and the other `T − 1` steps are noise. A model succeeds only if information (and gradient) survives `T − 1` steps.

**Part A: train on increasing lengths.**

```python
import numpy as np

from seq_dataset import make_first_token_task
from seq_train import train

# patience == epochs ON PURPOSE: this task sits at chance for many epochs and
# then jumps suddenly. Any smaller patience early-stops during that plateau and
# records a failure that never happened.
mem = dict(task="classification", input_size=2, out_features=2, hidden_size=32,
           optimizer="adam", lr=1e-3, batch_size=64, epochs=40, patience=40)

# The fourth variant is the point of the experiment, not an afterthought: an
# LSTM whose forget gate starts at sigmoid(0) ~ 0.5 decays its cell state by
# about half per step, so at T=50 roughly 0.5**50 ~ 1e-15 of the signal
# survives. forget_bias=1.0 starts the gate near "remember" instead.
variants = [("rnn", {}), ("lstm", {}), ("gru", {}), ("lstm_fb1", {"forget_bias": 1.0})]
SEEDS = (0, 1, 2)      # single-seed results on this task are a coin flip - see below

for T in (10, 25, 50, 100, 200):
    tr = make_first_token_task(2000, T, seed=1)
    va = make_first_token_task(500, T, seed=2)
    te = make_first_token_task(500, T, seed=3)
    for name, extra in variants:
        cell = "lstm" if name.startswith("lstm") else name
        accs = []
        for seed in SEEDS:
            res = train({**mem, "run_name": f"mem_{name}_T{T}_s{seed}",
                         "cell": cell, "seed": seed, **extra}, tr, va, te)
            accs.append(res["test_metric"])
        print(f"T={T:3d}  {name:8s}  {np.mean(accs):.3f} +/- {np.std(accs):.3f}  "
              f"(runs: {[round(a, 3) for a in accs]})")
```

Chance is 0.5. Plot **mean test accuracy against `T`**, one line per variant, with error bars across seeds.

> ⚠️ **Expect a messy, non-monotonic picture, and do not tidy it up.** On this task a single run either latches the signal or never leaves chance, so the seed-to-seed spread is enormous — a cell can score 1.000 at one `T` and 0.52 at a *shorter* one. That is why three seeds are the minimum here, and why you should read the mean ± std rather than any single number.
>
> **Do not assume the LSTM will beat the RNN.** With PyTorch's default initialisation it often does not: a plain `lstm` can sit at chance where `rnn` still solves the task, because its forget gate starts at 0.5 and the signal dies before learning can begin. The variant that behaves the way the theory predicts is usually `lstm_fb1`. Whatever you measure, report it — including a result that contradicts the textbook ordering, with an explanation of *why* initialisation can dominate architecture here.

**Part B: measure the gradient directly.** At every time step, how much does the loss change if you nudge that step's input? That is `|∂loss/∂x_t|`.

```python
import torch
from torch import nn
from seq_model import SeqModel
from seq_dataset import make_first_token_task


def grad_per_timestep(model, ds, n=256):
    """Mean |d loss / d x_t| for each t, over n examples."""
    # NOTE: train mode on purpose. On a GPU, cuDNN's RNN backward raises an error
    # in eval mode. This is safe here because dropout=0 in this experiment.
    model.train()
    x = ds.features[:n].clone().requires_grad_(True)
    y = ds.targets[:n]
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    return x.grad.abs().mean(dim=(0, 2))          # shape (T,)


T = 100
ds = make_first_token_task(512, T, seed=1)
profiles = {}
for cell in ("rnn", "lstm", "gru"):
    torch.manual_seed(0)
    m = SeqModel(cell=cell, input_size=2, hidden_size=32, out_features=2)
    profiles[cell] = grad_per_timestep(m, ds)     # at INITIALISATION, untrained
```

Plot `profiles[cell]` against `t` with a **log-scaled y-axis**, all three cells on one figure.

> ⚠️ **Read this result carefully — these profiles are taken at *initialisation*, on an untrained model.** An untrained LSTM has its forget gate at `sigmoid(0) ≈ 0.5`, so its cell state decays by about half per step (`0.5**50 ≈ 1e-15`). At initialisation an LSTM can therefore decay **as fast as, or faster than**, a vanilla RNN, and the ratio you measure may not favour the LSTM at all.
>
> That is not a contradiction of the theory — it is the point. The LSTM's long-range advantage comes from **learned** gates (a forget gate driven towards 1), not from its starting state, which is exactly why `forget_bias=1.0` exists. **Profile a trained model as well as an untrained one and compare the two.** The contrast is the real result here, and it is worth more than either number on its own.

**Task**

1. Report the Part A table (mean ± std over three seeds) and the plot. For **each** variant, at what `T` does it stop reliably beating chance? Rank the four variants, and state whether the ranking you measured matches the one the theory would predict. **Report what you measure**; where each variant breaks depends on seed and learning rate, so do not quote a threshold from a textbook. If `rnn` outlasts a plain `lstm`, say so and explain it — the forget-gate initialisation argument above is the place to start.
2. For Part B, report `grad[T−1] / grad[0]` for each cell, **both at initialisation and after training**. Which cell carries gradient furthest back in each case, and by how many orders of magnitude? If the untrained ordering surprises you, explain it using the forget-gate value at initialisation.
3. The label is at `t = 0`, so the gradient that matters is the one reaching `x_0`. Explain via repeated multiplication by the recurrent Jacobian why that is the *smallest* gradient for the vanilla RNN.
4. **Link to markets.** Volatility clustering means yesterday's *and last month's* turbulence both inform tomorrow's. Your volatility models use `window=30`. Run the best cell at `window ∈ {5, 30, 90}` and report whether more history helps. Does this match what Part A led you to expect about long-range memory?
5. Using the `lstm` and `lstm_fb1` columns you already have, quantify what the forget-gate initialisation bought you: at which `T` do they first diverge, and by how much? Explain why starting the gate near "remember" helps, in terms of how much of the cell state survives `T` steps at `sigmoid(0) = 0.5` versus `sigmoid(1) ≈ 0.73`. (Stretch: push to `T = 400` and find where even `lstm_fb1` gives up.)

### 5.3 LSTM variants: depth, direction, pooling

Sweep, changing one factor at a time from the base `cell="lstm", num_layers=1, bidirectional=False, pooling="last"`, on the volatility target with three seeds:

| # | What varies | Values |
|---|---|---|
| 1 | `cell` | `rnn`, `lstm`, `gru` |
| 2 | `num_layers` | `1`, `2`, `3` |
| 3 | `bidirectional` | `False`, `True` |
| 4 | `pooling` | `last`, `mean` |
| 5 | `forget_bias` | `None`, `1.0` |

**Task**

1. One table with skill **and parameter count** for every run. Bigger models must justify their size.
2. For each factor, state which setting you would ship and why, in one sentence.
3. **When is bidirectional legitimate?** Reading a 30-day window in both directions is fine, because the whole window lies in the past. Reading *future* data backwards is leakage. State precisely where the line is, and give one real deployment where a bidirectional model would be invalid.
4. Deliberately break `pooling="last"` on a bidirectional model by using `out[:, -1]` instead of `h_n`, and report what happens to skill. This is the bug warned about in Section 2.

### 5.4 Gradient clipping

Recurrent gradients can also **explode**. Market data makes this a live concern: a crash day is a 6-to-10-standard-deviation event, and one such window can produce a gradient far larger than typical ones (which is why we clip features at ±8 as a first defence).

Take the memory task at `T = 50` with the `rnn` cell and an aggressive setting (`optimizer="sgd"`, `lr=1e-2`), and run `clip ∈ {None, 5.0, 1.0, 0.1}`. Then repeat the comparison `clip ∈ {None, 1.0}` on the real volatility target, using a window that spans a crash if your ticker's history has one (e.g. March 2020).

**Task**

1. In TensorBoard, open `grad_norm/mean` for all runs and include a screenshot. Which run shows spikes, and what does its validation loss do at the same epochs?
2. Too small a threshold is also a failure. What happened with `clip=0.1`, and why does clipping *rescale* the gradient (preserving direction) rather than capping each component?
3. `clip_grad_norm_` runs **after `backward()` and before `step()`**. Move it after `step()` in a copy of the loop, show it has no effect, and explain why.

### 5.5 Overfitting, dropout, and `eval()` again

Financial data is small and noisy, so overfitting is the default, not the exception.

**Task**

1. Build a deliberately small training set (the first 400 windows of the volatility training set) and sweep `dropout ∈ {0.0, 0.2, 0.5}` with `num_layers=2`, plus `weight_decay ∈ {0, 1e-4, 1e-2}`. Watch `generalization_gap` in TensorBoard. Which regulariser helps, if either?
2. Reproduce Lab 6-2's `eval()` experiment on this model: evaluate five times with `model.train()` active and five times after `model.eval()`, using a helper that does **not** call `evaluate()` (which would repair the bug silently). Report both sets of numbers.

> 💡 `nn.LSTM(dropout=p)` applies dropout **only between stacked layers**, never within a time step's recurrence and never on a single-layer model. `SeqModel` also applies `Dropout` before the output layer.

### 5.6 Walk-forward validation

A single train/test split gives you one number from one market regime. **Walk-forward validation** repeats the experiment on several consecutive periods, always training on the past and testing on the next block, which is how a model would actually be used.

```python
from seq_dataset import (load_prices, build_stock_datasets, baseline_scores,
                         skill, walk_forward)
from seq_train import train

prices = load_prices("AAPL", start="2015-01-01")
rows = []
for k, prices_k, fracs in walk_forward(prices, n_folds=4, val_len=250, test_len=250):
    tr, va, te, scaler = build_stock_datasets(
        prices_k, target="volatility", window=30, fracs=fracs)
    base = baseline_scores(te, scaler, tr)
    cfg = dict(run_name=f"wf_vol_lstm_fold{k}", task="regression", cell="lstm",
               input_size=2, out_features=1, hidden_size=32, lr=1e-3,
               weight_decay=1e-4, clip=1.0, epochs=40, patience=8)
    res = train(cfg, tr, va, te, extras=scaler)
    rows.append((k, skill(res["test_loss"], base["recent_vol_mse"]),
                 skill(res["test_loss"], base["train_mean_mse"])))
    print(rows[-1])
```

**Task**

1. Report skill for each fold against both baselines, with the mean and standard deviation across folds.
2. Does the model's skill stay positive in every fold, or does it depend on the period? If the best and worst fold differ a lot, what does that say about trusting a single test split?
3. Why must each fold's scaler be **refitted** on that fold's training data rather than reusing the first fold's?

---

## 6️⃣ Inference: Loading `best.pt`

As in every labsheet, the last step is loading the saved model and running it on one input. Here you download the **most recent** prices, build the latest window exactly as in training, and make a prediction.

```python
"""predict_stock.py - predict from the latest data using a saved best.pt"""

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from seq_model import SeqModel

CHECKPOINT_PATH = "checkpoints/stock_AAPL_volatility_lstm/best.pt"   # hardcoded - change
TICKER = "AAPL"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
config, scaler = ckpt["config"], ckpt["extras"]
window, target = scaler["window"], scaler["target"]
print(f"Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f}), target = {target}")

# Rebuild the SAME architecture the checkpoint was trained with.
model = SeqModel(
    cell=config["cell"], input_size=config["input_size"],
    hidden_size=config.get("hidden_size", 64),
    num_layers=config.get("num_layers", 1),
    bidirectional=config.get("bidirectional", False),
    dropout=config.get("dropout", 0.0),
    out_features=config["out_features"],
    pooling=config.get("pooling", "last"),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()                      # disables dropout - NOT optional

# ---- download recent data and preprocess EXACTLY as in training ----
raw = yf.download(TICKER, period="1y", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
close = raw["Close"].astype(float)
volume = raw["Volume"].astype(float).replace(0, np.nan)
frame = pd.DataFrame({"ret": np.log(close).diff(), "dvol": np.log(volume).diff()})
frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
if len(frame) < window:
    raise ValueError(f"need at least {window} days, got {len(frame)}")

scale = np.array([scaler["ret_std"], scaler["dvol_std"]])       # the TRAIN-fitted scale
feats = np.clip(frame[["ret", "dvol"]].to_numpy() / scale, -8, 8).astype(np.float32)
x = torch.tensor(feats[-window:][None], device=device)          # (1, window, 2)
print(f"Latest data point used: {frame.index[-1].date()}")

with torch.no_grad():
    out = model(x)

if target == "volatility":
    # undo the target scaling, then undo the log
    daily_vol = float(np.exp(out.item() * scaler["t_std"] + scaler["t_mean"]))
    print(f"Predicted daily volatility over the next {scaler['h_vol']} days: "
          f"{daily_vol * 100:.2f}%   (annualised ~ {daily_vol * np.sqrt(252) * 100:.1f}%)")
elif target == "direction":
    prob = torch.softmax(out, dim=1)[0]             # softmax at display time, not in forward()
    print(f"P(up tomorrow) = {prob[1]:.3f}   P(down) = {prob[0]:.3f}")
else:
    pred_ret = out.item() * scaler["t_std"]
    print(f"Predicted next-day log-return: {pred_ret * 100:.3f}%")
```

> 💡 **Two things to notice.** (1) The scaler lives in the checkpoint's `extras`; forgetting to apply it, or to *invert* the target scaling at the end, is the classic way to get a forecast that is "off by a constant factor". (2) This script needs the network at inference time. For an offline demo, replace the download with your cached CSV from `load_prices`.

---

## 🧪 Student Exercise (Evaluative) — Is Volatility Forecastable? A Multi-Ticker Study

Using the infrastructure above, test the central claim of this lab on **real data from more than one market**: *returns are not forecastable, volatility is, and a recurrent model's usefulness depends on how it compares with simple baselines.*

### Part 1 — Data and diagnostics

Choose **three tickers** from different markets or asset types (for example one US large-cap such as `AAPL` or `MSFT`; one index such as `^GSPC` or `^NSEI`; one Indian stock such as `RELIANCE.NS` or `TCS.NS`). For each, report ticker, date range, number of trading days and download date. Then, for each ticker:

1. The **persistence-illusion** R² on price levels (Section 1.2).
2. The autocorrelation of `r_t` versus `|r_t|` at lags 1–20 (Section 1 task 3).

### Part 2 — The grid (primary ticker)

On your **primary** ticker and the **volatility** target (`h_vol = 5`, `window = 30`), train every combination of:

- **Cell:** RNN, LSTM, GRU
- **Depth:** 1 layer, 2 layers
- **Direction:** unidirectional, bidirectional *(over the input window only)*

That is 12 configurations × 3 seeds = 36 runs. Each is a config change; if you are copy-pasting a training loop, reread Section 3. On your other two tickers, run only the **best configuration of each cell** (3 configs × 3 seeds each).

### Requirements

1. **Results table:** ticker, cell, layers, direction, hidden size, parameters, best epoch, **skill vs recent-vol baseline** and **skill vs train-mean baseline** (mean ± std over seeds), training time.
2. **Baselines in every table:** train-mean, recent-vol persistence, **and the Lab 6 flattened-window MLP** (an MLP on the 30×2 window flattened to 60 features). An LSTM that cannot beat the MLP is not earning its cost.
3. **A stronger finance baseline (choose one):** an **EWMA volatility** forecast, `σ²_t = λσ²_{t−1} + (1−λ)r²_t` with `λ = 0.94`, rolled forward on the test period, or a plain linear regression on lagged `|r|`. Implement it yourself, convert it to the same scaled target, and report its skill.
4. **Tune the learning rate and hidden size per cell**, at least coarsely, and say whether you matched hidden size or matched *parameter count* across cells (an LSTM has about 4× an RNN's parameters at equal width). Justify the choice.
5. Apply Section 3's mechanics consistently (early stopping, clipping at a stated threshold, whichever regularisation helped) and **say what you applied**.
6. **Overlay all runs in one TensorBoard** and include screenshots of the SCALARS and HPARAMS tabs, including `grad_norm/mean`.
7. **Walk-forward evaluation (Section 5.6)** of your best configuration on your primary ticker, 4 folds, reporting skill per fold and mean ± std.
8. **The negative controls.** For your primary ticker, also run your best cell on the **return** and **direction** targets. Report skill/accuracy against their baselines with three seeds. State plainly whether you found any skill.
9. **Horizon.** Repeat the best configuration with `h_vol ∈ {1, 5, 20}`. Does volatility become *easier* to predict at longer horizons, and how does that show in skill?

### Analysis — answer these specifically

- **Which targets are forecastable?** Rank return, direction and volatility by the skill you measured, and connect each to your autocorrelation plots from Part 1.
- **Does a recurrent model beat the simple baselines on volatility?** Compare against recent-vol, EWMA/linear and the MLP. If it does not, explain why a simple rule can be nearly optimal when the underlying process (like GARCH) is itself simple. If it does, say by how much and at what parameter count.
- **Separate two effects.** An LSTM has ~4× the parameters of an RNN of the same width. Design and run a comparison that tells you whether the LSTM's advantage (if any) comes from its **gating** or merely its **size**. (One idea among several: give the RNN a larger hidden size to match the LSTM's parameter count.)
- **Is bidirectional valid and useful here?** Justify validity explicitly (Section 5.3 task 3), then report whether it helped.
- **How much do you trust your result?** Discuss seed variance, the single-regime test set, and what the walk-forward folds showed. A clearly argued "no reliable improvement over the baseline" earns full marks; an unexplained "my LSTM wins" does not.
- **A direction-accuracy trap (short answer).** Suppose a direction model reaches 53% test accuracy. Give two reasons this does not imply a profitable trading strategy (hint: think about transaction costs and about *which* days the model gets right).

### Submission

Submit the `best.pt` for your best-performing volatility configuration, together with a `predict_stock.py` run (Section 6) on a ticker of your choice, showing the printed prediction **and the date of the latest data point used**.

### Optional stretch

Use your volatility forecast for something concrete: scale a hypothetical position size **inversely** with predicted volatility ("risk targeting") and compare the realised volatility of that position with a constant-size one on the test period. This is the most common legitimate use of volatility forecasts, and it does not need the model to predict direction. *(Illustration only; this is not trading advice.)*

### Some things to think about

- Look at the **gradient-norm curves** of your RNN runs, not just the metric. If one run shows spikes, did your clipping threshold tame them, and did its final skill suffer or benefit?
- Notice how much of this lab was about **evaluation discipline** (splits, scalers, baselines, seeds) rather than the architecture. In noisy domains the discipline is most of the work.
- A sequence model still processes a series **one step at a time**: step `t` cannot start until step `t−1` finishes. That makes it slow to train compared with architectures that look at all positions in parallel. Keep this in mind as the course moves on.

> 📌 No reference solution is included in this lab sheet.

---
