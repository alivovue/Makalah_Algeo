import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _to_float(x: str) -> float:
    return float(x.strip())


def compute_t_u(xs, ys, xe, ye, xl, yl):
    s = np.array([xs, ys], dtype=float)
    e = np.array([xe, ye], dtype=float)
    p = np.array([xl, yl], dtype=float)

    d = e - s
    dd = float(d @ d)
    if dd <= 1e-12:
        return None

    L = math.sqrt(dd)
    dhat = d / L
    n_hat = np.array([-dhat[1], dhat[0]], dtype=float)

    t = float(((p - s) @ d) / dd)
    u = float((p - s) @ n_hat)

    return t, u, float(dhat[0]), float(dhat[1]), float(L)


def make_feature_row(dhat_x, dhat_y, L, feature_names):
    feat_map = {
        "bias": 1.0,
        "dhat_x": dhat_x,
        "dhat_y": dhat_y,
        "L": L,
    }
    return np.array([feat_map[name] for name in feature_names], dtype=float)


def ridge_fit(X, y, lam=1e-2, penalize_bias=False):
    n, k = X.shape
    XtX = X.T @ X
    I = np.eye(k, dtype=float)
    if not penalize_bias:
        I[0, 0] = 0.0
    A = XtX + lam * I
    Xty = X.T @ y
    beta = np.linalg.solve(A, Xty)
    return beta


def read_landing_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "match_id", "game_no", "team_name",
            "plane_start_x", "plane_start_y",
            "plane_end_x", "plane_end_y",
            "landing_x", "landing_y",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for r in reader:
            rows.append({
                "match_id": r["match_id"].strip(),
                "game_no": int(r["game_no"]),
                "team_name": r["team_name"].strip(),
                "xs": _to_float(r["plane_start_x"]),
                "ys": _to_float(r["plane_start_y"]),
                "xe": _to_float(r["plane_end_x"]),
                "ye": _to_float(r["plane_end_y"]),
                "xl": _to_float(r["landing_x"]),
                "yl": _to_float(r["landing_y"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to landing CSV (normalized coords).")
    ap.add_argument("--out", required=True, help="Output JSON model path.")
    ap.add_argument("--lambda_", type=float, default=1e-2, help="Ridge regularization strength.")
    ap.add_argument("--min_team_samples", type=int, default=5, help="Minimum samples to fit per-team model.")
    ap.add_argument("--penalize_bias", action="store_true", help="If set, ridge also penalizes bias term.")
    ap.add_argument("--features", default="bias,dhat_x,dhat_y,L",
                    help="Comma-separated feature names. Supported: bias,dhat_x,dhat_y,L")
    args = ap.parse_args()

    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    supported = {"bias", "dhat_x", "dhat_y", "L"}
    bad = [f for f in feature_names if f not in supported]
    if bad:
        raise ValueError(f"Unsupported features: {bad}. Supported: {sorted(supported)}")

    rows = read_landing_csv(args.csv)

    per_team = defaultdict(list)
    global_X, global_t, global_u = [], [], []

    dropped = 0
    for r in rows:
        tu = compute_t_u(r["xs"], r["ys"], r["xe"], r["ye"], r["xl"], r["yl"])
        if tu is None:
            dropped += 1
            continue
        t, u, dhat_x, dhat_y, L = tu
        xrow = make_feature_row(dhat_x, dhat_y, L, feature_names)

        per_team[r["team_name"]].append((xrow, t, u))
        global_X.append(xrow)
        global_t.append(t)
        global_u.append(u)

    if len(global_X) == 0:
        raise RuntimeError("No valid rows to train on (check plane paths and CSV values).")

    Xg = np.vstack(global_X)
    tg = np.array(global_t, dtype=float)
    ug = np.array(global_u, dtype=float)

    beta_t_global = ridge_fit(Xg, tg, lam=args.lambda_, penalize_bias=args.penalize_bias)
    beta_u_global = ridge_fit(Xg, ug, lam=args.lambda_, penalize_bias=args.penalize_bias)

    team_models = {}
    for team, samples in per_team.items():
        if len(samples) < args.min_team_samples:
            continue
        Xt = np.vstack([s[0] for s in samples])
        yt = np.array([s[1] for s in samples], dtype=float)
        yu = np.array([s[2] for s in samples], dtype=float)

        beta_t = ridge_fit(Xt, yt, lam=args.lambda_, penalize_bias=args.penalize_bias)
        beta_u = ridge_fit(Xt, yu, lam=args.lambda_, penalize_bias=args.penalize_bias)

        team_models[team] = {
            "n_samples": len(samples),
            "beta_t": beta_t.tolist(),
            "beta_u": beta_u.tolist(),
        }

    model = {
        "model_type": "team_ridge_tu",
        "coords": "normalized_0_1",
        "features": feature_names,
        "lambda": args.lambda_,
        "penalize_bias": bool(args.penalize_bias),
        "min_team_samples": args.min_team_samples,
        "dropped_rows_due_to_bad_plane": dropped,
        "global": {
            "beta_t": beta_t_global.tolist(),
            "beta_u": beta_u_global.tolist(),
            "n_samples": int(len(global_X)),
        },
        "teams": team_models,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"[OK] Saved model to: {out_path}")
    print(f"[INFO] Global samples: {len(global_X)} | Teams modeled: {len(team_models)} | Dropped rows: {dropped}")


if __name__ == "__main__":
    main()
