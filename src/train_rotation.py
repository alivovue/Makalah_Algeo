import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _to_float(x: str) -> float:
    return float(x.strip())


def ridge_fit(X, y, lam=1e-2, penalize_bias=False):
    n, k = X.shape
    XtX = X.T @ X
    I = np.eye(k, dtype=float)
    if not penalize_bias and k > 0:
        I[0, 0] = 0.0
    A = XtX + lam * I
    Xty = X.T @ y
    beta = np.linalg.solve(A, Xty)
    return beta


def read_rotation_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "match_id", "game_no", "team_name",
            "t_sec", "circle_phase",
            "pos_x", "pos_y",
            "circle_x", "circle_y", "circle_r",
        ]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for r in reader:
            rows.append({
                "match_id": r["match_id"].strip(),
                "game_no": int(r["game_no"]),
                "team_name": r["team_name"].strip(),
                "t_sec": int(float(r["t_sec"])),
                "circle_phase": int(float(r["circle_phase"])),
                "pos_x": _to_float(r["pos_x"]),
                "pos_y": _to_float(r["pos_y"]),
                "circle_x": _to_float(r["circle_x"]),
                "circle_y": _to_float(r["circle_y"]),
                "circle_r": _to_float(r["circle_r"]),
            })
    return rows


def compute_q(pos_x, pos_y, cx, cy, r):
    if r <= 1e-12:
        return None
    qx = (pos_x - cx) / r
    qy = (pos_y - cy) / r
    rho = math.sqrt(qx*qx + qy*qy)
    return qx, qy, rho


def make_feature_row(qx, qy, rho, dt_min, feature_names):           
    feat_map = {
        "bias": 1.0,
        "qx": qx,
        "qy": qy,
        "rho": rho,
        "dt_min": dt_min,
    }
    return np.array([feat_map[name] for name in feature_names], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to rotation CSV (normalized coords).")
    ap.add_argument("--out", required=True, help="Output JSON model path.")
    ap.add_argument("--lambda_", type=float, default=1e-2, help="Ridge regularization strength.")
    ap.add_argument("--min_team_pairs", type=int, default=5, help="Minimum (time-step) pairs to fit per-team model.")
    ap.add_argument("--penalize_bias", action="store_true", help="If set, ridge also penalizes bias term.")
    ap.add_argument("--features", default="bias,qx,qy,rho,dt_min",
                    help="Comma-separated features. Supported: bias,qx,qy,rho,dt_min")
    args = ap.parse_args()

    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    supported = {"bias", "qx", "qy", "rho", "dt_min"}
    bad = [f for f in feature_names if f not in supported]
    if bad:
        raise ValueError(f"Unsupported features: {bad}. Supported: {sorted(supported)}")

    rows = read_rotation_csv(args.csv)

    groups = defaultdict(list)
    for r in rows:
        key = (r["match_id"], r["game_no"], r["team_name"], r["circle_phase"])
        groups[key].append(r)

    per_team_samples = defaultdict(list)
    global_X, global_dqx, global_dqy = [], [], []
    dropped = 0
    n_pairs_total = 0

    for (match_id, game_no, team, phase), g in groups.items():
        g_sorted = sorted(g, key=lambda z: z["t_sec"])
        for i in range(len(g_sorted) - 1):
            a = g_sorted[i]
            b = g_sorted[i + 1]

            dt = b["t_sec"] - a["t_sec"]
            if dt <= 0:
                dropped += 1
                continue

            qa = compute_q(a["pos_x"], a["pos_y"], a["circle_x"], a["circle_y"], a["circle_r"])
            qb = compute_q(b["pos_x"], b["pos_y"], a["circle_x"], a["circle_y"], a["circle_r"])

            if qa is None or qb is None:
                dropped += 1
                continue

            qx, qy, rho = qa
            qx2, qy2, rho2 = qb

            dt_min = dt / 60.0
            xrow = make_feature_row(qx, qy, rho, dt_min, feature_names)

            dqx = qx2 - qx
            dqy = qy2 - qy

            per_team_samples[team].append((xrow, dqx, dqy))
            global_X.append(xrow)
            global_dqx.append(dqx)
            global_dqy.append(dqy)
            n_pairs_total += 1

    if len(global_X) == 0:
        raise RuntimeError("No valid rotation pairs to train on. Check CSV and timestamps.")

    Xg = np.vstack(global_X)
    yx = np.array(global_dqx, dtype=float)
    yy = np.array(global_dqy, dtype=float)

    beta_dqx_global = ridge_fit(Xg, yx, lam=args.lambda_, penalize_bias=args.penalize_bias)
    beta_dqy_global = ridge_fit(Xg, yy, lam=args.lambda_, penalize_bias=args.penalize_bias)

    team_models = {}
    for team, samples in per_team_samples.items():
        if len(samples) < args.min_team_pairs:
            continue
        Xt = np.vstack([s[0] for s in samples])
        ytx = np.array([s[1] for s in samples], dtype=float)
        yty = np.array([s[2] for s in samples], dtype=float)

        beta_x = ridge_fit(Xt, ytx, lam=args.lambda_, penalize_bias=args.penalize_bias)
        beta_y = ridge_fit(Xt, yty, lam=args.lambda_, penalize_bias=args.penalize_bias)

        team_models[team] = {
            "n_pairs": len(samples),
            "beta_dqx": beta_x.tolist(),
            "beta_dqy": beta_y.tolist(),
        }

    model = {
        "model_type": "rotation_ridge_dq",
        "coords": "normalized_0_1",
        "features": feature_names,
        "lambda": args.lambda_,
        "penalize_bias": bool(args.penalize_bias),
        "min_team_pairs": args.min_team_pairs,
        "dropped_pairs": dropped,
        "global": {
            "beta_dqx": beta_dqx_global.tolist(),
            "beta_dqy": beta_dqy_global.tolist(),
            "n_pairs": int(n_pairs_total),
        },
        "teams": team_models,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"[OK] Saved rotation model to: {out_path}")
    print(f"[INFO] Global pairs: {n_pairs_total} | Teams modeled: {len(team_models)} | Dropped pairs: {dropped}")


if __name__ == "__main__":
    main()
