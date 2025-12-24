import json
import math
from pathlib import Path

import numpy as np


def load_model_flexible(path: str):
    p = Path(path)

    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    script_dir = Path(__file__).resolve().parent
    p2 = script_dir / path
    if p2.exists():
        return json.loads(p2.read_text(encoding="utf-8"))

    p3 = script_dir.parent / path
    if p3.exists():
        return json.loads(p3.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Model not found. Tried:\n- {p}\n- {p2}\n- {p3}")


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def prompt_float(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  !! Please enter a valid number.")


def prompt_int(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("  !! Please enter a valid integer.")


def prompt_choice(prompt, choices, default=None):
    choices_lower = {c.lower(): c for c in choices}
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        key = s.lower()
        if key in choices_lower:
            return choices_lower[key]
        print(f"  !! Choose one of: {', '.join(choices)}")


def annotate_image(base_img_path: str, out_img_path: str, points, title=None):
    try:
        from PIL import Image, ImageDraw, ImageFont 

        img = Image.open(base_img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        if title:
            draw.text((10, 10), title, fill=(255, 255, 255, 220), font=font)

        for p in points:
            x = float(p["x"])
            y = float(p["y"])
            label = str(p.get("label", ""))
            color = p.get("color", (255, 0, 0))
            if len(color) == 3:
                color = (*color, 255)

            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
            draw.line((x - 14, y, x + 14, y), fill=color, width=3)
            draw.line((x, y - 14, x, y + 14), fill=color, width=3)

            if label:
                draw.text((x + 12, y + 12), label, fill=color, font=font)

        img.save(out_img_path)
        return out_img_path

    except Exception:
        try:
            import cv2 

            img = cv2.imread(base_img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {base_img_path}")

            if title:
                cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            for p in points:
                x = int(round(float(p["x"])))
                y = int(round(float(p["y"])))
                label = str(p.get("label", ""))
                color = p.get("color", (255, 0, 0)) 
                bgr = (int(color[2]), int(color[1]), int(color[0]))

                cv2.circle(img, (x, y), 10, bgr, 2)
                cv2.line(img, (x - 14, y), (x + 14, y), bgr, 2)
                cv2.line(img, (x, y - 14), (x, y + 14), bgr, 2)

                if label:
                    cv2.putText(img, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

            cv2.imwrite(out_img_path, img)
            return out_img_path
        except Exception as ex:
            raise RuntimeError(f"Failed to annotate image (Pillow+OpenCV both failed): {ex}")


def ask_coord_mode():
    mode = prompt_choice("Coordinate input mode? [1]=Pixels, [2]=Normalized : ", ["1", "2"], default="1")
    if mode == "1":
        W = prompt_int("  Map panel width W [1615]: ", default=1615)
        H = prompt_int("  Map panel height H [1620]: ", default=1620)
        return "px", W, H
    else:
        want_px_out = prompt_choice("Also show pixel outputs? [y/n] (default y): ", ["y", "n"], default="y")
        if want_px_out == "y":
            W = prompt_int("  Map panel width W [1615]: ", default=1615)
            H = prompt_int("  Map panel height H [1620]: ", default=1620)
            return "norm", W, H
        return "norm", None, None


def to_norm(x, y, mode, W, H):
    if mode == "norm":
        return float(x), float(y)
    if W is None or H is None:
        raise ValueError("W/H required for pixel->normalized conversion.")
    return float(x) / W, float(y) / H


def to_px(xn, yn, W, H):
    if W is None or H is None:
        return None
    return xn * W, yn * H


def compute_plane_basis(xs, ys, xe, ye):
    s = np.array([xs, ys], dtype=float)
    e = np.array([xe, ye], dtype=float)
    d = e - s
    dd = float(d @ d)
    if dd <= 1e-12:
        raise ValueError("Degenerate plane path: start and end are too close.")
    L = math.sqrt(dd)
    dhat = d / L
    n_hat = np.array([-dhat[1], dhat[0]], dtype=float)
    return s, e, d, float(dhat[0]), float(dhat[1]), float(L), n_hat


def make_landing_feature_row(dhat_x, dhat_y, L, feature_names):
    feat_map = {"bias": 1.0, "dhat_x": dhat_x, "dhat_y": dhat_y, "L": L}
    return np.array([feat_map[name] for name in feature_names], dtype=float)


def predict_tu(landing_model, team_name, xrow):
    teams = landing_model.get("teams", {})
    if team_name in teams:
        beta_t = np.array(teams[team_name]["beta_t"], dtype=float)
        beta_u = np.array(teams[team_name]["beta_u"], dtype=float)
        source = "team"
        n_samples = teams[team_name].get("n_samples", None)
    else:
        beta_t = np.array(landing_model["global"]["beta_t"], dtype=float)
        beta_u = np.array(landing_model["global"]["beta_u"], dtype=float)
        source = "global"
        n_samples = landing_model["global"].get("n_samples", None)

    t_hat = float(xrow @ beta_t)
    u_hat = float(xrow @ beta_u)
    return t_hat, u_hat, source, n_samples


def compute_q(pos_x, pos_y, cx, cy, r):
    if r <= 1e-12:
        raise ValueError("circle_r is too small / invalid.")
    qx = (pos_x - cx) / r
    qy = (pos_y - cy) / r
    rho = math.sqrt(qx * qx + qy * qy)
    return qx, qy, rho


def make_rotation_feature_row(qx, qy, rho, dt_min, feature_names):
    feat_map = {"bias": 1.0, "qx": qx, "qy": qy, "rho": rho, "dt_min": dt_min}
    return np.array([feat_map[name] for name in feature_names], dtype=float)


def predict_dq(rotation_model, team_name, xrow):
    teams = rotation_model.get("teams", {})
    if team_name in teams:
        beta_x = np.array(teams[team_name]["beta_dqx"], dtype=float)
        beta_y = np.array(teams[team_name]["beta_dqy"], dtype=float)
        source = "team"
        n_pairs = teams[team_name].get("n_pairs", None)
    else:
        beta_x = np.array(rotation_model["global"]["beta_dqx"], dtype=float)
        beta_y = np.array(rotation_model["global"]["beta_dqy"], dtype=float)
        source = "global"
        n_pairs = rotation_model["global"].get("n_pairs", None)

    dqx = float(xrow @ beta_x)
    dqy = float(xrow @ beta_y)
    return dqx, dqy, source, n_pairs


def landing_mode():
    print("\n=== LANDING MODE ===")
    print("Landing model was trained on NORMALIZED values, but you can input either pixels or normalized.\n")

    model_path = input("Landing model JSON path [model_landing.json]: ").strip() or "model_landing.json"
    model = load_model_flexible(model_path)
    if model.get("model_type") != "team_ridge_tu":
        raise ValueError("Landing model JSON has wrong model_type (expected team_ridge_tu).")

    feature_names = model["features"]
    teams_available = sorted(model.get("teams", {}).keys())

    print(f"\nLoaded landing model: {model_path}")
    print(f"Features: {feature_names}")
    print(f"Teams with specific models: {len(teams_available)} (fallback: global model)\n")

    mode, W, H = ask_coord_mode()

    print("\nSafety clamps (press Enter to use defaults):")
    tmin = prompt_float("  t_min [-0.2]: ", default=-0.2)
    tmax = prompt_float("  t_max [1.2]: ", default=1.2)
    ucap_in = input("  clamp |u| to U? (blank = no clamp) [e.g. 0.25]: ").strip()
    ucap = float(ucap_in) if ucap_in != "" else None

    want_pin = prompt_choice("\nGenerate pin image output? [y/n] (default n): ", ["y", "n"], default="n")
    base_img = None
    out_dir = None
    if want_pin == "y":
        if mode != "px":
            print("  NOTE: Pin output is drawn in PIXEL coordinates, so you should also provide W/H.")
            if W is None or H is None:
                W = prompt_int("  Map panel width W [1615]: ", default=1615)
                H = prompt_int("  Map panel height H [1620]: ", default=1620)

        base_img = input("  Base map image path (same coordinate system as your pixels): ").strip()
        out_dir = input("  Output folder for annotated images [./out]: ").strip() or "./out"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\nType 'exit' to go back. Type 'teams' to list known teams.\n")

    while True:
        team = input("Team name: ").strip()
        if team.lower() in ("exit", "quit", "q", "back"):
            break
        if team.lower() == "teams":
            print(", ".join(teams_available) if teams_available else "(No team-specific models.)")
            print()
            continue
        if team == "":
            print("  !! Team name cannot be empty.\n")
            continue

        if mode == "px":
            print("Plane start (PIXELS):")
            xs_in = prompt_float("  xs_px: ")
            ys_in = prompt_float("  ys_px: ")
            print("Plane end (PIXELS):")
            xe_in = prompt_float("  xe_px: ")
            ye_in = prompt_float("  ye_px: ")
        else:
            print("Plane start (NORMALIZED 0..1):")
            xs_in = prompt_float("  xs: ")
            ys_in = prompt_float("  ys: ")
            print("Plane end (NORMALIZED 0..1):")
            xe_in = prompt_float("  xe: ")
            ye_in = prompt_float("  ye: ")

        xs, ys = to_norm(xs_in, ys_in, mode, W, H)
        xe, ye = to_norm(xe_in, ye_in, mode, W, H)

        try:
            s, e, d, dhat_x, dhat_y, L, n_hat = compute_plane_basis(xs, ys, xe, ye)
        except ValueError as ex:
            print(f"  !! {ex}\n")
            continue

        xrow = make_landing_feature_row(dhat_x, dhat_y, L, feature_names)
        t_hat, u_hat, source, n_samples = predict_tu(model, team, xrow)

        t_hat = clamp(t_hat, tmin, tmax)
        if ucap is not None:
            u_hat = clamp(u_hat, -ucap, ucap)

        p_hat = s + t_hat * d + u_hat * n_hat
        x_norm, y_norm = float(p_hat[0]), float(p_hat[1])

        print("\nResult:")
        print(f"  model_used: {source}" + (f" (n_samples={n_samples})" if n_samples is not None else ""))
        print(f"  pred_t,u  : {t_hat:.6f}, {u_hat:.6f}")
        print(f"  landing   : x={x_norm:.6f}, y={y_norm:.6f} (NORMALIZED)")

        px = to_px(x_norm, y_norm, W, H)
        if px is not None:
            x_px, y_px = px
            print(f"  landing   : x={x_px:.2f}, y={y_px:.2f} (PIXELS @ {W}x{H})")

            if want_pin == "y" and base_img:
                out_path = str(Path(out_dir) / f"landing_{team}.png")
                try:
                    annotate_image(
                        base_img,
                        out_path,
                        points=[{"x": x_px, "y": y_px, "label": f"{team} LAND", "color": (255, 0, 0)}],
                        title="Landing Prediction",
                    )
                    print(f"  pin_image : {out_path}")
                except Exception as ex:
                    print(f"  !! Failed to write pin image: {ex}")

        print()


def rotation_mode():
    print("\n=== ROTATION MODE ===")
    print("You can input either pixels OR normalized.")
    print("IMPORTANT: pos and circle must be in the SAME unit system (both px OR both normalized).\n")

    model_path = input("Rotation model JSON path [model_rotation.json]: ").strip() or "model_rotation.json"
    model = load_model_flexible(model_path)
    if model.get("model_type") != "rotation_ridge_dq":
        raise ValueError("Rotation model JSON has wrong model_type (expected rotation_ridge_dq).")

    feature_names = model["features"]
    teams_available = sorted(model.get("teams", {}).keys())

    print(f"\nLoaded rotation model: {model_path}")
    print(f"Features: {feature_names}")
    print(f"Teams with specific models: {len(teams_available)} (fallback: global model)\n")

    mode, W, H = ask_coord_mode()

    print("\nOptional clamp for q (circle-relative) to avoid crazy outputs on tiny data.")
    qcap_in = input("Clamp |q| components to Q? (blank = no clamp) [e.g. 2.0]: ").strip()
    qcap = float(qcap_in) if qcap_in != "" else None

    want_pin = prompt_choice("\nGenerate pin image output? [y/n] (default n): ", ["y", "n"], default="n")
    base_img = None
    out_dir = None
    if want_pin == "y":
        if W is None or H is None:
            print("  NOTE: Pin output needs W/H to convert normalized->pixels.")
            W = prompt_int("  Map panel width W [1615]: ", default=1615)
            H = prompt_int("  Map panel height H [1620]: ", default=1620)

        base_img = input("  Base map image path (same coordinate system as your pixels): ").strip()
        out_dir = input("  Output folder for annotated images [./out]: ").strip() or "./out"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\nType 'exit' to go back. Type 'teams' to list known teams.\n")

    while True:
        team = input("Team name: ").strip()
        if team.lower() in ("exit", "quit", "q", "back"):
            break
        if team.lower() == "teams":
            print(", ".join(teams_available) if teams_available else "(No team-specific models.)")
            print()
            continue
        if team == "":
            print("  !! Team name cannot be empty.\n")
            continue

        if mode == "px":
            print("Current position (PIXELS):")
            pos_x = prompt_float("  pos_x_px: ")
            pos_y = prompt_float("  pos_y_px: ")
            print("Current circle (PIXELS):")
            cx = prompt_float("  circle_x_px: ")
            cy = prompt_float("  circle_y_px: ")
            cr = prompt_float("  circle_r_px: ")
        else:
            print("Current position (NORMALIZED 0..1):")
            pos_x = prompt_float("  pos_x: ")
            pos_y = prompt_float("  pos_y: ")
            print("Current circle (NORMALIZED 0..1):")
            cx = prompt_float("  circle_x: ")
            cy = prompt_float("  circle_y: ")
            cr = prompt_float("  circle_r: ")

        dt_sec = prompt_int("Predict forward by dt seconds [60]: ", default=60)
        if dt_sec <= 0:
            print("  !! dt must be positive.\n")
            continue

        try:
            qx, qy, rho = compute_q(pos_x, pos_y, cx, cy, cr)
        except ValueError as ex:
            print(f"  !! {ex}\n")
            continue

        dt_min = dt_sec / 60.0
        xrow = make_rotation_feature_row(qx, qy, rho, dt_min, feature_names)
        dqx, dqy, source, n_pairs = predict_dq(model, team, xrow)

        qx2 = qx + dqx
        qy2 = qy + dqy
        if qcap is not None:
            qx2 = clamp(qx2, -qcap, qcap)
            qy2 = clamp(qy2, -qcap, qcap)

        pred_x = cx + cr * qx2
        pred_y = cy + cr * qy2
        pred_rho = math.sqrt(qx2 * qx2 + qy2 * qy2)

        print("\nResult:")
        print(f"  model_used: {source}" + (f" (n_pairs={n_pairs})" if n_pairs is not None else ""))
        print(f"  q_now     : ({qx:.6f}, {qy:.6f}) | rho={rho:.6f}")
        print(f"  dq_pred   : ({dqx:.6f}, {dqy:.6f}) over dt={dt_sec}s")
        print(f"  q_next    : ({qx2:.6f}, {qy2:.6f}) | rho={pred_rho:.6f}")

        if mode == "px":
            print(f"  pos_next  : x={pred_x:.2f}, y={pred_y:.2f} (PIXELS)")
        else:
            print(f"  pos_next  : x={pred_x:.6f}, y={pred_y:.6f} (NORMALIZED)")

        print(f"  inside?   : {'YES' if pred_rho <= 1.0 else 'NO (outside circle)'}")

        if want_pin == "y" and base_img:
            if W is None or H is None:
                print("  !! Cannot draw pin without W/H.")
            else:
                if mode == "px":
                    now_px = (pos_x, pos_y)
                    next_px = (pred_x, pred_y)
                else:
                    now_px = to_px(pos_x, pos_y, W, H)
                    next_px = to_px(pred_x, pred_y, W, H)

                if now_px is not None and next_px is not None:
                    out_path = str(Path(out_dir) / f"rotation_{team}_{dt_sec}s.png")
                    try:
                        annotate_image(
                            base_img,
                            out_path,
                            points=[
                                {"x": now_px[0], "y": now_px[1], "label": f"{team} NOW", "color": (0, 255, 0)},
                                {"x": next_px[0], "y": next_px[1], "label": f"{team} NEXT", "color": (255, 0, 0)},
                            ],
                            title=f"Rotation Prediction (dt={dt_sec}s)",
                        )
                        print(f"  pin_image : {out_path}")
                    except Exception as ex:
                        print(f"  !! Failed to write pin image: {ex}")

        print()


def main():
    print("=== PUBG Predictor (Interactive) ===")
    print("You can choose coordinate input mode (pixels or normalized) per mode.\n")

    while True:
        mode = prompt_choice("Choose mode: [1]=Landing, [2]=Rotation, [q]=Quit : ", ["1", "2", "q"], default=None)
        if mode.lower() == "q":
            print("Bye.")
            break
        if mode == "1":
            landing_mode()
        elif mode == "2":
            rotation_mode()


if __name__ == "__main__":
    main()
