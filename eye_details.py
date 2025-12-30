import cv2
import mediapipe as mp
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# # 結果保存用ディレクトリ
# RESULT_DIR = './webcam_results'
# os.makedirs(RESULT_DIR, exist_ok=True)
# 結果保存用ディレクトリ（リポ直下の webcam_results）
RESULT_DIR = os.path.join(os.path.dirname(__file__), "webcam_results")
os.makedirs(RESULT_DIR, exist_ok=True)

GRAPH_PATH = os.path.join(RESULT_DIR, 'blink_graph.png')
VIDEO_PATH = os.path.join(RESULT_DIR, 'blink_tracking.mp4')

mp_face_mesh = mp.solutions.face_mesh
# 動画用なので static_image_mode=False に変更
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 実測した左右目尻間の距離（mm）- ユーザーの実測値
MEASURED_EYE_WIDTH_MM = 105.0
REFERENCE_STICKER_MM = 15.0  # 青い円形シールの実寸直径（mm）
BLINK_THRESHOLD_MM = 4.0     # これを下回ると「閉じている」と判定

# --- 幾何学計算ヘルパー ---
def get_distance_point_to_line(point, line_start, line_end):
    """点と直線の距離を計算（上瞼・下瞼の垂直距離用）"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    if np.linalg.norm(line_vec) == 0: return 0
    cross_prod = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    return abs(cross_prod / np.linalg.norm(line_vec))

# --- シール検出・スケール計算 ---
def calculate_scale_for_frame(image, landmarks, method='eye'):
    h, w = image.shape[:2]
    lm = landmarks
    
    # 1. 目尻間の距離 (バックアップ用かつ標準用)
    p_left_eye = np.array([lm[33].x * w, lm[33].y * h])
    p_right_eye = np.array([lm[263].x * w, lm[263].y * h])
    eye_width_px = np.linalg.norm(p_left_eye - p_right_eye)
    
    mm_per_px = 0.0
    
    if method == 'sticker':
        # シール検出ロジック (既存コードを軽量化して統合)
        # 探索範囲をおでこに限定
        p10 = np.array([lm[10].x * w, lm[10].y * h])
        p9 = np.array([lm[9].x * w, lm[9].y * h])
        v_dist = np.linalg.norm(p10 - p9)
        if v_dist == 0: v_dist = 30
        
        cx, cy = int((p10[0] + p9[0]) / 2), int(min(p10[1], p9[1]))
        roi_half = int(v_dist * 2.5)
        x1, x2 = max(0, cx - roi_half), min(w, cx + roi_half)
        y1, y2 = max(0, cy - roi_half*2), min(h, cy + roi_half)
        
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 青色マスク
            mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_area = 0
            best_r = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area and area > 50:
                    _, r = cv2.minEnclosingCircle(cnt)
                    if r > 0:
                        max_area = area
                        best_r = r
            
            if best_r > 0:
                diameter_px = best_r * 2
                mm_per_px = REFERENCE_STICKER_MM / diameter_px
                # 検出できた場合、緑の丸を描画
                cv2.circle(image, (int(x1+cx), int(y1+cy)), int(best_r), (0,255,0), 2)
                return mm_per_px

    # シールが見つからない、またはeyeモードの場合は目尻基準
    if eye_width_px > 0:
        return MEASURED_EYE_WIDTH_MM / eye_width_px
    
    return 0.0

# --- 目の詳細解析（数値計算のみ） ---
def calculate_eye_metrics(landmarks, w, h, mm_per_px):
    lm = landmarks
    def get_pt(idx): return np.array([lm[idx].x * w, lm[idx].y * h])
    
    # 左右のランドマーク定義
    eyes_indices = {
        'left':  {'outer':33, 'inner':133, 'top':159, 'bottom':145},
        'right': {'outer':263, 'inner':362, 'top':386, 'bottom':374}
    }
    
    metrics = {}
    
    for side, idxs in eyes_indices.items():
        p_outer = get_pt(idxs['outer'])
        p_inner = get_pt(idxs['inner'])
        p_top = get_pt(idxs['top'])
        p_bottom = get_pt(idxs['bottom'])
        
        # 基準線（目尻-目頭）からの垂直距離
        top_dist_px = get_distance_point_to_line(p_top, p_outer, p_inner)
        bottom_dist_px = get_distance_point_to_line(p_bottom, p_outer, p_inner)
        # print(f"{side} eye - top_dist_px: {top_dist_px}, bottom_dist_px: {bottom_dist_px}")
        total_px = top_dist_px + bottom_dist_px

        # Clamp values to 0 if total_mm is below the blink threshold
        if total_px * mm_per_px < BLINK_THRESHOLD_MM:
            top_mm = 0.0
            bottom_mm = 0.0
            total_mm = 0.0
        else:
            top_mm = top_dist_px * mm_per_px
            bottom_mm = bottom_dist_px * mm_per_px
            total_mm = total_px * mm_per_px
        
        metrics[side] = {
            'top_mm': top_mm,
            'bottom_mm': bottom_mm,
            'total_mm': total_mm
        }
    
    return metrics


def draw_eye_tracking_overlay(frame, landmarks, w, h):
    """Draws simple markers to visualize eye landmark tracking."""
    lm = landmarks

    def pt(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    eyes = {
        'L': {'outer': 33, 'inner': 133, 'top': 159, 'bottom': 145},
        'R': {'outer': 263, 'inner': 362, 'top': 386, 'bottom': 374},
    }

    for side, idxs in eyes.items():
        p_outer = pt(idxs['outer'])
        p_inner = pt(idxs['inner'])
        p_top = pt(idxs['top'])
        p_bottom = pt(idxs['bottom'])

        # Corner-to-corner baseline
        cv2.line(frame, p_outer, p_inner, (0, 255, 255), 2)

        # Key points
        for p in (p_outer, p_inner, p_top, p_bottom):
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
            cv2.circle(frame, p, 4, (0, 0, 0), 1)

        # Optional label near the top lid point
        cv2.putText(frame, side, (p_top[0] + 6, p_top[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, side, (p_top[0] + 6, p_top[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# --- メイン：動画解析実行関数 ---
def run_blink_analysis_video(duration_sec=5.0, scale_method='eye'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    print("Press SPACE to start recording...")

    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Press SPACE to start recording", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Blink Analysis - Standby', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key
            first_frame = frame
            break
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return

    if first_frame is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    print(f"\n=== START RECORDING ({duration_sec}s) ===")
    print("数回まばたきをしてください...")

    # Prepare video writer (MP4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vh, vw = first_frame.shape[:2]
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, float(fps), (vw, vh))
    if not writer.isOpened():
        writer = None
        print(f"[WARN] Could not open video writer: {VIDEO_PATH}")
    else:
        print(f"Tracking video will be saved to: {VIDEO_PATH}")

    start_time = time.time()
    data_log = {
        'time': [],
        'l_total': [], 'l_top': [], 'l_bottom': [],
        'r_total': [], 'r_top': [], 'r_bottom': []
    }

    def process_and_record_frame(frame, elapsed):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Visualize tracking
            draw_eye_tracking_overlay(frame, landmarks, w, h)

            # 1. Scale (Dynamic Scaling)
            mm_per_px = calculate_scale_for_frame(frame, landmarks, scale_method)

            # 2. Eye metrics
            metrics = calculate_eye_metrics(landmarks, w, h, mm_per_px)

            # 3. Log
            data_log['time'].append(elapsed)
            l_m = metrics['left']
            r_m = metrics['right']
            data_log['l_total'].append(l_m['total_mm'])
            data_log['l_top'].append(l_m['top_mm'])
            data_log['l_bottom'].append(l_m['bottom_mm'])
            data_log['r_total'].append(r_m['total_mm'])
            data_log['r_top'].append(r_m['top_mm'])
            data_log['r_bottom'].append(r_m['bottom_mm'])

            # Overlay text
            l_state = f"{l_m['total_mm']:.1f}mm"
            r_state = f"{r_m['total_mm']:.1f}mm"
            cv2.putText(frame, f"Rec: {elapsed:.1f}/{duration_sec:.0f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"L: {l_state} R: {r_state}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if l_m['total_mm'] < BLINK_THRESHOLD_MM or r_m['total_mm'] < BLINK_THRESHOLD_MM:
                print(f"[BLINK DETECTED] Time: {elapsed:.2f}s | L: {l_state} / R: {r_state}")
        else:
            cv2.putText(frame, "No face", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)

        return frame

    start_time = time.time()

    # Record the first frame immediately
    first_processed = process_and_record_frame(first_frame, 0.0)
    cv2.imshow('Blink Analysis', first_processed)

    while True:
        elapsed = time.time() - start_time
        if elapsed > duration_sec:
            break

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        processed = process_and_record_frame(frame, elapsed)
        cv2.imshow('Blink Analysis', processed)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("=== RECORDING FINISHED ===")
    return data_log

# --- グラフ描画 ---
def plot_blink_results(data):
    if not data['time']:
        print("No data recorded.")
        return

    t = data['time']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Left Eye
    ax1.set_title("Left Eye Dynamics")
    ax1.plot(t, data['l_total'], label='Opening (Total)', color='blue', linewidth=2)
    ax1.plot(t, data['l_top'], label='Upper Lid', color='red', linestyle='--', alpha=0.7)
    ax1.plot(t, data['l_bottom'], label='Lower Lid', color='green', linestyle='--', alpha=0.7)
    ax1.set_ylabel("Opening [mm]")
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # Right Eye
    ax2.set_title("Right Eye Dynamics")
    ax2.plot(t, data['r_total'], label='Opening (Total)', color='blue', linewidth=2)
    ax2.plot(t, data['r_top'], label='Upper Lid', color='red', linestyle='--', alpha=0.7)
    ax2.plot(t, data['r_bottom'], label='Lower Lid', color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Opening [mm]")
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    print(f"Graph saved to: {GRAPH_PATH}")
    plt.show()

# --- 実行 ---
if __name__ == "__main__":
    # # モード選択
    # print("Select scale method:")
    # print("  [1] Eye width (Default)")
    # print("  [2] Sticker (Blue, 15mm)")
    # choice = input("Enter 1 or 2: ").strip()
    # method = 'sticker' if choice == '2' else 'eye'
    method = 'eye'  # 強制的に eye モードに固定

    # 解析開始
    logged_data = run_blink_analysis_video(duration_sec=5.0, scale_method=method)
    
    # グラフ表示
    plot_blink_results(logged_data)