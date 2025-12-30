import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

# ==========================================
# 設定エリア
# ==========================================
# 保存先ディレクトリ（必要に応じて変更してください）
RESULT_DIR = './webcam_results_mm' 
os.makedirs(RESULT_DIR, exist_ok=True)

# 【重要】基準となる実測値
# 一般的な成人の目尻〜目尻の幅はおよそ95〜105mmです。
REAL_EYE_WIDTH_MM = 105.0 

# 保存ファイル名
VIDEO_PATH = os.path.join(RESULT_DIR, 'smile_mm_analysis.mp4')
GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'smile_mm_components.png')
GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'smile_mm_trajectory.png')

# MediaPipe設定
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ランドマークID
ID_EYE_L_OUTER = 33
ID_EYE_R_OUTER = 263
ID_NOSE_BRIDGE = 168 
ID_MOUTH_L = 61
ID_MOUTH_R = 291

def get_coords(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

def rotate_point(point, pivot, angle_rad):
    """顔の傾き補正"""
    px, py = point
    cx, cy = pivot
    tx, ty = px - cx, py - cy
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    nx = tx * cos_a - ty * sin_a
    ny = tx * sin_a + ty * cos_a
    return np.array([nx + cx, ny + cy])

def record_and_analyze_mm_fixed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return

    # MacBookなど高解像度対応
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

    frames_data = [] 
    
    print(f"--- 口角追跡・mm計測モード (修正版) ---")
    print(f"基準: 目尻間の幅 = {REAL_EYE_WIDTH_MM}mm")
    print(" [SPACE]: 録画開始")
    print(" [ESC]: 終了")

    is_recording = False
    start_time = 0
    
    baseline_l = None
    baseline_r = None
    calibration_frames = []
    CALIBRATION_COUNT = 10 # キャリブレーションフレーム数を少し増やす

    while True:
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        display_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            p_eye_l = get_coords(lm, ID_EYE_L_OUTER, w, h)
            p_eye_r = get_coords(lm, ID_EYE_R_OUTER, w, h)
            p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
            p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
            p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

            # --- 1. スケール計算 (mm/px) ---
            eye_width_px = np.linalg.norm(p_eye_r - p_eye_l)
            if eye_width_px == 0: eye_width_px = 1
            mm_per_px = REAL_EYE_WIDTH_MM / eye_width_px

            # --- 2. 視覚的ガイド (目元のラインは削除し、口角のみ強調) ---
            # 口角に緑のマーカーを表示
            cv2.circle(display_image, tuple(p_mouth_l.astype(int)), 5, (0, 255, 0), -1) # 左口角
            cv2.circle(display_image, tuple(p_mouth_r.astype(int)), 5, (0, 255, 0), -1) # 右口角

            # 傾き補正用計算
            delta_eyes = p_eye_r - p_eye_l
            face_angle = np.arctan2(delta_eyes[1], delta_eyes[0])
            
            # 補正後の座標
            fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
            fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)

            if is_recording:
                elapsed = time.time() - start_time
                
                # --- キャリブレーション（真顔の基準位置を取得）---
                if len(calibration_frames) < CALIBRATION_COUNT:
                    calibration_frames.append({'L': fixed_mouth_l, 'R': fixed_mouth_r})
                    cv2.putText(display_image, "Stay Neutral...", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
                else:
                    if baseline_l is None:
                        baseline_l = np.mean([f['L'] for f in calibration_frames], axis=0)
                        baseline_r = np.mean([f['R'] for f in calibration_frames], axis=0)
                        print("基準点設定完了。計測中...")

                    # --- 移動量計算 ---
                    vec_l_px = fixed_mouth_l - baseline_l
                    vec_r_px = fixed_mouth_r - baseline_r
                    
                    # mm変換 (左: 外側=マイナス方向を反転, 上=マイナス方向を反転)
                    l_dx_mm = -vec_l_px[0] * mm_per_px 
                    l_dy_mm = -vec_l_px[1] * mm_per_px
                    
                    # mm変換 (右: 外側=プラス方向, 上=マイナス方向を反転)
                    r_dx_mm = vec_r_px[0] * mm_per_px  
                    r_dy_mm = -vec_r_px[1] * mm_per_px 

                    frames_data.append({
                        'time': elapsed,
                        'L_dx': l_dx_mm, 'L_dy': l_dy_mm,
                        'R_dx': r_dx_mm, 'R_dy': r_dy_mm
                    })

                    # --- 【新機能】動きのベクトル（矢印）を表示 ---
                    # 実際の画面上でのベクトル開始位置（真顔位置）を計算
                    start_l_disp = rotate_point(baseline_l, p_nose, face_angle)
                    start_r_disp = rotate_point(baseline_r, p_nose, face_angle)
                    
                    # 矢印を描画 (スケールを少し強調して見やすくする: x2)
                    scale_vis = 1.0
                    end_l_disp = tuple((p_mouth_l).astype(int))
                    end_r_disp = tuple((p_mouth_r).astype(int))

                    # 左矢印 (青)
                    cv2.arrowedLine(display_image, tuple(start_l_disp.astype(int)), end_l_disp, (255, 200, 0), 4)
                    # 右矢印 (赤)
                    cv2.arrowedLine(display_image, tuple(start_r_disp.astype(int)), end_r_disp, (0, 0, 255), 4)

                    cv2.putText(display_image, "RECORDING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 数値表示 (リアルタイム)
                    info_l = f"L: Up={l_dy_mm:.1f}mm / Out={l_dx_mm:.1f}mm"
                    info_r = f"R: Up={r_dy_mm:.1f}mm / Out={r_dx_mm:.1f}mm"
                    cv2.putText(display_image, info_l, (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                    cv2.putText(display_image, info_r, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

                if elapsed > 4.5: # 4.5秒で自動停止
                    break
            
            else:
                cv2.putText(display_image, "Press SPACE to Record", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(display_image)

        cv2.imshow('Mouth Corner Analysis (Fixed)', display_image)
        key = cv2.waitKey(5)
        if key == 32 and not is_recording: 
            is_recording = True
            start_time = time.time()
        elif key == 27: 
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frames_data:
        analyze_vectors_mm_fixed(frames_data)

def analyze_vectors_mm_fixed(data):
    times = [d['time'] for d in data]
    L_dx = [d['L_dx'] for d in data]
    L_dy = [d['L_dy'] for d in data]
    R_dx = [d['R_dx'] for d in data]
    R_dy = [d['R_dy'] for d in data]

    # データ全体の最大振れ幅を取得（グラフのスケール統一用）
    all_vals = L_dx + L_dy + R_dx + R_dy
    max_val = max(np.max(np.abs(all_vals)), 15)
    limit = max_val * 1.1

    # --- グラフ描画 ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    # Left Graph (変更なし)
    ax1.plot(times, L_dy, label='Vertical Lift (UP)', color='blue', linewidth=2.5)
    ax1.plot(times, L_dx, label='Horizontal Widen (OUT)', color='cyan', linestyle='--', linewidth=2)
    ax1.set_title('Left Side Movement [mm] - (Ideal Pattern)')
    ax1.set_ylabel('Distance (mm)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=0.5)

    # Right Graph (変更点: 横方向の正負を逆にする)
    # R_dx の値を反転させたリストを作成してプロット
    R_dx_inverted = [-x for x in R_dx]
    
    ax2.plot(times, R_dy, label='Vertical Lift (UP)', color='orange', linewidth=2.5)
    ax2.plot(times, R_dx, label='Horizontal Widen (OUT)', color='red', linestyle='--', linewidth=2)
    ax2.set_title('Right Side Movement [mm]')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Distance (mm)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(GRAPH_PATH_COMPONENTS)

    # Trajectory Graph
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left Trajectory (変更なし: プラス方向メイン)
    ax_l.plot(L_dx, L_dy, color='blue', marker='o', markersize=3, alpha=0.5)
    ax_l.plot(L_dx[0], L_dy[0], 'bx', label='Start', markersize=8)
    ax_l.plot(L_dx[np.argmax(L_dy)], np.max(L_dy), 'bo', fillstyle='none', markersize=10)
    ax_l.set_title('Left Trajectory (mm)')
    ax_l.set_xlabel('Horizontal Widen (mm)')
    ax_l.set_ylabel('Vertical Lift (mm)')
    ax_l.grid(True)
    ax_l.set_xlim(-5, limit) # 右(プラス)方向に広い
    ax_l.set_ylim(-5, limit)
    ax_l.set_aspect('equal')
    
    # Left Angle Calculation
    idx_max_l = np.argmax(L_dy)
    if L_dx[idx_max_l] != 0:
        angle_l = math.degrees(math.atan2(L_dy[idx_max_l], L_dx[idx_max_l]))
    else: angle_l = 90
    ax_l.text(0.05, 0.95, f"Angle: {angle_l:.1f}°", transform=ax_l.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Right Trajectory (変更点: マイナス方向に描画範囲を広げる)
    ax_r.plot(R_dx, R_dy, color='orange', marker='o', markersize=3, alpha=0.5)
    ax_r.plot(R_dx[0], R_dy[0], 'rx', label='Start', markersize=8)
    ax_r.plot(R_dx[np.argmax(R_dy)], np.max(R_dy), 'ro', fillstyle='none', markersize=10)
    ax_r.set_title('Right Trajectory (mm)')
    ax_r.set_xlabel('Horizontal Widen (mm)')
    ax_r.grid(True)
    
    # ここを変更: 左(マイナス)方向に広く、右(プラス)方向は狭く設定
    ax_r.set_xlim(-10, limit) 
    
    ax_r.set_ylim(-5, limit)
    ax_r.set_aspect('equal')

    # Right Angle Calculation
    idx_max_r = np.argmax(R_dy)
    if R_dx[idx_max_r] != 0:
        angle_r = math.degrees(math.atan2(R_dy[idx_max_r], R_dx[idx_max_r]))
    else: angle_r = 90
    ax_r.text(0.05, 0.95, f"Angle: {angle_r:.1f}°", transform=ax_r.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(GRAPH_PATH_TRAJECTORY)
    
    print(f"解析完了！グラフを保存しました: {GRAPH_PATH_TRAJECTORY}")
    plt.show()

if __name__ == "__main__":
    record_and_analyze_mm_fixed()