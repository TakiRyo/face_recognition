import cv2
import mediapipe as mp
import numpy as np
import os

# 結果保存用ディレクトリ
RESULT_DIR = '/Users/takiguchiryosei/Documents/face_recognition/webcam_results'
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(RESULT_DIR, 'result.png')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

# 評価する特徴点
KEY_POINTS = {
    'Eyebrow': (105, 334),
    'Eye': (33, 263),   
    'Mouth': (61, 291),
}

# 実測した左右目尻間の距離（mm）- ユーザーの実測値
MEASURED_EYE_WIDTH_MM = 105.0
REFERENCE_STICKER_MM = 10.0  # 赤い円形シールの実寸直径（mm）

def select_scale_method():
    print("\nSelect scale method:")
    print("  [1] Eye width (landmarks 33-263)")
    print("  [2] Forehead RED sticker (10mm)")
    choice = input("Enter 1 or 2 (default 1): ").strip()
    return 'sticker' if choice == '2' else 'eye'

def calculate_scale_from_sticker(image, landmarks, sticker_mm=REFERENCE_STICKER_MM):
    """
    Forehead sticker-based mm/px scale estimation.
    - image: BGR frame (numpy array)
    - landmarks: MediaPipe FaceMesh landmarks (list of 468 items)
    Returns (mm_per_px, annotated_image)
    """
    h, w = image.shape[:2]
    lm = landmarks

    # 安全チェック
    if lm is None or len(lm) < 11:
        return None, image

    # ランドマーク座標（ピクセル）
    p10 = np.array([lm[10].x * w, lm[10].y * h])  # 生え際付近
    p9 = np.array([lm[9].x * w, lm[9].y * h])     # 眉間中央

    # 額の探索範囲（ROI）を作成
    # 垂直距離を基準にスケールすることで顔サイズにロバスト化
    v_dist = np.linalg.norm(p10 - p9)
    if v_dist == 0:
        v_dist = max(h * 0.05, 30)  # フォールバック

    # 額の中心を眉間と生え際の中点付近に設定
    cx = int((p10[0] + p9[0]) / 2)
    cy = int(min(p10[1], p9[1]))

    # 額は眉間より上にあるため、上方向に広めにとる
    roi_width = int(v_dist * 2.0)
    roi_height_top = int(v_dist * 2.5)
    roi_height_bottom = int(v_dist * 0.8)

    x1 = max(0, cx - roi_width // 2)
    x2 = min(w, cx + roi_width // 2)
    y1 = max(0, cy - roi_height_top)
    y2 = min(h, cy + roi_height_bottom)

    if x2 <= x1 or y2 <= y1:
        return None, image

    roi = image[y1:y2, x1:x2].copy()

    # HSV 変換
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 赤色マスク（0付近と180付近の2レンジ）
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # ノイズ除去（Open/Close）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_circle = None  # (center_xy, radius)

    roi_area = (x2 - x1) * (y2 - y1)
    min_area = max(100.0, roi_area * 0.0005)  # ROI比で下限面積を設定

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        (cx_r, cy_r), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)
        if circle_area <= 0:
            continue
        fill_ratio = area / circle_area  # 1に近いほど円に近い

        # 大きさも考慮してスコア化
        score = circularity * fill_ratio * area
        if score > best_score:
            best_score = score
            best_circle = ((cx_r, cy_r), radius)

    if best_circle is None or best_circle[1] <= 0:
        # デバッグ: ROI枠だけ描画して返す
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(debug_img, "Sticker not found", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(debug_img, "Sticker not found", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return None, debug_img

    # 直径(px)から mm/px を算出
    ((cx_r, cy_r), radius) = best_circle
    diameter_px = 2.0 * radius
    mm_per_px = float(sticker_mm) / float(diameter_px) if diameter_px > 0 else None

    # デバッグ描画（元画像に重ねる）
    debug_img = image.copy()
    center_abs = (int(x1 + cx_r), int(y1 + cy_r))
    radius_int = int(radius)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(debug_img, center_abs, radius_int, (0, 255, 0), 2)
    label = f"Sticker d={diameter_px:.1f}px | mm/px={mm_per_px:.4f}"
    cv2.putText(debug_img, label, (center_abs[0] - 80, center_abs[1] - radius_int - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(debug_img, label, (center_abs[0] - 80, center_abs[1] - radius_int - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return mm_per_px, debug_img

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return None
    print("--- 撮影モード: [SPACE]で決定 ---")
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)
        cv2.putText(display_frame, "READY: Press [SPACE] to Capture", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Webcam - Capture', display_frame)
        key = cv2.waitKey(1)
        if key == 32: # Space
            captured_frame = frame
            break
        elif key == 27: break # ESC
    cap.release()
    cv2.destroyAllWindows()
    return captured_frame

def apply_transform(image, angle, tx, ty):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(image, M, (w, h), borderValue=(200, 200, 200))

def manual_alignment_grid(image):
    """
    太い線は縦のみ＋横は等間隔グリッド
    """
    angle = 0.0
    tx = 0
    ty = 0
    
    base_image = image.copy()
    h, w = base_image.shape[:2]
    cx, cy = w // 2, h // 2 
    
    print("--- アライメントモード ---")
    print(" [矢印キー]: 移動  [Z/X]: 回転")
    print(" [Enter]: 確定")
    
    while True:
        display_img = apply_transform(base_image, angle, tx, ty)
        
        # --- グリッド描画 ---
        main_color = (0, 255, 255) # 黄色（太い）
        sub_color = (0, 150, 150)  # 暗めの黄色（細い）
        
        # 1. メインの縦線 (これだけ太く表示)
        cv2.line(display_img, (cx, 0), (cx, h), main_color, 2)
        
        # 2. 等間隔の水平ライン (位置合わせ用のガイド)
        step = 50 
        # 下方向
        for y in range(cy, h, step):
            cv2.line(display_img, (0, y), (w, y), sub_color, 1)
        # 上方向
        for y in range(cy, -1, -step):
            cv2.line(display_img, (0, y), (w, y), sub_color, 1)

        # 情報表示
        info = f"Angle: {angle:.1f}  X:{tx} Y:{ty}"
        cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        cv2.imshow('Manual Alignment', display_img)
        
        key = cv2.waitKey(0)
        
        if key == 13: # Enter
            break
        elif key == 27: # ESC
            return None
            
        step_move = 2
        step_angle = 0.2
        
        if key == ord('z'): angle += step_angle
        elif key == ord('x'): angle -= step_angle
        elif key == 81 or key == 2424832: tx -= step_move 
        elif key == 82 or key == 2555904: ty -= step_move 
        elif key == 83 or key == 2621440: tx += step_move 
        elif key == 84 or key == 2490368: ty += step_move 
        elif key == 0: ty -= step_move
        elif key == 1: ty += step_move
        elif key == 2: tx -= step_move
        elif key == 3: tx += step_move

    cv2.destroyAllWindows()
    return apply_transform(base_image, angle, tx, ty)

def analyze_symmetry_mm(image, scale_method: str = 'eye'):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("顔検出に失敗しました。")
        return image

    lm = results.multi_face_landmarks[0].landmark

    # --- 1. スケール算出: 2方式（目尻・シール）---
    p_left_eye = np.array([lm[33].x * w, lm[33].y * h])
    p_right_eye = np.array([lm[263].x * w, lm[263].y * h])
    eye_width_px = np.linalg.norm(p_left_eye - p_right_eye)

    mm_per_px = 0.0
    scale_info_method = 'eye'
    if scale_method == 'sticker':
        mm_from_sticker, sticker_dbg_img = calculate_scale_from_sticker(image, lm, REFERENCE_STICKER_MM)
        if mm_from_sticker is not None:
            mm_per_px = mm_from_sticker
            image = sticker_dbg_img  # ステッカー検出のデバッグ描画を反映
            scale_info_method = 'sticker'
        else:
            print("[INFO] Sticker not found. Falling back to eye-width scale.")
            mm_per_px = MEASURED_EYE_WIDTH_MM / eye_width_px if eye_width_px > 0 else 0.0
            scale_info_method = 'eye(fallback)'
    else:
        mm_per_px = MEASURED_EYE_WIDTH_MM / eye_width_px if eye_width_px > 0 else 0.0
        scale_info_method = 'eye'

    # 顔幅（正規化・比較用）
    face_width_px = abs(lm[33].x * w - lm[263].x * w)
    face_width_mm = face_width_px * mm_per_px

    # デバッグ出力
    print("\n" + "-"*60)
    print("[DEBUG] Scale method:")
    print(f"  method: {scale_info_method}")
    print("[DEBUG] Eye width (outer corners):")
    print(f"  eye_width_px (33-263): {eye_width_px:.2f} px | ref_mm: {MEASURED_EYE_WIDTH_MM:.1f}")
    print("[DEBUG] mm_per_px:")
    print(f"  {mm_per_px:.5f} mm/px")
    print("[DEBUG] Face width estimates:")
    print(f"  face_width_px: {face_width_px:.2f} px")
    print(f"  face_width_mm: {face_width_mm:.2f} mm")
    print("-"*60)

    # 画面左上にデバッグ情報を重ねて表示
    overlay_lines = [
        f"Scale: {scale_info_method}",
        f"Eye width: {eye_width_px:.1f}px => mm/px {mm_per_px:.4f}",
        f"Face: {face_width_px:.1f}px = {face_width_mm:.1f}mm"
    ]
    y0 = 25
    for i, line in enumerate(overlay_lines):
        cv2.putText(image, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(image, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    print("\n" + "="*45)
    print(f"{'PART':<12} | {'Y-DIFF(px)':<10} | {'SCORE(%)':<8} | {'DIFF(mm)':<8}")
    print("-" * 45)

    for name, (l_idx, r_idx) in KEY_POINTS.items():
        lp = np.array([lm[l_idx].x * w, lm[l_idx].y * h])
        rp = np.array([lm[r_idx].x * w, lm[r_idx].y * h])

        # 垂直方向の差分（ピクセル）
        y_diff_px = lp[1] - rp[1]
        
        # 物理距離（mm）への変換
        y_diff_mm = y_diff_px * mm_per_px
        
        # 従来通りの比率スコア（%）
        score_percent = (y_diff_px / face_width_px) * 100

        # ターミナル表示
        print(f"{name:<12} | {y_diff_px:10.1f} | {score_percent:7.1f}% | {abs(y_diff_mm):6.2f} mm")

        # 画像への描画
        color = (0, 255, 0) if abs(score_percent) < 2.5 else (0, 0, 255)
        cv2.circle(image, tuple(lp.astype(int)), 5, color, -1)
        cv2.circle(image, tuple(rp.astype(int)), 5, color, -1)
        cv2.line(image, tuple(lp.astype(int)), tuple(rp.astype(int)), (200, 200, 200), 1)

        # mm表示を優先して大きく描画
        label = f"{abs(y_diff_mm):.1f}mm ({score_percent:+.1f}%)"
        cv2.putText(image, label, (int(lp[0])-40, int(lp[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # 縁取り
        cv2.putText(image, label, (int(lp[0])-40, int(lp[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    print("="*45)
    print(f"Scale Reference: Eye Width (33-263) = {MEASURED_EYE_WIDTH_MM}mm")
    print(f"Face width = {face_width_mm:.2f} mm")
    return image

# --- Main ---
scale_method = 'eye'
try:
    scale_method = select_scale_method()
except Exception:
    pass

original_img = capture_from_webcam()
if original_img is not None:
    aligned_img = manual_alignment_grid(original_img)
    if aligned_img is not None:
        result_img = analyze_symmetry_mm(aligned_img, scale_method=scale_method)
        cv2.imwrite(RESULT_PATH, result_img)
        print(f'Results saved to: {RESULT_PATH}')
        cv2.imshow('Final Report (mm)', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()