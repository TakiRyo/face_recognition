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

# 虹彩（黒目）のインデックス（MediaPipe仕様）
LEFT_IRIS = [469, 470, 471, 472] # 左目虹彩の外周
RIGHT_IRIS = [474, 475, 476, 477] # 右目虹彩の外周
IRIS_DIAMETER_MM = 11.7 # 人間の虹彩の平均直径（mm）

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

def analyze_symmetry_mm(image):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("顔検出に失敗しました。")
        return image

    lm = results.multi_face_landmarks[0].landmark

    # --- 1. 虹彩（黒目）からスケール（mm/px）を計算 ---
    # 既存の計算（隣接点間）
    iris_p1 = np.array([lm[LEFT_IRIS[0]].x * w, lm[LEFT_IRIS[0]].y * h])
    iris_p2 = np.array([lm[LEFT_IRIS[1]].x * w, lm[LEFT_IRIS[1]].y * h])
    iris_diameter_px_legacy = np.linalg.norm(iris_p1 - iris_p2)

    # より厳密な直径（対向点間の縦横の平均）
    iris_top = np.array([lm[LEFT_IRIS[0]].x * w, lm[LEFT_IRIS[0]].y * h])
    iris_bottom = np.array([lm[LEFT_IRIS[2]].x * w, lm[LEFT_IRIS[2]].y * h])
    iris_left = np.array([lm[LEFT_IRIS[1]].x * w, lm[LEFT_IRIS[1]].y * h])
    iris_right = np.array([lm[LEFT_IRIS[3]].x * w, lm[LEFT_IRIS[3]].y * h])
    iris_vert_px = np.linalg.norm(iris_top - iris_bottom)
    iris_horz_px = np.linalg.norm(iris_left - iris_right)
    iris_diameter_px_avg = (iris_vert_px + iris_horz_px) / 2.0

    # 1ピクセルが何ミリか (mm/px)
    mm_per_px_legacy = IRIS_DIAMETER_MM / iris_diameter_px_legacy if iris_diameter_px_legacy > 0 else 0.0
    mm_per_px_avg = IRIS_DIAMETER_MM / iris_diameter_px_avg if iris_diameter_px_avg > 0 else 0.0

    # 顔幅（正規化・比較用）
    face_width_px = abs(lm[33].x * w - lm[263].x * w)
    face_width_mm_legacy = face_width_px * mm_per_px_legacy
    face_width_mm_avg = face_width_px * mm_per_px_avg

    # デバッグ出力
    print("\n" + "-"*60)
    print("[DEBUG] Iris diameters (px):")
    print(f"  legacy(adjacent): {iris_diameter_px_legacy:.2f}")
    print(f"  vertical(opposite): {iris_vert_px:.2f}")
    print(f"  horizontal(opposite): {iris_horz_px:.2f}")
    print(f"  average(opposite): {iris_diameter_px_avg:.2f}")
    print("[DEBUG] mm_per_px:")
    print(f"  legacy: {mm_per_px_legacy:.5f} mm/px")
    print(f"  average: {mm_per_px_avg:.5f} mm/px")
    print("[DEBUG] Face width estimates:")
    print(f"  face_width_px: {face_width_px:.2f} px")
    print(f"  legacy: {face_width_mm_legacy:.2f} mm | avg: {face_width_mm_avg:.2f} mm")
    try:
        target_mm = 120.0
        legacy_err = (face_width_mm_legacy - target_mm)
        avg_err = (face_width_mm_avg - target_mm)
        legacy_err_pct = (legacy_err / target_mm) * 100.0
        avg_err_pct = (avg_err / target_mm) * 100.0
        print(f"  vs 120mm -> legacy Δ: {legacy_err:+.2f} mm ({legacy_err_pct:+.1f}%), avg Δ: {avg_err:+.2f} mm ({avg_err_pct:+.1f}%)")
    except Exception as e:
        print(f"[DEBUG] comparison error: {e}")
    print("-"*60)

    # 画面左上にデバッグ情報を重ねて表示
    overlay_lines = [
        f"Iris(px) adj:{iris_diameter_px_legacy:.1f} v:{iris_vert_px:.1f} h:{iris_horz_px:.1f}",
        f"mm/px adj:{mm_per_px_legacy:.4f} avg:{mm_per_px_avg:.4f}",
        f"Face px:{face_width_px:.1f} mm adj:{face_width_mm_legacy:.1f} avg:{face_width_mm_avg:.1f}",
        "Target 120mm comparison shown in console"
    ]
    y0 = 25
    for i, line in enumerate(overlay_lines):
        cv2.putText(image, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(image, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    print("\n" + "="*45)
    print(f"{'PART':<12} | {'Y-DIFF(px)':<10} | {'SCORE(%)':<8} | {'DIFF(mm)':<8}")
    print("-" * 45)

    for name, (l_idx, r_idx) in KEY_POINTS.items():
        lp = np.array([lm[l_idx].x * w, lm[l_idx].y * h])
        rp = np.array([lm[r_idx].x * w, lm[r_idx].y * h])

        # 垂直方向の差分（ピクセル）
        y_diff_px = lp[1] - rp[1]
        
        # 物理距離（mm）への変換
        # 差分は平均スケールで表示（より安定）
        y_diff_mm = y_diff_px * mm_per_px_avg
        
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
    print(f"Scale Reference: Iris Diameter = {IRIS_DIAMETER_MM}mm")
    print(f"Face width (avg scale) = {face_width_mm_avg:.2f} mm")
    return image

# --- Main ---
original_img = capture_from_webcam()
if original_img is not None:
    aligned_img = manual_alignment_grid(original_img)
    if aligned_img is not None:
        result_img = analyze_symmetry_mm(aligned_img)
        cv2.imwrite(RESULT_PATH, result_img)
        print(f'Results saved to: {RESULT_PATH}')
        cv2.imshow('Final Report (mm)', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()