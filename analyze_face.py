import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================
# 設定: ここに画像のファイル名を入れてください
IMAGE_PATH = '/Users/takiguchiryosei/Documents/face_recognition/faces/para_3.png' 
# ==========================================

# MediaPipeの初期設定 (静止画モード: static_image_mode=True)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,       # 静止画専用の高精度モード
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

# 評価ポイントの定義 (MediaPipe ID)
KEY_POINTS = {
    'Eyebrow': (105, 334),  # 眉上 (左, 右)
    'Eye': (33, 263),       # 目尻 (左, 右)
    'Mouth': (61, 291),     # 口角 (左, 右)
}
# 垂直基準とする鼻筋のランドマーク (168:眉間, 6:鼻梁中央)
NOSE_AXIS_IDS = (168, 6)

def calculate_symmetry_static(landmarks, img_w, img_h):
    """鼻筋基準で傾き補正を行い、左右差を計算する関数"""
    
    # 1. 鼻筋ベクトルから画像の回転角度を算出
    nose_top = landmarks[NOSE_AXIS_IDS[0]]
    nose_bot = landmarks[NOSE_AXIS_IDS[1]]
    
    vec = np.array([nose_bot.x * img_w - nose_top.x * img_w,
                    nose_bot.y * img_h - nose_top.y * img_h])
    
    # 90度(垂直)からのズレを計算
    current_angle = np.arctan2(vec[1], vec[0])
    rotation_angle = (np.pi / 2) - current_angle # 補正に必要な回転量
    
    # 回転行列の作成
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array(((c, -s), (s, c)))
    
    # 2. 各パーツの座標変換と計測
    results = {}
    origin = np.array([nose_top.x * img_w, nose_top.y * img_h]) # 回転中心
    
    # 顔の幅（正規化用）を先に計算
    l_eye = np.array([landmarks[33].x * img_w, landmarks[33].y * img_h])
    r_eye = np.array([landmarks[263].x * img_w, landmarks[263].y * img_h])
    # 回転後の座標で幅を測る
    l_eye_rot = np.dot(rotation_matrix, l_eye - origin)
    r_eye_rot = np.dot(rotation_matrix, r_eye - origin)
    face_width = abs(l_eye_rot[0] - r_eye_rot[0])
    if face_width == 0: face_width = 1

    for name, (l_idx, r_idx) in KEY_POINTS.items():
        # 元座標
        l_pt = landmarks[l_idx]
        r_pt = landmarks[r_idx]
        l_raw = np.array([l_pt.x * img_w, l_pt.y * img_h])
        r_raw = np.array([r_pt.x * img_w, r_pt.y * img_h])

        # 回転補正
        l_rot = np.dot(rotation_matrix, l_raw - origin)
        r_rot = np.dot(rotation_matrix, r_raw - origin)
        
        # 高さの差分 (Y座標)
        y_diff = l_rot[1] - r_rot[1]
        
        # スコア化 (顔幅に対する％)
        score = (y_diff / face_width) * 100
        
        results[name] = {
            'score': score,
            'points': (l_raw, r_raw) # 描画は元画像に行うため生座標を保存
        }
        
    return results, np.degrees(rotation_angle), (origin, vec)

def main():
    # 画像読み込み
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"エラー: 画像ファイル '{IMAGE_PATH}' が見つかりません。")
        return

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 解析実行
    print("解析中...")
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("顔が検出されませんでした。")
        return

    face_landmarks = results.multi_face_landmarks[0] # 1人目を取得
    
    # 対称性計算
    analysis, tilt, _ = calculate_symmetry_static(face_landmarks.landmark, w, h)

    # --- 結果の描画 ---
    print(f"\n--- 診断結果 (補正角度: {tilt:.1f}度) ---")
    
    for i, (part, data) in enumerate(analysis.items()):
        score = data['score']
        lp, rp = data['points']
        
        # コンソール出力
        status = "OK"
        if abs(score) > 2.0: status = "ASYMMETRY (非対称)"
        print(f"{part.ljust(8)}: {score:5.1f}% ... {status}")

        # 画像への描画
        color = (0, 255, 0) # 緑
        if abs(score) > 2.0: color = (0, 0, 255) # 赤
        
        cv2.circle(image, tuple(lp.astype(int)), 5, color, -1)
        cv2.circle(image, tuple(rp.astype(int)), 5, color, -1)
        cv2.line(image, tuple(lp.astype(int)), tuple(rp.astype(int)), (200, 200, 200), 1)
        
        # 数値を画像に書き込み
        text = f"{score:.1f}%"
        cv2.putText(image, text, (int(lp[0])-20, int(lp[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 鼻筋（基準軸）の可視化
    nose_top_idx = NOSE_AXIS_IDS[0]
    nose_bot_idx = NOSE_AXIS_IDS[1]
    nt = face_landmarks.landmark[nose_top_idx]
    nb = face_landmarks.landmark[nose_bot_idx]
    pt_top = (int(nt.x * w), int(nt.y * h))
    pt_bot = (int(nb.x * w), int(nb.y * h))
    cv2.line(image, pt_top, pt_bot, (0, 255, 255), 2)
    cv2.putText(image, "Axis", pt_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 画像を保存・表示
    output_filename = 'result_' + IMAGE_PATH
    cv2.imwrite(output_filename, image)
    print(f"\n診断画像を保存しました: {output_filename}")
    
    cv2.imshow('Analysis Result', image)
    cv2.waitKey(0) # キーを押すまで待機
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()