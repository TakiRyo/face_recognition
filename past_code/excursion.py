import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Create directories for saving results
RESULT_DIR = '/Users/takiguchiryosei/Documents/face_recognition/excursion_results'

# Update paths for saving results
VIDEO_PATH = os.path.join(RESULT_DIR, 'motion_result_test.mp4')
GRAPH_PATH = os.path.join(RESULT_DIR, 'excursion_graph_test.png')

# Ensure the result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 保存ファイル名
OUTPUT_VIDEO_PATH = VIDEO_PATH  # トラッキング確認用動画
# ==========================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ランドマークID
# 左右の目尻（顔幅の正規化用）
ID_EYE_L = 33
ID_EYE_R = 263
# 鼻の付け根（不動の基準点）
ID_NOSE_BRIDGE = 168 
# 口角（測定対象）
ID_MOUTH_L = 61
ID_MOUTH_R = 291

def calculate_distance(p1, p2, w, h):
    """2点間のユークリッド距離を計算"""
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def record_and_analyze():
    # 1. カメラ起動
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return

    # 動画保存の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

    frames_data = [] # 解析データ保存用
    is_recording = False
    print("--- 準備完了 ---")
    print(" [SPACE]: 録画開始（3秒後に自動終了します）")
    print(" 「真顔」からスタートして、「思いっきり笑って」ください！")

    start_time = 0
    
    while True:
        success, image = cap.read()
        if not success: break

        # 鏡のように反転
        image = cv2.flip(image, 1)
        display_image = image.copy()
        
        # 録画中の処理
        if is_recording:
            # 経過時間
            elapsed = time.time() - start_time
            cv2.putText(display_image, f"REC: {elapsed:.1f}s", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(display_image, (w-50, 50), 10, (0, 0, 255), -1)
            
            # --- 解析 (MediaPipe) ---
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            current_data = {'time': elapsed, 'L': 0, 'R': 0, 'valid': False}

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # 座標取得
                p_eye_l = lm[ID_EYE_L]
                p_eye_r = lm[ID_EYE_R]
                p_nose = lm[ID_NOSE_BRIDGE]
                p_mouth_l = lm[ID_MOUTH_L]
                p_mouth_r = lm[ID_MOUTH_R]

                # 1. 顔の幅（正規化係数）
                face_width = calculate_distance(p_eye_l, p_eye_r, w, h)
                
                # 2. 鼻から口角までの距離（動きの絶対値）
                dist_l = calculate_distance(p_nose, p_mouth_l, w, h)
                dist_r = calculate_distance(p_nose, p_mouth_r, w, h)
                
                # 3. 正規化（顔の大きさによる誤差を排除）
                # 値は「顔幅に対する比率」になります
                norm_dist_l = dist_l / face_width
                norm_dist_r = dist_r / face_width
                
                current_data['L'] = norm_dist_l
                current_data['R'] = norm_dist_r
                current_data['valid'] = True

                # --- トラッキング確認用の描画 (動画に書き込む) ---
                # 口角に点を打つ
                pt_l = (int(p_mouth_l.x * w), int(p_mouth_l.y * h))
                pt_r = (int(p_mouth_r.x * w), int(p_mouth_r.y * h))
                pt_nose = (int(p_nose.x * w), int(p_nose.y * h))
                
                cv2.line(image, pt_nose, pt_l, (0, 255, 0), 1)
                cv2.line(image, pt_nose, pt_r, (0, 255, 0), 1)
                cv2.circle(image, pt_l, 3, (0, 255, 0), -1)
                cv2.circle(image, pt_r, 3, (0, 255, 0), -1)

            frames_data.append(current_data)
            writer.write(image) # 解析線が入った画像を保存

            # 4秒で自動終了
            if elapsed > 4.0:
                print("録画終了。解析を行います...")
                break

        else:
            # 待機画面
            cv2.putText(display_image, "Press [SPACE] to Start", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Motion Analysis', display_image)
        
        key = cv2.waitKey(5)
        if key == 32 and not is_recording: # SPACE
            is_recording = True
            start_time = time.time()
        elif key == 27: # ESC
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # データが空なら終了
    if not frames_data:
        return

    # --- グラフ描画と数値解析 ---
    analyze_data(frames_data)

def analyze_data(data):
    # 有効なデータのみ抽出
    valid_data = [d for d in data if d['valid']]
    if not valid_data:
        print("顔が検出されませんでした")
        return

    times = [d['time'] for d in valid_data]
    raw_l = [d['L'] for d in valid_data]
    raw_r = [d['R'] for d in valid_data]
    
    # ゼロ点補正（最初の5フレームの平均を「基準点0」とする）
    # これにより「移動量」だけを抽出する
    base_l = np.mean(raw_l[:5])
    base_r = np.mean(raw_r[:5])
    
    # 変位（移動量）に変換
    # ※口角は笑うと鼻に近づく＝距離が減るため、(基準 - 現在) をとる
    #  あるいは単純に変化量の絶対値をとる
    move_l = [abs(val - base_l) * 100 for val in raw_l] # x100で％表示っぽく
    move_r = [abs(val - base_r) * 100 for val in raw_r]

    # ピーク値（最大可動域）
    max_l = np.max(move_l)
    max_r = np.max(move_r)
    
    # 対称性スコア (健側を100とした場合の患側の割合)
    # 仮に大きい方を「健側」とする
    if max_l > max_r:
        ratio = (max_r / max_l) * 100
        diagnosis = f"Left side is stronger. Right is {ratio:.1f}% of Left."
    else:
        ratio = (max_l / max_r) * 100
        diagnosis = f"Right side is stronger. Left is {ratio:.1f}% of Right."

    # --- グラフプロット ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, move_l, label='Left Mouth', color='blue', linewidth=2)
    plt.plot(times, move_r, label='Right Mouth', color='orange', linewidth=2)
    
    plt.title(f"Range of Motion (Excursion)\n{diagnosis}")
    plt.xlabel("Time (sec)")
    plt.ylabel("Displacement (Normalized)")
    plt.legend()
    plt.grid(True)
    
    # ピーク地点にマーカー
    plt.plot(times[np.argmax(move_l)], max_l, 'bo')
    plt.plot(times[np.argmax(move_r)], max_r, 'ro')
    
    print("\n" + "="*30)
    print(f"Max Excursion Left : {max_l:.2f}")
    print(f"Max Excursion Right: {max_r:.2f}")
    print(diagnosis)
    print("="*30 + "\n")
    
    # Save the graph as an image file
    plt.savefig(GRAPH_PATH)
    print(f"Graph saved as '{GRAPH_PATH}' in the results folder.")
    plt.show()

if __name__ == "__main__":
    record_and_analyze()