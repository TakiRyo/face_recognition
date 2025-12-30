# face recognition. 


## eye_details.py. 
Webカメラ映像から「瞬き」や「瞼（まぶた）の動き」をミリメートル単位で解析し、グラフ化するツールです。顔面神経麻痺などの評価支援を目的としています。  
### 機能. 
動画解析 (5秒間): リアルタイムで目の動きをトラッキングし、データを記録します。  
mm単位の定量評価: 目尻・目頭間の距離（デフォルト）または基準シールを用いて、ピクセルをmmに変換します。  
詳細グラフ生成:  
上瞼・下瞼の独立した動き: それぞれがどれくらい動いたかを可視化。  
眼裂（目の開き）: 完全に閉じているか、隙間（兎眼）があるかを可視化。  
![alt text](image.png)
Author: taki. 

## coner_of_mouth_detail.py. 
未実装
口角がどの方向にどれくらい上がっているか. 
石田がやるらしい。  

## web_camera_select.py. 
目尻の長さか、シールか選べる。　　
params:  
eyebrow / eye / mouth threshold. 
メインで使っているもの。  
  
  
  
## no using anymore. 

## web_camera_eye.py. 
web camera 撮影。
目尻の長さを使用してピクセルからmmに変換.  

## analyze_face.py  
水平を自動で調整して、結果を表示。時々ずれるけど簡単

## manual_face_analyze.py  
水平を手動で調整する。面倒だけど、精度は良い。

## manual_face_video.py  
web cameraを使って撮影できる。水平は写真撮影後に手動で水平調整　　

この上の三つは静止画像に対しての評価で、今はdiffrenceに符号がついているけど、これを絶対値で評価するべき。要変更


## excursion.py  
広角がどれくらい上がるか？  

## independence.py  
口を開けた時に目を開けたままでいられるか？  