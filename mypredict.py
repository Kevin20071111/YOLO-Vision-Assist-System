from sympy import false

from ultralytics import YOLO

model = YOLO(r"C:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train7\weights\best.pt")
model.predict(
    source=r"C:\deeplearning\target\images",#要檢測的內容路徑
    save=True,#儲存結果
    show=False,#不馬上展示結果
    save_txt=True,#把預測結果保存為txt文件
)