from ultralytics import YOLO #導入模型

if __name__ == "__main__":
    model = YOLO(r"yolo11s.pt")
    model.train(
        data=r"test_dataset.yaml",#配置文件
        epochs=50, #訓練次數
        imgsz=1280, #縮放大小
        batch=-1,   #訓練時每次給幾張圖 -1為自動選擇最佳值
        cache="ram", #是否先將圖片緩存到記憶體
        workers=0, #記憶體打包圖片的進程數
    )