from ultralytics import YOLO

def main():
    # 1. 事前学習済みモデルの読み込み
    model = YOLO("yolo26s-pose.pt")

    # 2. ファインチューニングの学習を開始
    model.train(
        data="data.yaml",   # データセット設定ファイル
        epochs=500,         # 学習エポック数
        name="pose-tune",   # 学習セッションの名前
        device=0,           # GPU(CUDA:0)を指定
        batch=16,           # RTX 3050 (4GB) でメモリ不足(OOM)が出る場合は 8 や 4 に下げる
        workers=4           # Windows環境での安定動作のため並列数を調整
    )

if __name__ == '__main__':
    main()