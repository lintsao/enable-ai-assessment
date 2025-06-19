import cv2

cap = cv2.VideoCapture(0)  # 換成你確定的 index，例如 1

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

print("✅ Webcam opened. Reading frames...")

while True:
    ret, frame = cap.read()
    print("Read success:", ret)

    if not ret:
        continue

    # 測試畫面儲存
    cv2.imwrite("frame.jpg", frame)

    # 顯示畫面
    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
