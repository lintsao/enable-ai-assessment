import cv2

def list_cameras(max_index=10):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        cams.append(i)
        cap.release()
    return cams

if __name__ == "__main__":
    available = list_cameras()
    print("Available camera indices:", available)
    choice = input(f"Select camera index from {available}: ")
    try:
        idx = int(choice)
        if idx not in available:
            raise ValueError
    except ValueError:
        print(f"Invalid selection: {choice}")
        exit(1)

    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    print(f"Camera {idx} read OK:", ok)
    if ok:
        cv2.imshow(f"Camera {idx}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()

