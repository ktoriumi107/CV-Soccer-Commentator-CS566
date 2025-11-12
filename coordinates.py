import os, cv2, numpy as np
import field_locale

# Config
PATH = "test_images/orthogonal_field_view.png"
HOMOGRAPHY_POINTS = "homography_points.npy"
OUTPUT_PATH = "homography_output.jpg"
WARP_SIZE = (1280, 720)  # width, height for birds eye view

def draw_points(frame, points, color=(0, 0, 255)):
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 6, color, -1)

def save_points(points, path):
    np.save(path, np.array(points, dtype=np.float32))
    print(f"[saved] Homography points -> {path}")

def load_points(path):
    if not os.path.exists(path):
        return None
    pts = np.load(path)
    print(f"[loaded] Homography points from {path}")
    return pts

def select_points(frame, save_path):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Selected: {len(points)} -> ({x}, {y})")

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", click_event)

    print("\nSelect 4 points in order:")
    print("  1. Top-left")
    print("  2. Top-right")
    print("  3. Bottom-right")
    print("  4. Bottom-left")
    print("Press ENTER when done.\n")

    while True:
        temp = frame.copy()
        draw_points(temp, points)
        cv2.imshow("Select 4 Points", temp)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10) or len(points) >= 4:
            break
        elif key == 27:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    if len(points) == 4:
        save_points(points, save_path)
        return np.array(points, dtype=np.float32)
    else:
        print("Not enough points selected.")
        return None

def compute_homography(src_pts):
    dst_pts = np.array([
        [0, 0],
        [WARP_SIZE[0] - 1, 0],
        [WARP_SIZE[0] - 1, WARP_SIZE[1] - 1],
        [0, WARP_SIZE[1] - 1]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H, dst_pts

def process_image(image_path, src_pts):
    frame = cv2.imread(image_path)
    assert frame is not None, f"Cannot read image: {image_path}"

    H, dst_pts = compute_homography(src_pts)
    print("[homography] Matrix:\n", H)

    warped = cv2.warpPerspective(frame, H, WARP_SIZE)

    combined = np.hstack([
        cv2.resize(frame, (WARP_SIZE[0], WARP_SIZE[1])),
        warped
    ])

    cv2.imshow("Original (left) | Warped (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.splitext(OUTPUT_PATH)[0] + ".jpg", warped)
    print(f"[saved] Warped image -> {os.path.splitext(OUTPUT_PATH)[0] + '.jpg'}")

def process_video(video_path, src_pts):
    cap = cv2.VideoCapture(0 if str(video_path).lower() in ("webcam", "0") else video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    H, dst_pts = compute_homography(src_pts)
    print("[homography] Matrix:\n", H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, WARP_SIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        warped = cv2.warpPerspective(frame, H, WARP_SIZE)

        cv2.imshow("Original", frame)
        cv2.imshow("Warped (Top-Down)", warped)

        out.write(warped)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[saved] Warped video -> {OUTPUT_PATH}")

def main():
    # Load or select points
    src_pts = load_points(HOMOGRAPHY_POINTS)
    src_pts = None
    first_frame = None

    # If user hasn’t calibrated yet
    if src_pts is None:
        if os.path.splitext(PATH)[1].lower() in [".jpg", ".png"]:
            first_frame = cv2.imread(PATH)
        else:
            cap = cv2.VideoCapture(0 if PATH.lower() in ("webcam", "0") else PATH)
            ok, first_frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("Cannot read frame for calibration.")
        
        field_locale.get_keypoints(first_frame)


        src_pts = select_points(first_frame, HOMOGRAPHY_POINTS)
        if src_pts is None:
            return

    # Process input
    ext = os.path.splitext(PATH)[1].lower()
    if ext in [".jpg", ".png", ".jpeg"]:
        process_image(PATH, src_pts)
    else:
        process_video(PATH, src_pts)

if __name__ == "__main__":
    main()
