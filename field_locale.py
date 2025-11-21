import os, cv2, numpy as np

# work in meters - standard field size
W = 105
H = 68

def get_keypoints(frame):
    frame = color_approach(frame)

    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 10)

    # edge detection
    edges = cv2.Canny(blur, 100, 150, apertureSize=5)

    # Hough transform for lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                            minLineLength=100, maxLineGap=100)
    
    intersections = get_line_intersections(lines)

    visualize_lines_and_intersections(frame, lines, intersections)

    return lines, edges


def color_approach(frame):
    """Extract main green field area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green color range (tune if needed)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup — remove small spots, fill holes
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours (connected regions)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field_mask = np.zeros_like(mask)

    if contours:
        # Take the largest contour (assumed to be the field)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(field_mask, [largest_contour], -1, 255, -1)

    # Apply field mask to image
    field = cv2.bitwise_and(frame, frame, mask=field_mask)
    return field


def get_line_intersections(lines):
    intersections = []
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = segment_intersection(lines[i][0], lines[j][0])
                if pt is not None:
                    intersections.append(pt)
    return intersections


def segment_intersection(line1, line2):
    """Return intersection point of two line segments, or None if no intersection."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Line AB represented as a1x + b1y = c1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    # Line CD represented as a2x + b2y = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-6:
        return None  # Parallel

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    # Check if intersection lies within both segments
    if (min(x1, x2) - 5 <= x <= max(x1, x2) + 5 and
        min(y1, y2) - 5 <= y <= max(y1, y2) + 5 and
        min(x3, x4) - 5 <= x <= max(x3, x4) + 5 and
        min(y3, y4) - 5 <= y <= max(y3, y4) + 5):
        return (int(x), int(y))
    return None

def visualize_lines_and_intersections(frame, lines, intersections):
    vis = frame.copy()

    # Draw Hough lines in blue
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw intersection points in red
    for (x, y) in intersections:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.putText(vis, f"Lines: {len(lines) if lines is not None else 0}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Intersections: {len(intersections)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Detected Lines and Intersections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points_clockwise(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32)
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(angles)
    pts = pts[idx]

    # start from TL (smallest x+y)
    s = pts.sum(axis=1)
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)
    return pts


def get_field_corners_from_lines(lines):
    """Estimate the four field corners from Hough lines."""
    if lines is None or len(lines) < 4:
        print("Not enough lines for corner detection.")
        return None

    # Collect intersection points
    intersections = get_line_intersections(lines)
    if len(intersections) < 4:
        print("Not enough intersections.")
        return None

    pts = np.array(intersections, dtype=np.float32)

    # Keep only points roughly within frame bounds (sanity check)
    xs, ys = pts[:, 0], pts[:, 1]
    xmin, xmax = np.percentile(xs, [5, 95])
    ymin, ymax = np.percentile(ys, [5, 95])
    mask = (xs > xmin) & (xs < xmax) & (ys > ymin) & (ys < ymax)
    pts = pts[mask]

    # Approximate quadrilateral from convex hull
    hull = cv2.convexHull(pts)
    if len(hull) < 4:
        print("Convex hull too small.")
        return None

    # Simplify hull to 4 vertices
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) != 4:
        # fallback: take 4 extreme corners
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        approx = np.int32(box)
    corners = approx.reshape(-1, 2)
    corners = order_points_clockwise(corners)
    return corners


def compute_homography_auto(frame, lines):
    """Compute homography automatically using field borders."""
    corners = get_field_corners_from_lines(lines)
    if corners is None or len(corners) != 4:
        raise RuntimeError("Failed to extract 4 field corners.")

    # Field coordinates in meters (standard FIFA dimensions)
    dst = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ], dtype=np.float32)

    H_img2world, _ = cv2.findHomography(corners, dst)
    print("Homography:\n", H_img2world)

    # For visualization — warp to bird’s-eye view
    scale = 10  # pixels per meter
    out_w, out_h = int(W * scale), int(H * scale)
    H_scaled = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]) @ H_img2world
    warped = cv2.warpPerspective(frame, H_scaled, (out_w, out_h))

    # Draw detected corners
    vis = frame.copy()
    for (x, y) in corners:
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
    cv2.imshow("Detected Corners", vis)
    cv2.imshow("Bird's Eye View", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H_img2world, corners

if __name__ == "__main__":
    frame = cv2.imread("test_images/orthogonal_field_view.png")  # your test frame
    lines, edges = get_keypoints(frame)
    H, corners = compute_homography_auto(frame, lines)
