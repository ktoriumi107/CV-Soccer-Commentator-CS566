import os, cv2, numpy as np

# work in meters - standard field size
W = 105
H = 68

def get_keypoints(frame):
    color_approach(frame)

    return

    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detection
    edges = cv2.Canny(blur, 100, 150, apertureSize=5)

    # Hough transform for lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=50, maxLineGap=20)
    
    intersections = get_line_intersections(lines)

    visualize_intersections(frame, intersections)

    return lines, edges

def color_approach(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color range
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # Find major lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=100, maxLineGap=10)

    # Create mask for lines
    line_mask = np.zeros_like(mask)

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

    # Optional: combine with the green mask to focus on field region
    combined_mask = cv2.bitwise_and(mask, line_mask)

    # Apply mask to original frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Show results for debugging
    cv2.imshow("Green Mask", mask)
    cv2.imshow("Line Mask", line_mask)
    cv2.imshow("Field Boundary", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result, mask, line_mask

def get_line_intersections(lines):
    intersections = []

    if lines is not None:
        # compare each line to each remaining line (should be low ~ 10-50)    

        print(len(lines))

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

    # Check if intersection is within both line segments
    if (min(x1, x2) - 5 <= x <= max(x1, x2) + 5 and
        min(y1, y2) - 5 <= y <= max(y1, y2) + 5 and
        min(x3, x4) - 5 <= x <= max(x3, x4) + 5 and
        min(y3, y4) - 5 <= y <= max(y3, y4) + 5):
        return (int(x), int(y))
    return None
    
def visualize_intersections(frame, intersections):
    vis = frame.copy()
    for (x, y) in intersections:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.putText(vis, f"Intersections: {len(intersections)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow("Line Intersections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()