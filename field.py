import numpy as np
import cv2

def mask_field_green(img):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # filter to green only
    lower = np.array([25, 45, 30])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # clean up mask 
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
    return mask

def get_field_boundary_points(mask):
    # get the external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # get the contour with the largest area (should be the exterior of the field)
    contour = max(contours, key=cv2.contourArea)
    
    # convert to points
    pts = contour.reshape(-1, 2)
    return pts

def get_candidate_lines(points, max_lines, ransac_thresh, ransac_iter, min_inliers):
    if points is None or len(points) == 0:
        return []

    pts = points.copy()
    candidates = []

    # get the
    for _ in range(max_lines):
        line, inliers = ransac_fit_line(pts, threshold=ransac_thresh, iterations=ransac_iter)
        if line is None or inliers is None or len(inliers) < min_inliers:
            break

        inlier_pts = pts[inliers]
        candidates.append((line, inlier_pts, len(inliers)))

        # remove inliers so they don't get double counted in lines
        pts = np.delete(pts, inliers, axis=0)

    return candidates

def ransac_fit_line(points, threshold, iterations):
    if points is None or len(points) < 2:
        return None, None

    # keep track of best line
    best_line = None
    best_inliers = None
    n = len(points)

    for _ in range(iterations):
        # choose two random points
        i1, i2 = np.random.choice(n, 2, replace=False)
        p1 = points[i1]; p2 = points[i2]
        x1,y1 = p1; x2,y2 = p2

        # compute line equation of form ax+by+c=0
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1

        # compute the norm
        norm = np.hypot(a, b)
        if norm < 1e-8:
            continue

        # noramalize the line equation
        a, b, c = a/norm, b/norm, c/norm

        # compute distance to line for all points
        dists = np.abs(a*points[:,0] + b*points[:,1] + c)

        # find the points that lie within that line (closer than a min threshold)
        inliers = np.where(dists < threshold)[0]

        # choose best model as the one with the most inliers
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (a, b, c)

    return best_line, best_inliers

def line_segment_endpoints_from_inliers(inlier_pts):
    if inlier_pts is None or len(inlier_pts) < 2:
        return None, None

    pts = inlier_pts.astype(float)
    n = len(pts)

    # find the pair of points with max distance
    max_dist = -1
    p1 = None
    p2 = None

    # compare every pair of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]

            # euclidean distance
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # check if largest distance
            if dist > max_dist:
                max_dist = dist
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))

    return p1, p2

def filter_out_border(segments, size):
        filtered_segments = []

        # threshold for being near something
        threshold = 10

        # image size
        H, W, _ = size

        for (p1, p2) in segments:
            x1, y1 = p1
            x2, y2 = p2

            # skip if line is near a border
            if ((x1 < threshold and x2 < threshold) or
                (x1 > (W-threshold) and x2 > (W-threshold)) or
                (y1 < threshold and y2 < threshold) or
                (y1 > (H-threshold) and y2 > (H-threshold))):
                continue

            filtered_segments.append((p1,p2))
        
        return filtered_segments

def group_segments(segments):
    threshold = np.deg2rad(10)

    groups = []
    sets = []

    # for each potential line
    for (p1, p2) in segments:
        x1, y1 = p1
        x2, y2 = p2

        angle = np.arctan2((y2-y1),(x2-x1)) 

        # add first element
        if len(groups) == 0:
            groups.append((angle-threshold), (angle+threshold))
            sets.append([(p1,p2)])
            continue

        # try to match it to a preset group
        for i, (lower,upper) in groups:
            # if it is within the groups range, add it
            if lower<=angle and angle<=upper:
                sets[i].append((p1,p2))

                # extend bounds in the appropriate direction
                groups[i] =  (min(lower, angle - threshold),
                              max(upper, angle + threshold))
                break

            if i == len(groups):
                # no match, make a new group
                groups.append((angle-threshold, angle+threshold))
                sets.append([(p1, p2)])

    # get two most populated groups
    # this should be the correct lines but could be unstable # TODO TODO
    group_sizes = [(len(set), i) for i, set in enumerate(sets)]
    group_sizes.sort(reverse=True)

    # get two most populated groups
    top_groups = group_sizes[:2]

    final_lines = []

    # get the average line equation per group
    for group, i in top_groups:
        lines = []

        for (p1, p2) in group:
            # realized how silly I am
            print("No need for this, go back")

    return final_lines

def visualize(image, segments):
    out = image.copy()

    # draw the lines by endpoints
    for (p1, p2) in segments:
        cv2.line(out, p1, p2, (0, 0, 255), 2)
    return out

if __name__ == "__main__":
    # tuning params
    max_lines = 8
    ransac_thresh=2.0
    ransac_iter=2000
    min_inliers=30

    img = cv2.imread("test_images/oblique_field_view.png")

    green_mask = mask_field_green(img)
    cv2.imwrite("debug_green_mask.png", green_mask)

    boundary_pts = get_field_boundary_points(green_mask)

    candidates = get_candidate_lines(boundary_pts, max_lines, ransac_thresh,
                                     ransac_iter,min_inliers)

    segments = []

    for (line, inliers, _) in candidates:
        if inliers is None or len(inliers) == 0:
            continue
        p1, p2 = line_segment_endpoints_from_inliers(inliers)
        segments.append((p1, p2))

    final_points = filter_out_border(segments, img.shape)

    print(final_points)
    #final_points = group_segments(final_points)

    img = visualize(img, final_points)

    # Save and show
    cv2.imshow("RANSAC Candidates", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
