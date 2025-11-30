import math
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

def average_line(lines):
    # get the average line for a group of lines
    a, b, c = np.mean(np.array(lines), axis=0)

    # normalize
    norm = np.hypot(a,b)

    return (a/norm, b/norm, c/norm)

def group_lines(lines):
    angle_thresh = np.deg2rad(10)

    # [(lower_angle, upper_angle),]
    groups = []

    # list of lists of line equations
    sets = []

    for (a, b, c) in lines:
        theta = np.arctan2(b, a)

        # add first line
        if len(groups) == 0:
            groups.append((theta - angle_thresh, theta + angle_thresh))
            sets.append([(a, b, c)])
            continue

        group_found = False

        # try to fit line into a group
        for i, (low, high) in enumerate(groups):
            if low <= theta <= high:
                sets[i].append((a, b, c))
                
                # fitted, leave loop
                group_found = True
                break

        # if not placed in a group, make a new one
        if not group_found:
            groups.append((theta - angle_thresh, theta + angle_thresh))
            sets.append([(a, b, c)])

    # get two most populated groups
    # this should be the correct lines but could be unstable # TODO TODO
    group_sizes = [(len(set), i) for i, set in enumerate(sets)]
    group_sizes.sort(reverse=True)

    # get two most populated groups
    top_groups = group_sizes[:2]

    _, group1 = top_groups[0]
    _, group2 = top_groups[1]

    return (average_line(sets[group1])), average_line(sets[group2])

def visualize(image, segments):
    out = image.copy()

    # draw the lines by endpoints
    for (p1, p2) in segments:
        cv2.line(out, p1, p2, (0, 0, 255), 2)
    return out

def get_field_coordinate(line1, line2, camera_coordinate):
    # unpack
    x_c, y_c = camera_coordinate
    a1,b1,c1 = line1
    a2,b2,c2 = line2

    x_o,y_o = None

    # origin as intersection of two points
    # intersection of two lines, group as Ax=b: [a1 b1;a2 b2][x; y]=[-c1;-c2]
    # check A' is invertible
    if a1*b2 - a2*b1 < 1e-5:
        print("Nearly parallel lines. Failed to get coordinates.")
        return None
    else:
        # x = A'b, A'=[b2 -b1; -a2 a1] / (a1*b2-a2*b1)
        x_o = (c1*b2-c2*b1)/(a1*b2-a2*b1)
        y_o = (-c1*a2+c2*a1)/(a1*b2-a2*b1)

    # translate camera coordinate to origin at intersection frame
    # opencv uses top-left origin with increasing x and y to right and down
    # field should increase to the left for x and increase down for y
    x = -(x_c - x_o)
    y = y_c - y_o

    # project point onto either line
    # line direction dot coordinate / norm of line direction
    norm1 = math.sqrt(a1*a1 + b1*b1)
    norm2 = math.sqrt(a2*a2 + b2*b2)

    return ((b1*x-a1*y)/norm1,(b2*x-a2*y)/norm2)


### FUNCTIONS TO BE USED EXTERNALLY ###
def get_coordinates(img, objects):
    # tuning params
    max_lines = 8
    ransac_thresh=2.0
    ransac_iter=2000
    min_inliers=30

    # green to get the field only
    green_mask = mask_field_green(img)

    # use findContours to get the points of external contours (should be close to the field)
    boundary_pts = get_field_boundary_points(green_mask)

    # get candidate lines using RANSAC (returned as (line, inlier_pts, len(inliers)))
    candidate_lines = get_candidate_lines(boundary_pts, max_lines, ransac_thresh,
                                     ransac_iter,min_inliers)
    
    # just keep lines from candidates for now # TODO subject to change
    lines = [line for (line, inliers, coutn) in candidates]

    # get the two most voted for lines to keep as boundary lines 
    final_line1, final_line2 = group_lines(lines)

    # return the passed objects' field coordinate
    field_coords = []

    for object in objects:
        field_coords.append(get_field_coordinate(final_line1, final_line2, object))

    return field_coords

#######################################

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

    # just keep lines for now # TODO subject to change
    lines = [line for (line, inliers, coutn) in candidates]

    final_line1, final_line2 = group_lines(lines)

    H, W = img.shape[:2]

    img_out = img.copy()
    for (a, b, c) in [final_line1, final_line2]:

        # Compute intersection with image borders
        pts = []
        # left (x=0)
        if abs(b) > 1e-6:
            y = -(c + a*0) / b
            if 0 <= y < H: pts.append((0, int(y)))
        # right (x=W-1)
        x = W-1
        if abs(b) > 1e-6:
            y = -(c + a*x) / b
            if 0 <= y < H: pts.append((W-1, int(y)))
        # top (y=0)
        if abs(a) > 1e-6:
            x = -(c + b*0) / a
            if 0 <= x < W: pts.append((int(x), 0))
        # bottom (y=H-1)
        y = H-1
        if abs(a) > 1e-6:
            x = -(c + b*y) / a
            if 0 <= x < W: pts.append((int(x), H-1))

        if len(pts) == 2:
            cv2.line(img_out, pts[0], pts[1], (0, 0, 255), 2)

    cv2.imshow("Final Lines", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()