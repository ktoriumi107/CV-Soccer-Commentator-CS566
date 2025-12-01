import math
import numpy as np
import cv2

avg_past_lines = None
n = 0

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
    # get edges from mask
    edges = cv2.Canny(mask, 70, 150)

    # get the external contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # get the contour with the largest area (should be the exterior of the field)
    contour = max(contours, key=lambda c: cv2.arcLength(c, False))
    
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
        x1,y1 = p1
        x2,y2 = p2

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

def filter_out_border(lines, size):
    H, W, _ = size
    threshold = 10

    non_border = []

    for (a,b,c) in lines:
        pts = []

        # left border when x = 0
        if abs(b) > 1e-6:
            y = -(c + a*0) / b
            pts.append((0, y))

        # right border when x = W
        if abs(b) > 1e-6:
            y = -(c + a*W) / b
            pts.append((W, y))

        # top border when y = 0
        if abs(a) > 1e-6:
            x = -(c + b*0) / a
            pts.append((x, 0))

        # bottom border when y = H
        if abs(a) > 1e-6:
            x = -(c + b*H) / a
            pts.append((x, H))

        # find all intersections with the border
        pts = [(x,y) for x,y in pts 
               if -threshold <= x <= W+threshold and 
                  -threshold <= y <= H+threshold]

        # skip if not enough points
        if len(pts) < 2:
            continue

        # compute midpoint of the segment inside frame
        mx = (pts[0][0] + pts[1][0]) / 2
        my = (pts[0][1] + pts[1][1]) / 2

        # if midpoint near a border, then the entire segment is on the border
        if (mx < threshold or mx > W-threshold or
            my < threshold or my > H-threshold):
            continue

        # midpoint is not near a border, it is safe to append
        non_border.append((a,b,c))

    return non_border

def average_line(lines):
    # get the average line for a group of lines
    a, b, c = np.mean(np.array(lines), axis=0)

    # normalize
    norm = np.hypot(a,b)

    return (a/norm, b/norm, c/norm)

def group_lines(lines, size, angle_diff):
    angle_thresh = np.deg2rad(5)

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

    # sort groups by number of inliers
    group_sizes = [(len(set), i) for i, set in enumerate(sets)]
    group_sizes.sort(reverse=True)

    # get the biggest group
    _, group1 = group_sizes[0]
    a1,b1,c1 = average_line(sets[group1])

    # get the next biggest group that is close to perpendicular
    i = 1 # index of group
    
    # go through all groups
    while i < len(group_sizes):
        _, curr_group = group_sizes[i]
        a_curr, b_curr, c_curr = average_line(sets[curr_group])

        # choose second group if the angle difference is big enough
        difference = abs(math.atan2(b_curr, a_curr)-math.atan2(b1,a1))
        if math.degrees(min(math.pi-difference, difference)) >= angle_diff:
            return ((a1,b1,c1),(a_curr,b_curr,c_curr))
        
        i += 1
            
    # end of groups reached, return perpendicular line
    # TODO come up with better value for c2
    H, W = size

    # have line 2 cross the center of the screen
    c2 = -(-b1*W/2 + a1*H/2)

    return ((a1,b1,c1),(-b1, a1, c2))

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

    x_o = None
    y_o = None

    # origin as intersection of two points
    # intersection of two lines, group as Ax=b: [a1 b1;a2 b2][x; y]=[-c1;-c2]
    # check A' is invertible
    if a1*b2 - a2*b1 < 1e-5:
        print("First line", a1,b1,c1)
        print("Second line", a2,b2,c2)
        print("Nearly parallel lines. Failed to get coordinates.")
        return None
    else:
        # x = A'b, A'=[b2 -b1; -a2 a1] / (a1*b2-a2*b1)
        print("Got coordinates")
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
def get_coordinates(img, objects, show_lines):
    # tuning params
    max_lines = 4
    ransac_thresh=2.0
    ransac_iter=2000
    min_inliers=30

    # green to get the field only
    green_mask = mask_field_green(img)

    # use findContours to get the points of external contours (should be close to the field)
    boundary_pts = get_field_boundary_points(green_mask)

    # get candidate lines using RANSAC (returned as (line, inlier_pts, len(inliers)))
    candidates = get_candidate_lines(boundary_pts, max_lines, ransac_thresh,
                                     ransac_iter,min_inliers)
    
    # filter candidates
    filtered_lines = []

    for line, inliers, count in candidates:
        # only counts lines with enough votes
        if count > 10:
            filtered_lines.append(line)

    # filter out border
    filtered_lines = filter_out_border(filtered_lines, size=img.shape)

    # get the two most voted for lines to keep as boundary lines 
    final_line1, final_line2 = group_lines(filtered_lines, size=img.shape)        

    # declare use of global vars
    global avg_past_lines
    global n

    # update history of lines
    if avg_past_lines is None:
        avg_past_lines = final_line1
        n = 1
    else:
        # refuse new values if they deviate too far from historyical average
        # TODO makes field system fail entirely given a camera change
        difference = 100

        # compute differences for each line
        diff1 = np.linalg.norm(np.array(avg_past_lines) - np.array(final_line1))

        # update only if difference is below threshold
        final_line1 = final_line1 if diff1 < difference else avg_past_lines

        # update history
        n += 1
        avg1 = (np.array(avg_past_lines) * (n-1) + np.array(final_line1)) / n
        avg_past_lines = avg1

    # return the passed objects' field coordinate
    field_coords = []

    for object in objects:
        field_coords.append(get_field_coordinate(final_line1, final_line2, object))

    if show_lines:
        return field_coords, (final_line1, final_line2)
    else: 
        return field_coords

def visualize_points(points):
    color=(0,0,255)
    radius=10

    W, H = (1000, 1000)
    field = np.zeros((H, W, 3), dtype=np.uint8)

    # remove Nones
    points = [p for p in points if p is not None]
    if not points:
        return field

    u_coords, v_coords = zip(*points)
    u_min, u_max = min(u_coords), max(u_coords)
    v_min, v_max = min(v_coords), max(v_coords)

    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)

    for u, v in points:
        # normalize to canvas coordinates
        x = int((u - u_min) / u_span * (W - 1))
        y = H - 1 - int((v - v_min) / v_span * (H - 1)) 
        cv2.circle(field, (x, y), radius, color, -1)
    return field

def visualize_lines(frame, lines):
    H, W = frame.shape[:2]
    result = frame.copy()

    color = (0, 0, 255)
    
    for (a,b,c) in lines:
        pts = []

        # left border (x=0)
        if abs(b) > 1e-6:
            y = int(-(c + a*0) / b)
            if 0 <= y < H:
                pts.append((0, y))

        # right border (x=W-1)
        if abs(b) > 1e-6:
            y = int(-(c + a*(W-1)) / b)
            if 0 <= y < H:
                pts.append((W-1, y))

        # top border (y=0)
        if abs(a) > 1e-6:
            x = int(-(c + b*0) / a)
            if 0 <= x < W:
                pts.append((x, 0))

        # bottom border (y=H-1)
        if abs(a) > 1e-6:
            x = int(-(c + b*(H-1)) / a)
            if 0 <= x < W:
                pts.append((x, H-1))

        # draw line if we got two points
        if len(pts) >= 2:
            cv2.line(result, pts[0], pts[1], color, 3)
    return result
#######################################

if __name__ == "__main__":
    # tuning params
    max_lines = 8
    ransac_thresh=20
    ransac_iter=3000
    min_inliers=10
    angle_diff = 3 # degrees

    img = cv2.imread("test_images/oblique_field_view.png")

    green_mask = mask_field_green(img)
    cv2.imwrite("debug_green_mask.png", green_mask)

    boundary_pts = get_field_boundary_points(green_mask)

    candidates = get_candidate_lines(boundary_pts, max_lines, ransac_thresh,
                                     ransac_iter,min_inliers)

    # just keep lines for now # TODO subject to change
    lines = [line for (line, inliers, coutn) in candidates]

    final_line1, final_line2 = group_lines(lines, img.shape[:2], angle_diff)

    H, W = img.shape[:2]

    img_out = green_mask.copy()
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