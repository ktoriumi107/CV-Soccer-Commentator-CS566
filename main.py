from collections import defaultdict, deque
import math
import os
import cv2
import easyocr
import time
import numpy as np
from ultralytics import YOLO
import field

# =======================
# Config
# =======================
VIDEO_PATH = "soccer_vid_flipped.mp4"     # or "webcam" / 0
USE_GPU    = False                        # True if you installed CUDA torch & want GPU OCR

# Player detection & size thresholds
CONF_PLAYER = 0.40
MIN_BBOX_AREA_FRAC  = 0.006               # allow OCR on smaller players

# OCR cadence & quality gating
OCR_EVERY_N_FRAMES  = 4                   # run OCR per-track every N frames (near ball OR unlabeled)
OCR_CONF_MIN        = 0.50                # EasyOCR confidence floor per read
LEGIBILITY_MIN      = 0.60                # minimum legibility score for trying OCR
LEGIBILITY_WEIGHTS  = "models/legibility.pth"  # optional legibility model; proxy used if missing

# Label smoothing / switching
LABEL_WINDOW_SIZE   = 5
LABEL_MIN_CONSENSUS = 2
LABEL_TTL_FRAMES    = 50                  # used if sticky lock is disabled
LABEL_SHOW_CONF_MIN = 0.55
DISPLAY_SWITCH_CONF_MIN = 0.65            # show jersey if (stable and conf >= this)

# Sticky jersey behavior (no revert to ID after lock)
STICKY_JERSEY_AFTER_LOCK = True           # once we lock a number, keep showing it
CHANGE_GUARD_VOTES       = 4              # require this many strong votes for a *different* number to switch

# Ball proximity gating (to prioritize OCR for likely-possessing players)
NEAR_BALL_RADIUS_PX = 180

# Detector
DETECT_CONF   = 0.25
DETECT_IOU    = 0.70
DETECT_IMGSZ  = 1280
DETECT_MAXDET = 300
DETECT_CLASSES = [0, 32]  # person (0), sports ball (32)

# =======================
# Helpers
# =======================
def bbox_area(b): return b[2]*b[3]
def bbox_center(b): return (b[0]+b[2]/2, b[1]+b[3]/2)
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def _clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W-1))
    x2 = max(0, min(int(x2), W-1))
    y1 = max(0, min(int(y1), H-1))
    y2 = max(0, min(int(y2), H-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def prefilter_crop(crop):
    """ Unsharp + CLAHE on luminance to help OCR. """
    if crop is None or crop.size == 0:
        return crop
    blur = cv2.GaussianBlur(crop, (0,0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(crop, 1.6, blur, -0.6, 0)
    yuv = cv2.cvtColor(sharp, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def torso_crop(frame, bbox_xywh, mode="full", return_box=False):
    """
    Return the whole player box (no shift or banding), resized + prefiltered.
    """
    H, W = frame.shape[:2]
    x, y, w, h = map(int, bbox_xywh)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (None, (x1, y1, x2, y2)) if return_box else None

    target_h = 128
    scale = target_h / max(1, crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), target_h), interpolation=cv2.INTER_CUBIC)
    crop = prefilter_crop(crop)
    return (crop, (x1, y1, x2, y2)) if return_box else crop

def legibility_proxy(crop):
    if crop is None or crop.size == 0: return 0.0
    if min(crop.shape[:2]) < 28: return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.clip(g.std()/64.0, 0, 1))

def crop_around(frame, cx, cy, size=256):
    H, W = frame.shape[:2]
    x1 = max(0, int(cx - size//2))
    y1 = max(0, int(cy - size//2))
    x2 = min(W, x1 + size)
    y2 = min(H, y1 + size)
    return frame[y1:y2, x1:x2], (x1, y1)

def foot_point(bbox):
    x,y,w,h = bbox
    return (x + w/2.0, y + h*0.95)

# --- OCR helpers (multi-variant EasyOCR; same pipeline as your image path) ---
def _upsample_and_pad(bgr, scale=2.5, pad=8):
    h, w = bgr.shape[:2]
    up = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))

def _adaptive_bin(gray):
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 9)
    return th

def _contrast_stretch(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _make_variants(crop_bgr):
    base = _upsample_and_pad(prefilter_crop(crop_bgr), scale=2.5, pad=10)
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    g_cs = _contrast_stretch(g)
    th = _adaptive_bin(g_cs)
    inv = cv2.bitwise_not(g_cs)
    th_inv = _adaptive_bin(inv)

    def to_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    variants = [
        ("rgb_base", to_rgb(base)),
        ("rgb_cs",   to_rgb(cv2.cvtColor(g_cs, cv2.COLOR_GRAY2BGR))),
        ("rgb_th",   to_rgb(cv2.cvtColor(th,   cv2.COLOR_GRAY2BGR))),
        ("rgb_inv",  to_rgb(cv2.cvtColor(inv,  cv2.COLOR_GRAY2BGR))),
        ("rgb_thi",  to_rgb(cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR))),
    ]
    return variants

def _extract_best_digits(results):
    # results: [(bbox, text, conf), ...]
    cands = []
    for _, text, conf in results or []:
        if not text:
            continue
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if 1 <= len(digits) <= 3:
            cands.append((digits, float(conf)))
    if not cands:
        return None
    return max(cands, key=lambda t: t[1])  # (digits, conf)

def read_jersey_easyocr(crop_bgr, reader, verbose=True):
    """
    Try multiple preproc variants + permissive thresholds.
    Returns (digits or "", conf, variant_name)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, "none"

    tried = []
    for name, rgb in _make_variants(crop_bgr):
        results = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            low_text=0.25,
            text_threshold=0.5,
            link_threshold=0.5,
            allowlist="0123456789"
        )
        best = _extract_best_digits(results)
        tried.append((name, results, best))
        if best:
            if verbose:
                print(f"[OCR] {name} -> digits={best[0]} conf={best[1]:.3f}")
            return best[0], best[1], name

    if verbose:
        for name, results, best in tried:
            print(f"[OCR] {name} -> none; raw={results}")
    return "", 0.0, "none"

def read_jersey_any(crop_bgr, reader, prefer="easyocr", verbose=True):
    """
    EasyOCR-only wrapper (multi-variant). Returns (digits, conf, tag).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, "none"
    # Use the same multi-pass you used for images
    return read_jersey_easyocr(crop_bgr, reader, verbose=verbose)

# =======================
# Sticky Number Cache (no revert to ID after lock)
# =======================
class NumberCache:
    """
    Sticky jersey cache.
    - First, we "lock" a jersey number using a small consensus (LABEL_MIN_CONSENSUS).
    - If STICKY_JERSEY_AFTER_LOCK=True, we KEEP showing it even when OCR is missing.
    - We only change the locked number if a *different* number earns CHANGE_GUARD_VOTES
      with confidence >= OCR_CONF_MIN (hysteresis).
    """
    def __init__(self,
                 window_size=LABEL_WINDOW_SIZE,
                 min_consensus=LABEL_MIN_CONSENSUS,
                 ttl_frames=LABEL_TTL_FRAMES,            # still used if stickiness is off
                 show_conf_min=LABEL_SHOW_CONF_MIN):
        self.window_size   = window_size
        self.min_consensus = min_consensus
        self.ttl_frames    = ttl_frames
        self.show_conf_min = show_conf_min

        self.votes   = defaultdict(lambda: deque(maxlen=self.window_size))  # tid -> deque[(num, conf)]
        self.counts  = defaultdict(lambda: defaultdict(int))                 # tid -> {num -> count}
        self.stable  = {}     # tid -> (num, conf, last_frame)
        self.locked  = set()  # tids that are sticky-locked
        self.last_seen = defaultdict(lambda: -10**9)  # tid -> last frame with any OCR

    def is_stable(self, tid, fidx):
        if tid not in self.stable:
            return False
        num, conf, last = self.stable[tid]
        if STICKY_JERSEY_AFTER_LOCK and (tid in self.locked):
            return True  # locked numbers are always considered stable/visible
        return (fidx - last) <= self.ttl_frames and conf >= self.show_conf_min

    def update(self, tid, num, conf, fidx):
        if not num or not num.isdigit() or not (1 <= len(num) <= 3):
            return
        conf = float(conf)
        if conf < OCR_CONF_MIN:
            return

        self.votes[tid].append((num, conf))
        self.counts[tid][num] += 1
        self.last_seen[tid] = fidx

        # If already locked to another number, require guard votes to switch.
        if tid in self.locked and tid in self.stable:
            locked_num, locked_conf, _ = self.stable[tid]
            if num != locked_num:
                comp_votes = sum(1 for (n, c) in self.votes[tid] if n == num and c >= OCR_CONF_MIN)
                if comp_votes >= CHANGE_GUARD_VOTES:
                    # switch lock
                    new_conf = np.mean([c for (n,c) in self.votes[tid] if n==num]) if any(n==num for (n,_) in self.votes[tid]) else conf
                    self.stable[tid] = (num, float(min(0.99, max(0.45, new_conf))), fidx)
            else:
                # reinforce locked label confidence
                self.stable[tid] = (locked_num, float(min(0.99, max(locked_conf, conf))), fidx)
            return

        # Not locked yet → look for initial consensus to lock
        best_num = None
        best_pair = (-1, -1.0)  # (count, avg_conf)
        for n, cnt in self.counts[tid].items():
            confs = [c for (nn, c) in self.votes[tid] if nn == n]
            avgc = sum(confs)/len(confs) if confs else 0.0
            cand = (cnt, avgc)
            if cand > best_pair:
                best_pair = cand
                best_num = n

        if best_num is not None and best_pair[0] >= self.min_consensus:
            locked_conf = float(min(0.99, max(0.45, best_pair[1])))
            self.stable[tid] = (best_num, locked_conf, fidx)
            if STICKY_JERSEY_AFTER_LOCK:
                self.locked.add(tid)  # become sticky

    def lookup(self, tid, fidx):
        """
        Returns: (text, conf, is_stable)
        Sticky behavior: if locked, always return the locked jersey (no TTL decay).
        """
        if tid in self.stable:
            num, conf, last = self.stable[tid]
            if STICKY_JERSEY_AFTER_LOCK and (tid in self.locked):
                return num, conf, True
            if (fidx - last) <= self.ttl_frames and conf >= self.show_conf_min:
                return num, conf, True

        # No lock yet → show most recent decent vote with a light decay (optional)
        if self.votes[tid]:
            num, conf = self.votes[tid][-1]
            frames_since = fidx - self.last_seen[tid]
            decayed = max(0.0, conf - frames_since / (4.0 * self.ttl_frames))
            if decayed >= self.show_conf_min:
                return num, decayed, False

        return "", 0.0, False

# =======================
# Team color classifier (2-team, torso-jersey-color-based, frozen team colors)
# =======================
class TeamColorModel:
    """
    2-team colour model using *torso crop* HSV colours.

    - For each tid, keeps an EMA of a jersey-focused HSV mean from the torso crop.
    - The first jersey colour we see becomes Team 1's reference.
    - The first sufficiently different jersey colour becomes Team 2's reference.
    - After both references are set, we DO NOT update them anymore (no drift).
    - For each tid, team is chosen by nearest reference + per-tid majority voting
      to avoid flickering.
    """
    def __init__(self,
                 diff_thresh=25.0,   # distance required to consider a second team colour
                 ema_alpha=0.4,      # EMA smoothing for per-tid colour
                 vote_window=8,      # how many recent frames to vote over per tid
                 min_votes=3):       # minimum votes for a team to become / stay stable
        self.diff_thresh = float(diff_thresh)
        self.ema_alpha   = float(ema_alpha)
        self.vote_window = int(vote_window)
        self.min_votes   = int(min_votes)

        self.tid_color = {}   # tid -> np.array([H,S,V])
        self.team_refs = {}   # team_id -> np.array([H,S,V]), frozen once set
        self.tid_votes = defaultdict(lambda: deque(maxlen=self.vote_window))  # tid -> deque[team_id]
        self.tid_team  = {}   # tid -> stable team_id

    def _mean_hsv_from_torso(self, crop_bgr):
        """
        Compute a jersey-focused mean HSV from the torso crop:
        - Central band of the crop (avoid shorts/grass/arms).
        - Only reasonably saturated pixels (avoid grey/white/background).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        if h < 4 or w < 4:
            return None

        # Central 60% region
        y0, y1 = int(0.2 * h), int(0.8 * h)
        x0, x1 = int(0.2 * w), int(0.8 * w)
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        # Keep only reasonably saturated pixels (jersey is usually coloured, not grey)
        sat = roi[:, :, 1]
        mask = sat > 40
        if np.any(mask):
            region = roi[mask]
        else:
            # Fallback if jersey is low-sat / washed out
            region = roi.reshape(-1, 3)

        mean_hsv = region.mean(axis=0).astype(np.float32)
        return mean_hsv

    def update(self, tid, torso_crop_bgr):
        """
        Update the per-tid jersey colour estimate from the torso crop.
        Call this once per frame for each visible tid.
        """
        m = self._mean_hsv_from_torso(torso_crop_bgr)
        if m is None:
            return

        if tid in self.tid_color:
            self.tid_color[tid] = (
                (1.0 - self.ema_alpha) * self.tid_color[tid] +
                self.ema_alpha * m
            )
        else:
            self.tid_color[tid] = m

    def _maybe_init_team_refs(self, c):
        """
        Initialise up to 2 team reference colours using this colour c.
        Once both are set, they are frozen.
        """
        # No team yet: this becomes Team 1
        if 1 not in self.team_refs:
            self.team_refs[1] = c.copy()
            return

        # Only Team 1 exists: check if c is far enough to be Team 2
        if 2 not in self.team_refs:
            d = np.linalg.norm(c - self.team_refs[1])
            if d > self.diff_thresh:
                self.team_refs[2] = c.copy()
            # If it's not far enough, we keep only Team 1 for now

    def get_team(self, tid):
        """
        Return a *stable* team ID (1 or 2) for this tid, or None if unknown.

        Uses:
        - Frozen global team_refs (once both are set).
        - Per-tid vote buffer to smooth assignments.
        """
        if tid not in self.tid_color:
            return self.tid_team.get(tid, None)

        c = self.tid_color[tid]

        # Initialise team refs if needed (and still possible).
        self._maybe_init_team_refs(c)

        # If still no team references, we can't classify yet.
        if 1 not in self.team_refs:
            return self.tid_team.get(tid, None)

        # Compute raw team assignment by nearest fixed reference.
        if 2 not in self.team_refs:
            # Only one team known so far -> everyone is Team 1 for now.
            raw_team = 1
        else:
            d1 = np.linalg.norm(c - self.team_refs[1])
            d2 = np.linalg.norm(c - self.team_refs[2])
            raw_team = 1 if d1 <= d2 else 2

        votes = self.tid_votes[tid]
        votes.append(raw_team)

        c1 = votes.count(1)
        c2 = votes.count(2)

        # If we already have a stable team, be conservative about switching.
        if tid in self.tid_team:
            current = self.tid_team[tid]
            other   = 1 if current == 2 else 2
            curr_count  = c1 if current == 1 else c2
            other_count = c2 if current == 1 else c1

            # Keep current if still well-supported.
            if curr_count >= self.min_votes and curr_count >= other_count:
                return current

            # Switch only if the other team clearly dominates.
            if other_count >= self.min_votes and other_count > curr_count:
                self.tid_team[tid] = other
                return other

            # Otherwise keep current.
            return current

        # No stable team yet -> lock in if there's a clear majority.
        if c1 >= self.min_votes and c1 > c2:
            self.tid_team[tid] = 1
            return 1
        if c2 >= self.min_votes and c2 > c1:
            self.tid_team[tid] = 2
            return 2

        # Not enough evidence yet: return the raw team as a provisional label.
        return raw_team

# =======================
# IMM-lite Ball Tracker (CV + CA KFs) + LK refine
# =======================
class IMMBallTracker:
    """
    IMM-lite: runs 2 motion models (CV & CA) and selects by residual each update.
    Handles abrupt direction changes via innovation-spike resets and temporary noise inflation.
    """
    def __init__(self, fps=30, max_age=20, trail=10, jump_thresh=40.0, boost_frames=6):
        self.fps = max(1, fps)
        self.dt = 1.0 / self.fps
        self.max_age = max_age
        self.jump_thresh = float(jump_thresh)      # pixels; residual norm to trigger hard reset
        self.boost_frames = int(boost_frames)      # how long to inflate process noise after a jump
        self.boost_left = 0

        # Two Kalman filters: CV (4D state) and CA (6D state)
        self.kf_cv = self._init_kf_cv()
        self.kf_ca = self._init_kf_ca()

        self.has_state = False
        self.age = 0
        self.hist = deque(maxlen=trail)
        self.prev_gray = None

        # Active model
        self.active = "CA"  # default to CA (more responsive)
        self.last_meas_conf = 0.0

    def _init_kf_cv(self):
        # State: [x, y, vx, vy]
        kf = cv2.KalmanFilter(4, 2)
        dt = self.dt
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([1e-2, 1e-2, 2e-1, 2e-1]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([6.0, 6.0]).astype(np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        return kf

    def _init_kf_ca(self):
        # State: [x, y, vx, vy, ax, ay]
        kf = cv2.KalmanFilter(6, 2)
        dt = self.dt
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([3e-2, 3e-2, 5e-1, 5e-1, 6.0, 6.0]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([6.0, 6.0]).astype(np.float32)
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
        return kf

    def _set_state(self, x, y):
        self.kf_cv.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
        self.kf_ca.statePost = np.array([[x],[y],[0],[0],[0],[0]], dtype=np.float32)
        self.has_state = True
        self.age = 0
        self.hist.clear()
        self.hist.append((x, y))

    def predict(self):
        if not self.has_state: return None
        pred_cv = self.kf_cv.predict()
        pred_ca = self.kf_ca.predict()
        x_cv, y_cv = float(pred_cv[0]), float(pred_cv[1])
        x_ca, y_ca = float(pred_ca[0]), float(pred_ca[1])
        if self.active == "CV":
            x, y = x_cv, y_cv
        else:
            x, y = x_ca, y_ca
        self.age += 1
        self.hist.append((x, y))
        if self.age > self.max_age:
            self.has_state = False
            return None
        return (x, y)

    def correct(self, x, y, conf=1.0):
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self.has_state:
            self._set_state(x, y)
            self.last_meas_conf = conf
            return

        if self.boost_left > 0:
            self._boost_process_noise(scale=5.0)
            self.boost_left -= 1
        else:
            self._restore_process_noise()

        H_cv = self.kf_cv.measurementMatrix
        H_ca = self.kf_ca.measurementMatrix
        z_cv = H_cv @ (self.kf_cv.statePre if self.kf_cv.statePre is not None else self.kf_cv.statePost)
        z_ca = H_ca @ (self.kf_ca.statePre if self.kf_ca.statePre is not None else self.kf_ca.statePost)
        res_cv = np.linalg.norm((meas - z_cv).ravel())
        res_ca = np.linalg.norm((meas - z_ca).ravel())

        if min(res_cv, res_ca) > self.jump_thresh:
            self._set_state(x, y)
            self.boost_left = self.boost_frames
            self.last_meas_conf = conf
            return

        if res_cv < res_ca:
            self.active = "CV"
            self.kf_cv.correct(meas)
            self.kf_ca.correct(meas)
        else:
            self.active = "CA"
            self.kf_ca.correct(meas)
            self.kf_cv.correct(meas)

        self.age = 0
        self.hist.append((x, y))
        self.last_meas_conf = conf

    def _boost_process_noise(self, scale=5.0):
        self.kf_cv.processNoiseCov = (np.diag([1e-2,1e-2,2e-1,2e-1]) * scale).astype(np.float32)
        self.kf_ca.processNoiseCov = (np.diag([3e-2,3e-2,5e-1,5e-1,6.0,6.0]) * scale).astype(np.float32)

    def _restore_process_noise(self):
        self.kf_cv.processNoiseCov = np.diag([1e-2,1e-2,2e-1,2e-1]).astype(np.float32)
        self.kf_ca.processNoiseCov = np.diag([3e-2,3e-2,5e-1,5e-1,6.0,6.0]).astype(np.float32)

    def refine_with_lk(self, frame_bgr, guess_xy, patch=21):
        if guess_xy is None: return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p0 = np.array([[guess_xy]], dtype=np.float32)  # (1,1,2)
        if self.prev_gray is None:
            self.prev_gray = gray
            return guess_xy
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                               winSize=(patch, patch), maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = gray
        if st is not None and st[0][0] == 1:
            x, y = float(p1[0,0,0]), float(p1[0,0,1])
            return (x, y)
        return guess_xy

    @property
    def is_stale(self):  # predicted but not corrected this frame
        return self.age > 0

    @property
    def center(self):
        if not self.has_state: return None
        if self.active == "CV":
            s = self.kf_cv.statePost
            return (float(s[0]), float(s[1]))
        s = self.kf_ca.statePost
        return (float(s[0]), float(s[1]))

# =======================
# Optional Legibility Net
# =======================
_leg_net = None
DEVICE = "cuda" if USE_GPU else "cpu"

def try_load_legibility():
    global _leg_net
    if not os.path.exists(LEGIBILITY_WEIGHTS): return False
    try:
        import torch, torch.nn as nn
        import torchvision.models as tvm
        import torchvision.transforms as T
        class LegibilityNet(nn.Module):
            def __init__(self, ckpt_path, device):
                super().__init__()
                m = tvm.resnet34(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 1)
                self.model = m.to(device)
                sd = torch.load(ckpt_path, map_location=device)
                self.model.load_state_dict(sd, strict=True)
                self.model.eval()
                self.tf = T.Compose([
                    T.ToPILImage(),
                    T.Resize((224,224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                self.device = device
            @torch.no_grad
            def score(self, crops_bgr):
                if not crops_bgr: return []
                import torch
                xs = [self.tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
                x = torch.stack(xs, 0).to(self.device)
                logits = self.model(x).squeeze(1)
                return torch.sigmoid(logits).float().cpu().numpy().tolist()
        _leg_net = LegibilityNet(LEGIBILITY_WEIGHTS, DEVICE)
        print("[legibility] loaded:", LEGIBILITY_WEIGHTS)
        return True
    except Exception as e:
        print("[legibility] failed:", e)
        _leg_net = None
        return False

def legibility_score(crop):
    if _leg_net is None: return legibility_proxy(crop)
    return _leg_net.score([crop])[0]

# =======================
# Init models
# =======================
det = YOLO("yolov8n.pt")  # try yolov8s.pt if GPU allows
reader = easyocr.Reader(['en'], gpu=USE_GPU)
_ = try_load_legibility()
tracker_cfg = {"tracker": "bytetrack.yaml"}
ball = IMMBallTracker(fps=30, max_age=20, trail=10, jump_thresh=40.0, boost_frames=6)
team_model = TeamColorModel()  # team color model

# =======================
# Main
# =======================
def main():
    cap = cv2.VideoCapture(0 if str(VIDEO_PATH).lower() in ("webcam","0") else VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
    frame_idx = 0
    cache = NumberCache()
    last_ocr_frame = defaultdict(lambda: -9999)

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        # ---------- Detect + Track players & (global) balls ----------
        results = det.track(
            source=frame, persist=True, verbose=False, **tracker_cfg,
            conf=DETECT_CONF, iou=DETECT_IOU, imgsz=DETECT_IMGSZ,
            max_det=DETECT_MAXDET, classes=DETECT_CLASSES
        )
        res = results[0]
        boxes = res.boxes
        ball_cands = []
        players = []

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                clsid = int(b.cls[0].item())
                conf  = float(b.conf[0].item())
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                w, h = x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2

                if clsid == 32 and conf >= 0.20:
                    area = (x2-x1)*(y2-y1)
                    ball_cands.append((conf, -area, cx, cy))
                elif clsid == 0 and conf >= CONF_PLAYER:
                    tid = int(b.id[0].item()) if b.id is not None else -1
                    if tid >= 0:
                        players.append((tid, (int(x1),int(y1),int(w),int(h)), conf))

        box_points = []
        for tid, (x1, y1, w, h), conf in players:
            bx = x1 + w / 2   # horizontal center
            by = y1 + h       # bottom edge
            box_points.append((bx, by))

        show_lines = False

        if show_lines:
            field_coords, lines = field.get_coordinates(frame, box_points, show_lines)
            if lines is not None: 
                with_lines = field.visualize_lines(frame, lines)
                cv2.imshow("Field with lines", with_lines)
        else:
            field_coords, _ = field.get_coordinates(frame, box_points, frame.size)
            field.visualize_points(field_coords)
         
        # ---------- IMM hybrid ball update ----------
        if ball_cands:
            ball_cands.sort(reverse=True)
            conf, _, cx, cy = ball_cands[0]
            ball.correct(cx, cy, conf=conf)
        else:
            ball.predict()

        pred_xy = ball.predict()
        if pred_xy is not None:
            local, (ox, oy) = crop_around(frame, pred_xy[0], pred_xy[1], size=256)
            dets = det.predict(source=local, verbose=False, conf=0.15, iou=0.7, classes=[32], imgsz=320)
            if dets and len(dets[0].boxes) > 0:
                cands = []
                for bb in dets[0].boxes:
                    c = float(bb.conf[0].item())
                    if c < 0.15: continue
                    xx1,yy1,xx2,yy2 = bb.xyxy[0].cpu().numpy()
                    ccx, ccy = ox + (xx1+xx2)/2, oy + (yy1+yy2)/2
                    area = (xx2-xx1)*(yy2-yy1)
                    cands.append((c, -area, ccx, ccy))
                if cands:
                    cands.sort(reverse=True)
                    c, _, ccx, ccy = cands[0]
                    ball.correct(ccx, ccy, conf=c)
            else:
                refined = ball.refine_with_lk(frame, pred_xy, patch=21)
                if refined is not None:
                    ball.correct(refined[0], refined[1], conf=max(0.0, ball.last_meas_conf - 0.1))

        ball_center = ball.center

        # Snap near foot for possession
        if ball_center is not None and players:
            bx, by = ball_center
            best = None
            for tid, bbox, _ in players:
                fx, fy = foot_point(bbox)
                d = math.hypot(fx - bx, fy - by)
                if best is None or d < best[0]:
                    best = (d, fx, fy)
            if best and best[0] < 28:
                nx = 0.3*bx + 0.7*best[1]
                ny = 0.3*by + 0.7*best[2]
                ball.correct(nx, ny, conf=1.0)
                ball_center = (nx, ny)

        # ---------- Draw ball ----------
        if ball_center is not None:
            color = (0,165,255) if ball.is_stale else (0,255,255)
            cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 10, color, -1)
            for i in range(1, len(ball.hist)):
                p1 = (int(ball.hist[i-1][0]), int(ball.hist[i-1][1]))
                p2 = (int(ball.hist[i][0]),   int(ball.hist[i][1]))
                cv2.line(frame, p1, p2, color, 2)

        # ---------- Players: OCR & labels (ID by default → switch to jersey when stable) ----------
        font = cv2.FONT_HERSHEY_SIMPLEX
        active_tids = []
        for tid, bbox, pconf in players:
            active_tids.append(tid)
            x,y,w,h = bbox

            # Torso crop used for team-color estimation + ROI debug box
            crop, roi = torso_crop(frame, (x, y, w, h), return_box=True)
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

            # Update team color model from this torso crop
            team_model.update(tid, crop)
            team_id = team_model.get_team(tid)

            jersey, jconf, stable = cache.lookup(tid, frame_idx)

            # Show ID by default; switch to jersey when stable enough
            show_jersey = bool(jersey) and stable and (jconf >= DISPLAY_SWITCH_CONF_MIN)

            # Include team label if known
            if team_id is not None:
                if show_jersey:
                    label_text = f"Team {team_id} #{jersey}"
                else:
                    label_text = f"Team {team_id} ID {tid}"
            else:
                label_text = f"#{jersey}" if show_jersey else f"ID {tid}"

            color = (0,255,0) if show_jersey else (200,200,200)
            cv2.putText(frame, label_text, (x, y-8), font, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, label_text, (x, y-8), font, 0.8, color, 2, cv2.LINE_AA)

            # Skip OCR for tiny boxes
            if bbox_area(bbox) < MIN_BBOX_AREA_FRAC * (frame.shape[0]*frame.shape[1]):
                continue

            # OCR scheduling: unlabeled tracks aggressively; also near-ball; plus a staggered periodic try.
            has_stable   = cache.is_stable(tid, frame_idx)
            no_label_yet = (jersey == "") and (not has_stable)
            near = (ball_center is not None and dist(bbox_center(bbox), ball_center) <= NEAR_BALL_RADIUS_PX)

            should_ocr = (
                no_label_yet or near or ((frame_idx + tid) % 12 == 0)
            ) and (frame_idx - last_ocr_frame[tid] >= OCR_EVERY_N_FRAMES)

            if not should_ocr:
                continue

            # FULL BODY CROP (matches your image success)
            crop_full = frame[y:y+h, x:x+w]

            leg = legibility_score(crop_full) if crop_full is not None else 0.0
            if leg < (LEGIBILITY_MIN * 0.85) and not no_label_yet:
                continue

            last_ocr_frame[tid] = frame_idx
            digits, pred_conf, tag = read_jersey_any(crop_full, reader, prefer="easyocr", verbose=False)
            if digits:
                cache.update(tid, digits, pred_conf, frame_idx)
                # Uncomment for debug:
                # print(f"[UPDATE] f={frame_idx} tid={tid} => #{digits} (conf={pred_conf:.2f}) via {tag}")

        # ---------- UI ----------
        cv2.imshow("SoccerRT", frame)
        frame_idx += 1
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# =======================
# Single-image helper (unchanged, uses same OCR path)
# =======================
def run_on_image(image_path):
    frame = cv2.imread(image_path)
    assert frame is not None, f"Cannot open {image_path}"
    H, W = frame.shape[:2]

    results = det.predict(source=frame, verbose=False, conf=DETECT_CONF,
                          iou=DETECT_IOU, imgsz=DETECT_IMGSZ, classes=DETECT_CLASSES)
    res = results[0]
    boxes = res.boxes
    players = []
    balls = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            clsid = int(b.cls[0].item())
            conf  = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if clsid == 0 and conf >= CONF_PLAYER:
                players.append((int(x1), int(y1), int(w), int(h)))
            elif clsid == 32 and conf >= 0.25:
                balls.append(((int((x1 + x2) / 2), int((y1 + y2) / 2)), conf))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop, roi = torso_crop(frame, (x, y, w, h), return_box=True)
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

        if crop is not None:
            cv2.imwrite("debug_torso_crop.jpg", crop)
        cv2.imwrite("debug_frame_with_roi.jpg", frame)

        leg = legibility_score(crop) if crop is not None else 0.0
        print(f"Legibility score: {leg:.3f}  ROI={roi}")

        label_txt = "ID?"
        color = (200, 200, 200)
        if crop is not None and leg >= LEGIBILITY_MIN * 0.85:
            digits, conf, variant = read_jersey_any(crop, reader, prefer="easyocr", verbose=True)
            print("OCR best:", digits, conf, "via", variant)
            if digits:
                label_txt = f"#{digits}"
                color = (0, 255, 0)

        cv2.putText(frame, label_txt, (x, y - 8), font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label_txt, (x, y - 8), font, 0.8, color, 2, cv2.LINE_AA)

    for (cx, cy), conf in balls:
        cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 255), -1)

    cv2.imshow("Soccer Image", frame)
    cv2.imwrite("ocr_image.jpg", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #run_on_image("0_2.jpg")
    main()
