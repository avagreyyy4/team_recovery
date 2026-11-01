# -*- coding: utf-8 -*-
"""Court-line and player tracking with SAM 3 (local / Quest runner).

Tracks every huddle clip of a matchup and writes <features-out-root>/<matchup>/all_sets.csv.
Clips are read from <drive-root>/<matchup>/huddles/huddle_<n>_set_<m>.mp4; per-game config
comes from analysis/modeling/parameters.csv.

Usage:
    python3 analysis/modeling/sam3_automation.py --matchup Calvin_vs_Hope
    python3 analysis/modeling/sam3_automation.py                    # every matchup

Needs a GPU, torch/sam3, and HF_TOKEN (SAM 3 weights are gated on Hugging Face).
Per-huddle results are checkpointed, so reruns only redo missing work.
"""

import argparse
import gc
import glob
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

# stimulus_map.py lives in analysis/feature_extraction/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_extraction'))
import stimulus_map

REPO_ROOT = stimulus_map.REPO_ROOT
DEFAULT_DRIVE_ROOT = Path(os.environ.get(
    "COMPUTER_VISION_DRIVE_ROOT", str(REPO_ROOT / "output" / "Computer_Vision")))
# Separate from DEFAULT_DRIVE_ROOT (above) - that one's for reading the large raw huddle clips (wherever
# COMPUTER_VISION_DRIVE_ROOT points, e.g.
DEFAULT_FEATURES_ROOT = REPO_ROOT / "data"
DEFAULT_PARAMETERS_CSV = REPO_ROOT / "analysis" / "modeling" / "parameters.csv"

PACKAGES = [
    "pandas",
    "huggingface_hub",
    "Pillow",
    "psutil",
    "scipy==1.13.1",
    "scikit-image==0.24.0",
    "torch",
    "torchvision",
    "opencv-python-headless",
    "matplotlib",
    "git+https://github.com/facebookresearch/sam3.git",
    "numpy==1.26.4",          # LAST -- sam3 needs numpy<2, and several of the
                              # packages above will pull numpy 2.x if left to it
]


def install_dependencies():
    for pkg in PACKAGES:
        print(f"Installing {pkg}...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                           capture_output=True, text=True)
        print("  ok" if r.returncode == 0 else f"  FAILED: {r.stderr[-300:]}")
    import numpy
    print("numpy:", numpy.__version__, "(must be 1.26.x)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--matchup", help="Only process matchups whose folder name contains this substring. "
                           "Comma-separate multiple substrings to select several matchups at once "
                           "(default: every matchup in stimulus_map.csv)",
    )
    parser.add_argument(
        "--sets", help="Comma-separated set numbers to (re)run, e.g. 1,2,3 "
                        "(default: 1..number_sets from parameters.csv, skipping sets with no clips yet)",
    )
    parser.add_argument(
        "--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT,
        help="Folder containing <matchup>/huddles/ clips (input) -- same convention as "
             "the COMPUTER_VISION_DRIVE_ROOT env var",
    )
    parser.add_argument(
        "--features-out-root", type=Path, default=DEFAULT_FEATURES_ROOT,
        help="Folder to write <matchup>/all_sets.csv tracking output into (default: "
             "this repo's data/ folder) -- deliberately independent of --drive-root, "
             "which is only for reading the (much larger) input clips",
    )
    parser.add_argument("--parameters-csv", type=Path, default=DEFAULT_PARAMETERS_CSV)
    parser.add_argument("--stimulus-map", type=Path, default=stimulus_map.DEFAULT_STIMULUS_MAP_PATH)
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token; falls back to $HF_TOKEN, then an interactive login() prompt",
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="pip install the torch/sam3/etc. dependencies, then exit",
    )
    parser.add_argument(
        "--debug-homography-frame", default=None,
        help="Debug: after full model init, run just the line-detection step on "
             "frame 0 of this video path, print what it found, then exit.",
    )
    parser.add_argument(
        "--repeat-huddle", type=Path, default=None,
        help="Repeatability check: run SAM3 on this single huddle clip --repeat-n "
             "times and report how much the resulting formation features vary "
             "across runs (stochasticity check). Skips the normal matchup loop.",
    )
    parser.add_argument(
        "--repeat-n", type=int, default=5,
        help="Number of times to (re)run --repeat-huddle (default: 5)",
    )
    parser.add_argument(
        "--repeat-out-dir", type=Path, default=None,
        help="Where to save repeatability outputs (default: "
             "<clip's matchup dir>/huddle_features/_repeatability/<clip stem>/)",
    )
    parser.add_argument(
        "--jitter-homography", type=Path, default=None,
        help="Calibration-jitter check: track this single STATIC_CAMERA huddle "
             "clip once, then for --jitter-n draws add random pixel noise to the "
             "8 detected court-registration points, refit the homography, and "
             "reproject the SAME tracked player positions through it (no further "
             "SAM3 calls). Reports mean/SD of gyration_closeness and the other "
             "closeness metrics across draws -- isolates how much per-video "
             "calibration noise alone moves the meter measurements, decoupled "
             "from SAM3's own run-to-run tracking stochasticity (see "
             "--repeat-huddle for that). Skips the normal matchup loop.",
    )
    parser.add_argument(
        "--jitter-n", type=int, default=30,
        help="Number of jitter draws (default: 30)",
    )
    parser.add_argument(
        "--jitter-px", type=float, default=2.5,
        help="Max pixel jitter added to each registration point, independently "
             "per x/y, drawn uniformly from [-px, +px] (default: 2.5, i.e. the "
             "middle of the requested +-2-3px range)",
    )
    parser.add_argument(
        "--jitter-out-dir", type=Path, default=None,
        help="Where to save jitter-check outputs (default: "
             "<clip's matchup dir>/huddle_features/_jitter/<clip stem>/)",
    )
    parser.add_argument(
        "--jitter-seed", type=int, default=None,
        help="Seed for the jitter RNG (default: unseeded/random each run)",
    )
    parser.add_argument(
        "--rerun-below-points", type=int, default=None,
        help="Before the normal matchup loop, delete any existing checkpoint "
             "(for --matchup, or every STATIC_CAMERA matchup if --matchup is "
             "omitted) whose huddle re-detects fewer than this many of the 8 "
             "base court-registration points on its own calibration frames "
             "(cheap line-detection only, no SAM3 tracking). The normal loop "
             "then reprocesses just those huddles, now benefiting from the "
             "last-good-homography fallback. No-op for non-STATIC_CAMERA "
             "matchups. E.g. --rerun-below-points 8 to rerun anything that "
             "didn't get the full 8.",
    )
    return parser.parse_args()


def hf_login(token=None):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/xet"), ignore_errors=True)
    from huggingface_hub import login
    token = token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF_TOKEN not set -- falling back to interactive login().")
        login()


def load_sam3_image_model():
    import sam3 as _sam3_pkg
    from sam3 import build_sam3_image_model
    sam3_pkg_dir = os.path.dirname(_sam3_pkg.__file__)
    bpe_matches = glob.glob(os.path.join(sam3_pkg_dir, "**", "bpe_simple_vocab_16e6.txt.gz"), recursive=True)
    print("bpe found:", bpe_matches)
    model = build_sam3_image_model(bpe_path=bpe_matches[0], load_from_HF=True)
    print("Model loaded.")
    return model


def patch_sam3_propagation_classifier():
    # Works around a bug in sam3's parse_action_history_for_propagation: it
    # only returns "propagation_full" (the path that caches results before
    # reading them) when a session's action_history is completely empty. But
    # any real usage always adds prompts (points/text) before propagating, so
    # action_history is never empty -- every session's FIRST-EVER propagation
    # gets misrouted into "propagation_partial" (a path meant for incremental
    # updates *after* a full propagation has already cached results), which
    # then crashes with "No cached outputs found." Confirmed against sam3's
    # own official example notebook, which hits the identical bug -- this
    # isn't a misuse on our end. Fix: treat "no propagation action has
    # happened yet in this session" as the trigger for propagation_full,
    # instead of "no actions at all happened yet."
    from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity

    def parse_action_history_for_propagation(self, inference_state):
        action_history = inference_state["action_history"]
        if len(action_history) == 0:
            return "propagation_full", None
        if "propagation" in action_history[-1]["type"]:
            if action_history[-1]["type"] in ["propagation_fetch"]:
                return "propagation_fetch", None
            elif action_history[-1]["type"] in ["propagation_partial", "propagation_full"]:
                if (
                    len(action_history) > 1
                    and action_history[-2]["type"] in ["propagation_partial", "propagation_full"]
                ) or action_history[-1]["frame_idx"] in [0, inference_state["num_frames"] - 1]:
                    return "propagation_fetch", None
                else:
                    return action_history[-1]["type"], action_history[-1]["obj_ids"]
        if not any("propagation" in a["type"] for a in action_history):
            return "propagation_full", None
        obj_ids = []
        for action in action_history[::-1]:
            if "propagation" in action["type"]:
                break
            if action["type"] in ["add", "refine"]:
                obj_ids.extend(action["obj_ids"])
        obj_ids = list(set(obj_ids)) if len(obj_ids) > 0 else None
        propagation_type = "propagation_partial" if obj_ids is not None else "propagation_fetch"
        return propagation_type, obj_ids

    Sam3VideoInferenceWithInstanceInteractivity.parse_action_history_for_propagation = (
        parse_action_history_for_propagation
    )
    print("Patched sam3 propagation-type classifier "
          "(fixes 'No cached outputs found' on a session's first propagation).")


# ══════════════════════════════════════════════════════════════════════════════
# Everything below assumes numpy/pandas/torch/cv2/PIL/scipy/skimage/sam3 are
# already importable -- they're imported at the bottom, inside
# `if __name__ == "__main__":`, so `--install-deps` and `--help` work even
# before those packages exist.
# ══════════════════════════════════════════════════════════════════════════════

# ── CAMERA CONFIG (defaults; overridden per matchup from parameters.csv) ──
SET_NUMBER = None
STATIC_CAMERA = True
STATIC_PROBE_FRAMES = [0, 7, 14]

# Last successfully-computed static-camera homography, reused when a later
# huddle's own calibration frames fail to detect enough court lines (the
# camera genuinely doesn't move on a STATIC_CAMERA game, so an old homography
# is still valid -- better than silently dropping the whole huddle). Reset
# per matchup in process_game() so it never leaks across games.
LAST_STATIC_H = None
LAST_STATIC_NAMED = None

# True while processing a matchup in MANUAL_HOMOGRAPHY_MATCHUPS -- makes
# prepare_huddle_frames skip auto-detection entirely instead of letting it
# compete with (and potentially wrongly override) the manual points. Set/reset
# per matchup in process_game().
FORCE_MANUAL_HOMOGRAPHY = False

# ── LINE-DETECTION CONFIG ──
FRAME_STEP, MAX_FRAMES = 7, 120
CONF, MAX_LINES, MIN_MASK_PX, IOU_THRESH, MIN_EXT_PX = 0.4, 6, 500, 0.5, 20
ANGLE_TOL_DEG, MAX_EXTRAP_PX, SAME_ANGLE_DEG, SAME_LINE_PX, DUP_PX = 25, 300, 12, 40, 15
SIDELINE_MAX_DEG, ENDLINE_MAX_DEG = 20, 55

# ── PLAYER CONFIG ──
SAM3_PROMPT = "female volleyball player playing"
COURT_MARGIN, TRACK_MARGIN = 0.5, 0.5
COURT_W, COURT_H, NET_X = 18.0, 9.0, 9.0
EXPECTED_PER_SIDE = 6
DEDUP_M      = 0.5
MIN_MASK_AREA = 500     # px; masks below this are noise, skipped everywhere
COURT_COORDS_MAP = {
    "corner_right_far":[18,9],"corner_right_close":[18,0],"corner_left_far":[0,9],"corner_left_close":[0,0],
    "attack_left_far":[6,9],"attack_left_close":[6,0],"attack_right_far":[12,9],"attack_right_close":[12,0],
    "net_far":[9,9],"net_close":[9,0]}
# net_far/net_close aren't independently detected -- they're the midpoint of the two attack-line points on
# their side (see run_pipeline).
BASE_COURT_KEYS = [k for k in COURT_COORDS_MAP if k not in ("net_far", "net_close")]
FULL_REG_POINTS = len(BASE_COURT_KEYS)

# Manually-labeled fallback for matchups where automatic line-detection can't
# reliably find enough of the 8 BASE_COURT_KEYS (e.g. court markings too faint
# or occluded in this camera's framing). One click-labeled frame, shared as
# the static-camera calibration for every huddle in the matchup -- same court/
# camera setup for both games below, so one set of points covers both.
# build_homography() only needs >=4 of these to produce a usable H.
MANUAL_HOMOGRAPHY_POINTS = {
    "corner_left_far":    [451, 504],
    "corner_right_close": [1774, 761],
    "corner_right_far":   [1448, 494],
    "attack_left_close":  [704, 778],
    "attack_left_far":    [790, 504],
    "attack_right_close": [1245, 772],
    "attack_right_far":   [1122, 500],
}
MANUAL_HOMOGRAPHY_MATCHUPS = {"Ithaca_vs_NYU", "NYU_vs_Eastern"}

# Locked-from-huddle fallback: for STATIC_CAMERA matchups where per-huddle
# auto-detection occasionally finds enough points to win the STATIC_CAMERA
# ratchet in prepare_huddle_frames (see its n_named_base comparison) while
# describing the WRONG court geometry -- not too few points to trust (that
# case already falls back safely), but a confidently wrong correspondence
# that then gets used outright, collapsing on-court player detection to
# ~0 for that huddle. Diagnosed on Ithaca_vs_WS, Ithaca_vs_WilliamPaterson,
# Ithaca_vs_WS_Reg, Smith_vs_NYU, and Swarthmore_vs_Smith, each with several
# huddles showing left_n/right_n <= 1 for the ENTIRE clip scattered across
# every set, not just huddle 1 -- inconsistent with a one-off detection
# failure, consistent with the per-huddle ratchet occasionally losing to a
# bad-but-plausible-point-count detection. Unlike MANUAL_HOMOGRAPHY_POINTS
# (hand-clicked pixel coordinates), this sources a real calibration via
# compute_static_homography() on one specific huddle clip already confirmed
# (by inspecting its tracked output) to find a full/plausible team on both
# sides -- then forces every huddle in the matchup to reuse it, same as
# MANUAL_HOMOGRAPHY_MATCHUPS's skip-detection behavior. Populate with the
# confirmed-good clip's filename once chosen; see process_game() and
# prepare_huddle_frames() for where this is consumed.
LOCKED_HOMOGRAPHY_SOURCE_HUDDLE = {
    # matchup: "huddle_<N>_set_<S>.mp4" -- sourced from that SAME matchup's
    # own huddle clips (chosen from its already-tracked all_sets.csv as the
    # huddle closest to a realistic 6-per-side team count with typical
    # (non-outlier) tightness stats and a long-enough clip (>=100 frames) to
    # give a stable calibration) -- OR (source_matchup, "huddle_<N>_set_<S>.mp4")
    # to borrow an already-verified calibration from a DIFFERENT matchup's
    # huddle folder, for a matchup that shares the same physical camera/court
    # but has no tracked output of its own yet to pick a confirmed-good huddle
    # from (see process_game()'s resolution of this tuple form). Confirm the
    # two matchups really do share a camera before trusting a borrowed entry
    # for a full run -- e.g. via compute_static_homography() on one clip from
    # each and comparing the named pixel points, the same check used to
    # sanity-check candidates before locking one in originally.
    "Ithaca_vs_WS": "huddle_9_set_3.mp4",
    # Ithaca_vs_WilliamPaterson's own-clip pick (huddle_16_set_4.mp4) failed
    # compute_static_homography() when tested in isolation -- it likely
    # tracked well originally by inheriting a working calibration from a
    # different clip via the old per-huddle fallback, not from its own lines
    # being independently detectable. Same court/year/camera as Ithaca_vs_WS,
    # and that calibration is now proven across 4 matchups (itself plus 3
    # successful borrows), so borrow it here too instead of hunting for a
    # second self-sourced candidate.
    "Ithaca_vs_WilliamPaterson": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    "Ithaca_vs_WS_Reg": "huddle_15_set_2.mp4",
    # Smith_vs_NYU and Swarthmore_vs_Smith are also played at Ithaca's court
    # (per direct confirmation, not derivable from parameters.csv's home_team
    # field -- same situation as NYU_vs_Eastern being hosted at Ithaca despite
    # neither team being Ithaca). Borrowing here instead of the previous
    # self-sourced picks, which were never verified against
    # compute_static_homography() in isolation the way Ithaca_vs_WilliamPaterson's
    # was found to fail -- the auto-verify-and-retry mechanism will catch and
    # self-correct if this borrow turns out wrong for either.
    "Smith_vs_NYU": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    "Swarthmore_vs_Smith": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    # Ithaca_vs_Clarkson, NYU_vs_Swarthmore, and UnionCollege_vs_WS are on
    # the same Ithaca-court camera setup as Ithaca_vs_WS/WilliamPaterson/
    # WS_Reg above (Ithaca is the home team for Clarkson; the other two are
    # part of the same tournament hosted there), but had no huddle clips
    # synced yet as of this change, so there's no tracked output of their own
    # to pick a confirmed-good source huddle from -- borrowing Ithaca_vs_WS's
    # already-verified calibration as a starting safety net. Re-verify with
    # compute_static_homography() once each has real clips (not yet confirmed
    # pixel-for-pixel identical to Ithaca_vs_WS's camera framing).
    "Ithaca_vs_Clarkson": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    "NYU_vs_Swarthmore": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    "UnionCollege_vs_WS": ("Ithaca_vs_WS", "huddle_9_set_3.mp4"),
    # Wittenberg_vs_Transylvania
    "Wittenberg_vs_Transylvania": "huddle_43_set_1.mp4",
}

# True while processing a matchup in LOCKED_HOMOGRAPHY_SOURCE_HUDDLE -- same
# role as FORCE_MANUAL_HOMOGRAPHY, kept as a separate flag so the two
# mechanisms stay independently toggleable and MANUAL_HOMOGRAPHY_MATCHUPS's
# existing behavior is untouched. Set/reset per matchup in process_game().
FORCE_LOCKED_HOMOGRAPHY = False

# ── LATE-ENTERING PLAYERS (throwaway-detect + point-inject; NO text reprompt) ──
SCAN_FRAMES    = [30, 45]     # fixed for every matchup, not configured per-game
NEW_IOU_THRESH = 0.20
NEW_MIN_PX     = 500
NEW_DEDUP_M    = 0.7

# ══════════════════════════════════════════════════════════════════════════════
# FOOT ESTIMATION (FINAL -- matches the single-huddle decision, pose removed)
# Foot = mask bottom, UNLESS the mask is SHORT for its depth AND her feet look
# hidden (a mask below her, or a teammate beside reaching lower). Then drop to the
# modeled floor (head + one depth-scaled body). Wide masks (dives / sprawls /
# merged clusters) always use the bottom; the wide gate is checked FIRST and is
# ══════════════════════════════════════════════════════════════════════════════
WIDE_ENTER_ASPECT = 1.9
WIDE_EXIT_ASPECT  = 1.6
WIDE_HYST_FRAMES  = 2
WIDE_DIRECT_OVERLAP_FRAC = 0.7   # see mask_foot_final: a "wide" mask this
                                 # directly covered by another mask is more
                                 # likely bent/entangled in a huddle than diving
HEIGHT_SHORT_FRAC = 0.95    # mask reaching <= this fraction of a full body is "short"

OCC_COVER_FRAC  = 0.50      # % of the bottom edge another mask must cover ("below" test)
OCC_BAND_FRAC   = 0.25      # band depth below the bottom, as fraction of expected height
OCC_NEIGHBOR_DY = 0.30      # a neighbor reaching this*ref_height lower => feet hidden
OCC_DIRECT_COVER_FRAC = 0.90   # this mask's OWN bottom-edge pixels overlapped directly
                                # by another mask's pixels (as opposed to _occluded_below's
                                # band strictly beneath the mask) -- catches a teammate's
                                # mask drawn right over this player's lower body/feet in a
                                # tight huddle scrum, where there's no clean gap below to
                                # measure "below" coverage against

# ── FRAGMENT FILTER ──
# Short slivers are excluded from the height-model fit, but KEPT for position
# (flagged, never silently dropped) -- matches the final single-cell behavior.
FRAGMENT_MIN_FRAC = 0.55

# ── PER-PLAYER SMOOTHING (OFF by final decision -- smooth later on the CSV) ──
SMOOTH_POS       = False
SMOOTH_POS_WIN   = 5
ID_STABILITY_MIN = 8

# ── DEPTH-AWARE HEIGHT MODEL ──
# Fit per-huddle (two-pass, fragment-excluded). FIXED_* lets you paste coefficients
# from a rally clip instead; leave None to fit per-huddle.
FIXED_HSLOPE     = None
FIXED_HINTERCEPT = None
HEIGHT_FIT_BINS      = 12
HEIGHT_FIT_PCT       = 95
HEIGHT_FIT_MIN_N     = 3
HEIGHT_FIT_MIN_MASKS = 20
HEIGHT_MIN_PX        = 20.0
HEIGHT_SPAN_WARN     = 120

# ─────────────────────────────────────────────────────────────────────────────
def flag_duplicates(np_dict, min_sep=DUP_PX):
    names = [k for k,v in np_dict.items() if v is not None]; bad=set()
    for a,b in itertools.combinations(names,2):
        pa=np.asarray(np_dict[a],float).flatten()[:2]; pb=np.asarray(np_dict[b],float).flatten()[:2]
        if np.linalg.norm(pa-pb)<min_sep: bad.add(a); bad.add(b)
    return bad

def run_pipeline(frame_bgr):
    from sam3.model.sam3_image_processor import Sam3Processor
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_np  = np.array(img_pil); H_img, W_img = img_np.shape[:2]
    processor = Sam3Processor(model, confidence_threshold=CONF)
    state = processor.set_image(img_pil); res = processor.set_text_prompt("white court lines", state)
    scores = [s.item() for s in res["scores"]]
    drop=set()
    for i in range(len(res["masks"])):
        for j in range(i+1,len(res["masks"])):
            if i in drop or j in drop: continue
            m1=res["masks"][i].squeeze(0).cpu().numpy().astype(bool); m2=res["masks"][j].squeeze(0).cpu().numpy().astype(bool)
            inter=(m1&m2).sum(); union=(m1|m2).sum()
            if union>0 and inter/union>IOU_THRESH:
                loser=(i if m1.sum()<m2.sum() else j) if scores[i]==scores[j] else (i if scores[i]<scores[j] else j)
                drop.add(loser)
    keep=[i for i in range(len(res["masks"])) if i not in drop]
    res["scores"]=[res["scores"][i] for i in keep]; res["masks"]=[res["masks"][i] for i in keep]; scores=[scores[i] for i in keep]
    keep=[i for i in range(len(res["masks"])) if res["masks"][i].squeeze(0).cpu().numpy().sum()>=MIN_MASK_PX]
    res["scores"]=[res["scores"][i] for i in keep]; res["masks"]=[res["masks"][i] for i in keep]; scores=[scores[i] for i in keep]
    while len(scores)>MAX_LINES:
        w=int(np.argmin(scores)); res["scores"].pop(w); res["masks"].pop(w); scores.pop(w)
    def clip_curve(c):
        if len(c)==0: return c
        return c[(c[:,0]>=0)&(c[:,0]<=W_img)&(c[:,1]>=0)&(c[:,1]<=H_img)]
    def make_eval(spline,u_min,u_max,slope,horizontal):
        def eval_curve(t_vals):
            out=[]
            for t in t_vals:
                vi=float(spline(np.clip(t,u_min,u_max)))+slope*(t-np.clip(t,u_min,u_max))
                out.append([t,vi] if horizontal else [vi,t])
            return np.array(out)
        return eval_curve
    fitted_lines=[]
    for i,mask in enumerate(res["masks"]):
        m=mask.squeeze(0).cpu().numpy().astype(np.uint8); skel=skeletonize(m).astype(np.uint8)
        ys,xs=np.where(skel>0)
        if len(xs)<10: ys,xs=np.where(m>0)
        if len(xs)<10: continue
        order=np.argsort(xs); xs,ys=xs[order],ys[order]
        if len(xs)>20:
            angles=np.arctan2(np.diff(ys.astype(float)),np.diff(xs.astype(float))); med=np.median(angles)
            good=np.abs(angles-med)<np.radians(ANGLE_TOL_DEG); good=np.append(good,True); xs,ys=xs[good],ys[good]
            if len(xs)<10: continue
        x_range=xs.max()-xs.min(); y_range=ys.max()-ys.min(); horizontal=x_range>=y_range
        u,v=(xs,ys) if horizontal else (ys,xs); order=np.argsort(u); u,v=u[order].astype(float),v[order].astype(float)
        _,idx=np.unique(u,return_index=True); u,v=u[idx],v[idx]
        spline=UnivariateSpline(u,v,k=1,s=len(u)*8); u_min,u_max=float(u.min()),float(u.max())
        slope=spline.derivatives(u_min+1)[1]; eval_curve=make_eval(spline,u_min,u_max,slope,horizontal)
        ext_span=u_max-u_min
        detected=clip_curve(eval_curve(np.linspace(u_min,u_max,300)))
        ext_lo=clip_curve(eval_curve(np.linspace(u_min-ext_span,u_min,200)))
        ext_hi=clip_curve(eval_curve(np.linspace(u_max,u_max+ext_span,200)))
        fitted_lines.append(dict(id=i,horizontal=horizontal,u_min=u_min,u_max=u_max,eval=eval_curve,
            detected=detected,ext_lo=ext_lo,ext_hi=ext_hi,score=scores[i],x_range=x_range,y_range=y_range))
    def _ang(ln): p0,p1=ln["detected"][0],ln["detected"][-1]; return np.arctan2(p1[1]-p0[1],p1[0]-p0[0])
    def _perp(a,b):
        p0,p1=b["detected"][0],b["detected"][-1]; d=p1-p0; L=np.linalg.norm(d)+1e-9; m=a["detected"][len(a["detected"])//2]
        return abs((d[0]*(p0[1]-m[1])-(p0[0]-m[0])*d[1])/L)
    def _len(ln): return np.linalg.norm(ln["detected"][-1]-ln["detected"][0])
    dupdrop=set()
    for a in range(len(fitted_lines)):
        for b in range(a+1,len(fitted_lines)):
            if a in dupdrop or b in dupdrop: continue
            da=abs(_ang(fitted_lines[a])-_ang(fitted_lines[b])); da=min(da,np.pi-da)
            close=min(_perp(fitted_lines[a],fitted_lines[b]),_perp(fitted_lines[b],fitted_lines[a]))<SAME_LINE_PX
            if da<np.radians(SAME_ANGLE_DEG) and close:
                dupdrop.add(a if _len(fitted_lines[a])<_len(fitted_lines[b]) else b)
    if dupdrop: fitted_lines=[ln for k,ln in enumerate(fitted_lines) if k not in dupdrop]
    def line_line_intersect(la,lb):
        p1,p2=la["detected"][0].astype(float),la["detected"][-1].astype(float)
        p3,p4=lb["detected"][0].astype(float),lb["detected"][-1].astype(float)
        d1,d2=p2-p1,p4-p3; cross=d1[0]*d2[1]-d1[1]*d2[0]
        if abs(cross)<1e-9: return None
        t=((p3[0]-p1[0])*d2[1]-(p3[1]-p1[1])*d2[0])/cross; u=((p3[0]-p1[0])*d1[1]-(p3[1]-p1[1])*d1[0])/cross
        pt=p1+t*d1
        if not (0<=pt[0]<=W_img and 0<=pt[1]<=H_img): return None
        L1,L2=np.linalg.norm(d1)+1e-9,np.linalg.norm(d2)+1e-9
        if max(0.0,-t,t-1)*L1>MAX_EXTRAP_PX or max(0.0,-u,u-1)*L2>MAX_EXTRAP_PX: return None
        return pt
    pairwise_pts={}
    for i in range(len(fitted_lines)):
        for j in range(i+1,len(fitted_lines)):
            pt=line_line_intersect(fitted_lines[i],fitted_lines[j])
            if pt is not None: pairwise_pts[(i,j)]=pt
    for i,ln in enumerate(fitted_lines):
        relevant=[pt for (a,b),pt in pairwise_pts.items() if a==i or b==i]
        if len(ln["ext_hi"])>1 and relevant:
            bc=len(ln["ext_hi"])-1
            for pt in relevant:
                dists=np.linalg.norm(ln["ext_hi"]-pt,axis=1); cut=int(np.argmin(dists))
                if dists[cut]<30 and cut<bc: bc=cut
            ln["ext_hi"]=ln["ext_hi"][:bc+1]
        if len(ln["ext_lo"])>1 and relevant:
            bc=0
            for pt in relevant:
                dists=np.linalg.norm(ln["ext_lo"]-pt,axis=1); cut=int(np.argmin(dists))
                if dists[cut]<30 and cut>bc: bc=cut
            ln["ext_lo"]=ln["ext_lo"][bc:]
    for ln in fitted_lines:
        for key in ["ext_hi","ext_lo"]:
            if len(ln[key])>1 and np.linalg.norm(ln[key][-1]-ln[key][0])<MIN_EXT_PX: ln[key]=np.empty((0,2))
    for i,ln in enumerate(fitted_lines):
        relevant=[pt for (a,b),pt in pairwise_pts.items() if a==i or b==i]
        for pt in relevant:
            det=ln["detected"]; t=np.dot(pt-det[0],det[-1]-det[0])/(np.linalg.norm(det[-1]-det[0])**2+1e-9)
            if len(ln["ext_hi"])>1 and t<=1.0: ln["ext_hi"]=np.empty((0,2))
            if len(ln["ext_lo"])>1 and t>=0.0: ln["ext_lo"]=np.empty((0,2))
    def line_angle_off_horiz(ln):
        pts=ln["detected"]; n=len(pts); p1,p2=pts[n//4],pts[3*n//4]
        ang=np.degrees(np.arctan2(p2[1]-p1[1],p2[0]-p1[0]))
        return abs(((ang+90)%180)-90)
    def cx(ln): return ln["detected"][:,0].mean()
    def cy(ln): return ln["detected"][:,1].mean()
    for ln in fitted_lines:
        a=line_angle_off_horiz(ln); ln["angle"]=a
        ln["type"]="sideline" if a<=SIDELINE_MAX_DEG else ("endline" if a<=ENDLINE_MAX_DEG else "attack")
        ln["label"]="unknown"
    attacks=[ln for ln in fitted_lines if ln["type"]=="attack"]; sidelines=[ln for ln in fitted_lines if ln["type"]=="sideline"]; endlines=[ln for ln in fitted_lines if ln["type"]=="endline"]
    if len(attacks)>=2: attacks.sort(key=cx); attacks[0]["label"]="attack_left"; attacks[-1]["label"]="attack_right"
    elif len(attacks)==1: attacks[0]["label"]="attack_left" if cx(attacks[0])<W_img/2 else "attack_right"
    if len(sidelines)>=2: sidelines.sort(key=cy); sidelines[0]["label"]="sideline_far"; sidelines[-1]["label"]="sideline_close"
    elif len(sidelines)==1:
        others=[ln for ln in fitted_lines if ln["type"] in ("endline","attack")]
        ref=float(np.mean([cy(o) for o in others])) if others else H_img/2
        sidelines[0]["label"]="sideline_far" if cy(sidelines[0])<ref else "sideline_close"
    if len(endlines)>=2: endlines.sort(key=cx); endlines[0]["label"]="endline_left"; endlines[-1]["label"]="endline_right"
    elif len(endlines)==1: endlines[0]["label"]="endline_left" if cx(endlines[0])<W_img/2 else "endline_right"
    inter_by_label={}
    for (i,j),pt in pairwise_pts.items():
        la,lb=fitted_lines[i]["label"],fitted_lines[j]["label"]
        if la and lb and la!="unknown" and lb!="unknown": inter_by_label[(la,lb)]=pt; inter_by_label[(lb,la)]=pt
    gi=lambda a,b: inter_by_label.get((a,b))
    named={"corner_right_far":gi("endline_right","sideline_far"),"corner_right_close":gi("endline_right","sideline_close"),
           "corner_left_far":gi("endline_left","sideline_far"),"corner_left_close":gi("endline_left","sideline_close"),
           "attack_left_far":gi("attack_left","sideline_far"),"attack_left_close":gi("attack_left","sideline_close"),
           "attack_right_far":gi("attack_right","sideline_far"),"attack_right_close":gi("attack_right","sideline_close")}
    alf,arf=named.get("attack_left_far"),named.get("attack_right_far"); alc,arc=named.get("attack_left_close"),named.get("attack_right_close")
    named["net_far"]=(alf+arf)/2 if (alf is not None and arf is not None) else None
    named["net_close"]=(alc+arc)/2 if (alc is not None and arc is not None) else None
    return fitted_lines, named

# ── geometry helpers ──
def build_homography(named):
    ip,cp=[],[]
    for nm,xy in COURT_COORDS_MAP.items():
        v=named.get(nm)
        if v is None: continue
        p=np.asarray(v,float).flatten()[:2]
        if not np.isfinite(p).all(): continue
        ip.append([p[0],p[1]]); cp.append([float(xy[0]),float(xy[1])])
    if len(ip)<4: return None
    H,_=cv2.findHomography(np.array(ip,np.float32),np.array(cp,np.float32),cv2.RANSAC,5.0); return H

def n_named(named):
    return sum(1 for nm in COURT_COORDS_MAP
               if named.get(nm) is not None
               and np.isfinite(np.asarray(named[nm],float).flatten()[:2]).all())

def n_named_base(named):
    """Count of the 8 independently detected base points (excludes the two derived net points)."""
    if named is None: return 0
    return sum(1 for nm in BASE_COURT_KEYS
               if named.get(nm) is not None
               and np.isfinite(np.asarray(named[nm],float).flatten()[:2]).all())

def project_point(px,py,H):
    o=cv2.perspectiveTransform(np.array([[[px,py]]],np.float32),H); return float(o[0,0,0]),float(o[0,0,1])

def _court_bounds_ok(mx,my,named,margin):
    # Shared by is_on_court (projects raw pixel coords first) and the
    # ever-on-court check in _featurize_tracked, which works off `tracks`
    # -- already-projected court-meters positions -- so it doesn't need to
    # re-derive this boundary logic separately (and risk it drifting out of
    # sync with this).
    la=named.get("corner_left_far") is not None or named.get("corner_left_close") is not None
    ra=named.get("corner_right_far") is not None or named.get("corner_right_close") is not None
    return ((-margin if la else -999)<=mx<=(COURT_W+margin if ra else 999)) and (-margin<=my<=COURT_H+margin)

def is_on_court(fx,fy,H,named,margin=COURT_MARGIN):
    if H is None or fx is None: return True
    mx,my=project_point(fx,fy,H)
    return _court_bounds_ok(mx,my,named,margin)

def _iou(a,b):
    a=a.astype(bool); b=b.astype(bool); u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum())/float(u) if u else 0.0

def detect_masks_at(frame_dir, frame_index, prompt, min_px=NEW_MIN_PX):
    # Fresh throwaway session so detection never touches the tracking session.
    sid=video_predictor.handle_request(dict(type="start_session",resource_path=frame_dir))["session_id"]
    resp=video_predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=frame_index,text=prompt))
    masks=resp["outputs"]["out_binary_masks"]
    video_predictor.handle_request(dict(type="close_session",session_id=sid))
    return [m for m in masks if np.count_nonzero(m)>=min_px]

# ── mask stats: np.where once per mask, cached; read by the height fit + fragment
#    filter + the wide-gate inputs (uses 95th-pct y_max like the single cell).
#    y_max_true (the actual bottom row, not the 95th-pct) is cached alongside so
#    mask_foot_final/_embedded_shorter can reuse this same np.where instead of
#    each independently re-scanning the same mask array. ──
def mask_stats(mask):
    ys,xs=np.where(mask)
    if len(xs)==0: return None
    y_max=float(np.percentile(ys,95)); y_min=float(ys.min())
    return dict(y_min=y_min, y_max=y_max, y_max_true=float(ys.max()), h=y_max-y_min,
                x_min=float(xs.min()), x_max=float(xs.max()),
                foot_x=float((xs.min()+xs.max())/2))

def fit_height_model_cached(cache, exclude=None):
    pts=[(st['y_min'], st['h']) for key,st in cache.items()
         if st is not None and (not exclude or key not in exclude)]
    if len(pts) < HEIGHT_FIT_MIN_MASKS:
        print("    height model: too few masks -> per-frame ref fallback"); return None, None
    P=np.array(pts,dtype=float)
    bins=np.linspace(P[:,0].min(), P[:,0].max(), HEIGHT_FIT_BINS+1)
    env=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        sel=P[(P[:,0]>=lo)&(P[:,0]<hi)]
        if len(sel) >= HEIGHT_FIT_MIN_N:
            env.append([sel[:,0].mean(), float(np.percentile(sel[:,1], HEIGHT_FIT_PCT))])
    if len(env) < 3:
        print("    height model: too few depth bins -> per-frame ref fallback"); return None, None
    env=np.array(env,dtype=float)
    slope,intercept=np.polyfit(env[:,0],env[:,1],1)
    span=P[:,0].max()-P[:,0].min()
    print(f"    height model: h ≈ {slope:.3f}*y_head + {intercept:.1f}  "
          f"({len(env)} bins, {len(P)} masks, y_head span {span:.0f}px)")
    if span < HEIGHT_SPAN_WARN:
        print("    WARNING: little depth spread -> height model weakly identified")
    return float(slope), float(intercept)

def expected_height(y_head, slope, intercept, img_h):
    if slope is None: return None
    return float(np.clip(slope*y_head + intercept, HEIGHT_MIN_PX, float(img_h)))

def is_fragment(st, img_h, hslope, hintercept):
    if st is None: return True
    eh = expected_height(st['y_min'], hslope, hintercept, img_h)
    if eh is None: return False
    return st['h'] < FRAGMENT_MIN_FRAC * eh

# ── occlusion guard + foot (identical logic to the final single-cell) ──
def _occluded_below(mask, y_min, y_max, other_masks, img_w, img_h, hslope, hintercept):
    y_bot=int(y_max)
    cols_here=np.where(mask[max(0,y_bot-3):y_bot+1,:].any(axis=0))[0]
    if len(cols_here)==0 or not other_masks: return False, 0.0
    eh=expected_height(float(y_min), hslope, hintercept, img_h)
    band=int(OCC_BAND_FRAC*eh) if eh else int(0.5*(y_max-y_min)+15)
    y0,y1=max(0,y_bot-2), min(y_bot+band, img_h)
    if y1<=y0: return False, 0.0
    union=np.zeros(img_w, bool)
    for m in other_masks: union |= m[y0:y1,:].any(axis=0)
    cover=float(union[cols_here].mean())
    return cover>=OCC_COVER_FRAC, cover

def _directly_covered_bottom(mask, y_max, other_masks, img_w, frac_thresh=OCC_DIRECT_COVER_FRAC, band_px=4):
    """True if another mask covers the row just below this mask's bottom edge."""
    y_bot=int(y_max)
    cols_here=np.where(mask[max(0,y_bot-3):y_bot+1,:].any(axis=0))[0]
    if len(cols_here)==0 or not other_masks: return False
    y0,y1=max(0,y_bot-1), min(y_bot+band_px, mask.shape[0])
    if y1<=y0: return False
    union=np.zeros(img_w, bool)
    for om in other_masks: union |= om[y0:y1,:].any(axis=0)
    return float(union[cols_here].mean()) >= frac_thresh

def _embedded_shorter(x0, x1, y_max, other_stats, ref_height):
    # other_stats: cached mask_stats() dicts for the other masks (x_min/x_max/
    # y_max already computed there -- avoids a fresh np.where per other mask).
    if not other_stats or ref_height is None: return False
    w=max(1,x1-x0)
    for ost in other_stats:
        if ost is None: continue
        ox0,ox1=ost['x_min'],ost['x_max']
        if min(x1,ox1)-max(x0,ox0) <= 0.2*w: continue
        if ost['y_max']-y_max > OCC_NEIGHBOR_DY*ref_height: return True
    return False

def mask_foot_final(mask, img_h, img_w, other_masks, hslope, hintercept, wide=None, st=None, other_stats=None):
    # returns (foot_x, foot_y, extrapolated_bool)
    # st: this mask's cached mask_stats() dict, if already computed -- skips
    # this function's own np.where(mask) scan. other_stats: cached mask_stats()
    # dicts for other_masks, in the same order -- used by _embedded_shorter
    # instead of it re-scanning every other mask from scratch.
    if st is not None:
        y_min=st['y_min']; y_max=st['y_max_true']; x_min=st['x_min']; x_max=st['x_max']
    else:
        ys,xs=np.where(mask)
        if len(xs)==0: return None, None, False
        y_max=float(ys.max()); y_min=float(ys.min())
        x_min,x_max=float(xs.min()),float(xs.max())
    mask_h=y_max-y_min; mask_w=x_max-x_min
    foot_x=float((x_min+x_max)/2)
    if wide is None: wide = mask_w > WIDE_ENTER_ASPECT*mask_h
    if wide:
        # Normally a wide mask = a genuine dive/sprawl -> trust the raw bottom, no extrapolation.
        mask_area=float(np.count_nonzero(mask))
        directly_covered = mask_area>0 and any(
            np.logical_and(mask, om).sum() > WIDE_DIRECT_OVERLAP_FRAC*mask_area
            for om in (other_masks or [])
        )
        if not directly_covered:
            return foot_x, y_max, False
    ref_height=expected_height(y_min, hslope, hintercept, img_h)
    if ref_height is None or mask_h > HEIGHT_SHORT_FRAC*ref_height:
        return foot_x, y_max, False
    occ,_=_occluded_below(mask, y_min, y_max, other_masks or [], img_w, img_h, hslope, hintercept)
    emb=_embedded_shorter(x_min, x_max, y_max, other_stats or [], ref_height)
    direct=_directly_covered_bottom(mask, y_max, other_masks or [], img_w)
    if occ or emb or direct:
        return foot_x, float(min(y_min+ref_height, img_h-1)), True
    return foot_x, y_max, False

def wide_gate(oid, mask_w, mask_h, state):
    ratio=mask_w/max(mask_h,1.0)
    s=state.get(oid, {'mode':False,'pend':0})
    if s['mode']:
        s['pend']=s['pend']+1 if ratio<WIDE_EXIT_ASPECT else 0
        if s['pend']>=WIDE_HYST_FRAMES: s['mode'],s['pend']=False,0
    else:
        s['pend']=s['pend']+1 if ratio>WIDE_ENTER_ASPECT else 0
        if s['pend']>=WIDE_HYST_FRAMES: s['mode'],s['pend']=True,0
    state[oid]=s
    return s['mode']

def dedup_points(pts,min_sep=DEDUP_M):
    out=[]
    for p in pts:
        if all(np.hypot(p[0]-q[0],p[1]-q[1])>=min_sep for q in out): out.append(p)
    return out

def robust_hull_area(pos,k=2.5):
    p=np.asarray(pos,float)
    if len(p)<3: return np.nan
    c=np.median(p,0); d=np.linalg.norm(p-c,axis=1); med=np.median(d); mad=np.median(np.abs(d-med))+1e-6
    q=p[d<med+k*1.4826*mad]; q=q if len(q)>=3 else p
    try: return float(ConvexHull(q).volume)
    except: return np.nan

def feats(pos):
    d={}; n=len(pos); p=np.array(pos,float) if n else np.empty((0,2)); d['n']=n
    d['cx']=float(p[:,0].mean()) if n else np.nan; d['cy']=float(p[:,1].mean()) if n else np.nan
    if n>=2:
        dm=cdist(p,p); pw=dm[np.triu_indices(n,1)]
        d['pair_mean']=float(pw.mean()); d['pair_max']=float(pw.max()); d['pair_min']=float(pw.min()); d['pair_std']=float(pw.std())
        dm2=dm.copy(); np.fill_diagonal(dm2,np.inf); d['nn_mean']=float(dm2.min(axis=1).mean())
        d['width']=float(p[:,0].max()-p[:,0].min()); d['depth']=float(p[:,1].max()-p[:,1].min()); d['bbox_area']=d['width']*d['depth']
        d['width_std']=float(p[:,0].std()); d['depth_std']=float(p[:,1].std())
        c=p.mean(0); d['r_gyration']=float(np.sqrt(((p-c)**2).sum(1).mean())); d['net_dist']=float(np.abs(p[:,0]-NET_X).mean())
    else:
        for k in ['pair_mean','pair_max','pair_min','pair_std','nn_mean','width','depth','bbox_area','width_std','depth_std','r_gyration','net_dist']: d[k]=np.nan
    if n>=3:
        try: h=ConvexHull(p); d['hull_area']=float(h.volume); d['hull_perim']=float(h.area)
        except: d['hull_area']=np.nan; d['hull_perim']=np.nan
        d['hull_area_robust']=robust_hull_area(pos)
    else: d['hull_area']=d['hull_perim']=d['hull_area_robust']=np.nan
    return d

def smooth_tracks(track_dict, win):
    sm=defaultdict(dict)
    for oid,fr in track_dict.items():
        fis=sorted(fr)
        sx=pd.Series([fr[f][0] for f in fis], index=fis).rolling(win, center=True, min_periods=1).median()
        sy=pd.Series([fr[f][1] for f in fis], index=fis).rolling(win, center=True, min_periods=1).median()
        for f in fis: sm[oid][f]=(float(sx[f]), float(sy[f]))
    return sm

def compute_static_homography(video_path):
    """Detect court lines and fit the homography for a static-camera clip (no tracking)."""
    cap=cv2.VideoCapture(video_path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices=[f for f in STATIC_PROBE_FRAMES if f < total] or [0]
    all_named={}
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx); ret,fb=cap.read()
        if not ret: break
        try:
            _,named=run_pipeline(fb)
            all_named[frame_idx]=named
        except Exception as e:
            print(f"    line {frame_idx}: {e}")
    cap.release()
    Hpf={k:build_homography(n) for k,n in all_named.items()}; Hpf={k:H for k,H in Hpf.items() if H is not None}
    if not Hpf: return None, None
    f0=max(Hpf, key=lambda k: n_named(all_named[k]))
    return Hpf[f0], all_named[f0]


# ── one huddle: calibration + frame extraction (no SAM3 tracking session yet) ──
def prepare_huddle_frames(video_path, huddle_number):
    """Homography detection, frame extraction and late-entering-player scan for one clip."""
    cap=cv2.VideoCapture(video_path); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    global LAST_STATIC_H, LAST_STATIC_NAMED

    if FORCE_MANUAL_HOMOGRAPHY or FORCE_LOCKED_HOMOGRAPHY:
        # A manually-labeled (MANUAL_HOMOGRAPHY_MATCHUPS) or locked-from-huddle
        # (LOCKED_HOMOGRAPHY_SOURCE_HUDDLE) matchup -- skip line-detection
        # entirely rather than let it compete with the seeded calibration. The
        # normal STATIC_CAMERA comparison below only falls back to
        # LAST_STATIC_NAMED when a huddle's own detection finds FEWER base
        # points, so a confidently WRONG detection that happens to find as
        # many (or more) points than the seeded calibration would otherwise win
        # and silently replace it -- exactly the failure mode both mechanisms
        # need avoided.
        cap.release()
        H_static, named_static = LAST_STATIC_H, LAST_STATIC_NAMED
        reason = "manually-labeled" if FORCE_MANUAL_HOMOGRAPHY else "locked-from-huddle"
        print(f"    using {reason} homography (auto-detection skipped for this matchup)")
    else:
        if STATIC_CAMERA:
            frame_indices=[f for f in STATIC_PROBE_FRAMES if f < total] or [0]
        else:
            frame_indices=list(range(0,min(total,MAX_FRAMES*FRAME_STEP),FRAME_STEP))
        all_named={}
        for fi,frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx); ret,fb=cap.read()
            if not ret: break
            try:
                _,named=run_pipeline(fb); bad=flag_duplicates(named)
                if bad:
                    prev=all_named.get(frame_indices[fi-1],{}) if fi>0 else {}
                    for nm in bad: named[nm]=prev.get(nm)
                all_named[frame_idx]=named
            except Exception as e: print(f"    line {frame_idx}: {e}")
        cap.release()

        Hpf={k:build_homography(n) for k,n in all_named.items()}; Hpf={k:H for k,H in Hpf.items() if H is not None}

        if not Hpf:
            if STATIC_CAMERA and LAST_STATIC_H is not None:
                print("    no homography detected -> reusing last static-camera homography")
                f0=None; H_static, named_static = LAST_STATIC_H, LAST_STATIC_NAMED
            else:
                print("    no homography -> skip"); return None
        elif STATIC_CAMERA:
            f0=max(Hpf, key=lambda k: n_named(all_named[k]))
            H_static=Hpf[f0]; named_static=all_named[f0]
            n_base=n_named_base(named_static)
            print(f"    STATIC_CAMERA: using frame {f0} homography ({n_named(named_static)} pts, "
                  f"{n_base}/{FULL_REG_POINTS} base)")
            # Camera doesn't move on a STATIC_CAMERA game, so a worse-than-usual
            # detection on THIS huddle's own probe frames is noise, not signal --
            # prefer the last homography we trusted more (more base points) over
            # silently tracking through a degraded one. LAST_STATIC_NAMED is only
            # ever set below to whatever we end up using here, so this can only
            # ever hold steady or improve across huddles, never ratchet down.
            if LAST_STATIC_NAMED is not None and n_named_base(LAST_STATIC_NAMED) > n_base:
                print(f"    only {n_base}/{FULL_REG_POINTS} base points on this huddle's own probe "
                      f"frames -> reusing last static-camera homography instead "
                      f"({n_named_base(LAST_STATIC_NAMED)}/{FULL_REG_POINTS} base)")
                H_static, named_static = LAST_STATIC_H, LAST_STATIC_NAMED
        else:
            f0=min(all_named); H_static=None; named_static=None

    if STATIC_CAMERA:
        LAST_STATIC_H, LAST_STATIC_NAMED = H_static, named_static

    def getH(fi):
        if STATIC_CAMERA: return H_static
        if fi in Hpf: return Hpf[fi]
        return Hpf[min(Hpf, key=lambda k: abs(k-fi))]
    def getNamed(fi):
        if STATIC_CAMERA: return named_static
        return all_named.get(fi, all_named[f0])

    FRAME_DIR=tempfile.mkdtemp(prefix=f"vb_h{huddle_number}_"); cap=cv2.VideoCapture(video_path); nf=0
    while True:
        ret,fb=cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(FRAME_DIR,f"{nf:05d}.jpg"),fb); nf+=1
    cap.release(); ff=sorted(os.listdir(FRAME_DIR))

    # ── SCAN for late-entering players (throwaway sessions; only one session live) ──
    scan_cands={}
    for F in SCAN_FRAMES:
        F=min(F, len(ff)-1)
        scan_cands[F]=detect_masks_at(FRAME_DIR, F, SAM3_PROMPT)
        print(f"    scan frame {F}: {len(scan_cands[F])} candidates")

    return ff, FRAME_DIR, getH, getNamed, fps, scan_cands


# ── one huddle ──
def process_one_huddle(video_path, huddle_number):
    prep = prepare_huddle_frames(video_path, huddle_number)
    if prep is None:
        return None
    ff, FRAME_DIR, getH, getNamed, fps, scan_cands = prep

    # ── main tracking session: detect frame 0, reinforce top-6/side, propagate ──
    sid=video_predictor.handle_request(dict(type="start_session",resource_path=FRAME_DIR))["session_id"]
    try:
        return _process_one_huddle_session(sid, FRAME_DIR, ff, getH, getNamed, huddle_number, fps, scan_cands)
    finally:
        video_predictor.handle_request(dict(type="close_session",session_id=sid))
        shutil.rmtree(FRAME_DIR,ignore_errors=True); torch.cuda.empty_cache(); gc.collect()


def _process_one_huddle_session(sid, FRAME_DIR, ff, getH, getNamed, huddle_number, fps, scan_cands):
    tracked = _track_huddle_session(sid, FRAME_DIR, ff, getH, getNamed, huddle_number, fps, scan_cands)
    return _featurize_tracked(tracked, ff, getH, getNamed, huddle_number, fps)


# ── phase 1: SAM3 detect/propagate/inject + stats + height model. Independent
#    of which homography will eventually be used to project pixel positions
#    into court meters -- getH/getNamed are only consulted here to decide
#    on-court membership for the frame-0 seed prompts and the late-entering-
#    player injection, both keyed off the SAME homography that will be used
#    to featurize. Cached so a homography-jitter check can call this ONCE and
#    reproject the same tracked masks through many candidate homographies
#    (see _featurize_tracked + run_homography_jitter_check) instead of paying
#    for SAM3 tracking again per draw. ──
def _track_huddle_session(sid, FRAME_DIR, ff, getH, getNamed, huddle_number, fps, scan_cands):
    out=video_predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=0,text=SAM3_PROMPT))["outputs"]
    oids,masks,probs=out["out_obj_ids"],out["out_binary_masks"],out["out_probs"]; IMG_H,IMG_W=masks.shape[1],masks.shape[2]

    H0=getH(0); named0=getNamed(0)
    r0=[i for i in range(len(masks)) if np.count_nonzero(masks[i])>=MIN_MASK_AREA]
    cands=[]
    for i in r0:
        others=[masks[j] for j in r0 if j!=i]
        fx,fy,_=mask_foot_final(masks[i], IMG_H, IMG_W, others, None, None)
        if fx is None or not is_on_court(fx,fy,H0,named0): continue
        mx,_=project_point(fx,fy,H0); cands.append(dict(i=i,prob=float(probs[i]),side=('left' if mx<NET_X else 'right')))
    keep=set()
    for s in ('left','right'):
        keep|={c['i'] for c in sorted([c for c in cands if c['side']==s],key=lambda c:-c['prob'])[:EXPECTED_PER_SIDE]}
    for i in keep:
        ys_m,xs_m=np.where(masks[i]); cxp,cyc=float(xs_m.mean()),float(ys_m.mean()); cyt=float(ys_m.min()+(ys_m.max()-ys_m.min())*0.1)
        video_predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=0,obj_id=int(oids[i]),
            points=torch.tensor([[cxp/IMG_W,cyc/IMG_H],[cxp/IMG_W,cyt/IMG_H]],dtype=torch.float32),
            point_labels=torch.tensor([1,1],dtype=torch.int32)))

    opf={}
    for r in video_predictor.handle_stream_request(request=dict(type="propagate_in_video",session_id=sid,output_prob_thresh=0.25)):
        opf[r["frame_index"]]=r["outputs"]

    # ── INJECT late-entering players (point prompt, fresh obj_id, re-propagate) ──
    # This is the exact original inject-into-the-same-session-then-re-propagate
    # pattern. It relies entirely on patch_sam3_propagation_classifier() (see
    # above) to fix the sam3 library bug that broke it: a session's genuinely
    # first propagation was being misclassified as an incremental "partial"
    # update, which crashes because nothing's cached yet. Once that first
    # propagation is correctly classified as "full", every frame (including
    # each late entrant's own frame) is already cached by the time we inject
    # and re-propagate, which is exactly the scenario "partial" propagation is
    # designed for -- so no rebuild/second session is needed after all.
    next_oid=int(max(int(o) for o in oids))+1
    added_positions=[]; n_new=0
    for F in sorted(scan_cands):
        if F not in opf: continue
        existing=[m for m in opf[F]['out_binary_masks'] if np.count_nonzero(m)>=NEW_MIN_PX]
        H_F=getH(F); named_F=getNamed(F)
        for cm in scan_cands[F]:
            if any(_iou(cm,em)>NEW_IOU_THRESH for em in existing): continue
            fx,fy,_=mask_foot_final(cm, IMG_H, IMG_W, existing, None, None)
            if fx is None or not is_on_court(fx,fy,H_F,named_F,margin=TRACK_MARGIN): continue
            mx,my=project_point(fx,fy,H_F)
            if any(np.hypot(mx-ax,my-ay)<NEW_DEDUP_M for ax,ay in added_positions): continue
            ys_c,xs_c=np.where(cm); ncx,ncy=float(xs_c.mean()),float(ys_c.mean()); ncy_t=float(ys_c.min()+(ys_c.max()-ys_c.min())*0.1)
            video_predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=F,obj_id=next_oid,
                points=torch.tensor([[ncx/IMG_W,ncy/IMG_H],[ncx/IMG_W,ncy_t/IMG_H]],dtype=torch.float32),
                point_labels=torch.tensor([1,1],dtype=torch.int32)))
            added_positions.append((mx,my)); n_new+=1
            print(f"    + new player obj {next_oid} @ frame {F} court=({mx:.1f},{my:.1f})")
            next_oid+=1
    if n_new:
        print(f"    re-propagating with {n_new} injected player(s)...")
        opf={}
        for r in video_predictor.handle_stream_request(request=dict(type="propagate_in_video",session_id=sid,output_prob_thresh=0.25)):
            opf[r["frame_index"]]=r["outputs"]

    # ── mask stats cache (height fit + fragment filter + wide-gate inputs) ──
    stats_cache={}
    for fi in opf:
        for mi,m in enumerate(opf[fi]['out_binary_masks']):
            if np.count_nonzero(m) >= MIN_MASK_AREA:
                stats_cache[(fi,mi)] = mask_stats(m)

    # ── height model (two-pass: fit, drop fragments, refit) ──
    if FIXED_HSLOPE is not None:
        HSLOPE, HINTERCEPT = FIXED_HSLOPE, FIXED_HINTERCEPT
        print(f"    height model: FIXED h ≈ {HSLOPE:.3f}*y_head + {HINTERCEPT:.1f}")
    else:
        HSLOPE, HINTERCEPT = fit_height_model_cached(stats_cache)
        if HSLOPE is not None:
            frags={key for key,st in stats_cache.items() if is_fragment(st, IMG_H, HSLOPE, HINTERCEPT)}
            if frags:
                print(f"    {len(frags)} fragment masks -> refitting without them")
                s2,i2 = fit_height_model_cached(stats_cache, exclude=frags)
                if s2 is not None: HSLOPE, HINTERCEPT = s2, i2

    return dict(opf=opf, stats_cache=stats_cache, HSLOPE=HSLOPE, HINTERCEPT=HINTERCEPT,
                IMG_H=IMG_H, IMG_W=IMG_W, n_new=n_new)


# ── phase 2: project the tracked masks through a given homography (getH/
#    getNamed) into court-meter positions and compute per-frame features.
#    Pure numpy -- no SAM3/GPU calls -- so it's cheap to call many times with
#    different homographies over the SAME `tracked` bundle from
#    _track_huddle_session (that's the whole point of the jitter check).
#    return_kept=True additionally returns {frame_index: {oid: (mx,my), ...}}
#    -- the exact set of object ids PASS A below decided actually counts
#    (i.e. made it into left_points/right_points), each with its raw
#    (pre-dedup, pre-CSV-smoothing) court-meter position, as opposed to
#    every mask SAM3 ever tracked. sam3_video_maker.py uses this to only
#    draw boxes for players who weren't filtered out (off-court, etc.), and
#    to run its own display-only smoothing across frames by oid, without
#    duplicating this filtering/position logic itself. Default False keeps
#    every existing caller's return value (just the DataFrame) unchanged.
#    dedup=False skips the dedup_points() merge-nearby-points step (PASS B)
#    entirely, so left_points/right_points reflect every distinct tracked
#    oid with no post-hoc merging. Default True preserves every existing
#    caller's exact prior behavior; sam3_video_maker.py is the only caller
#    that passes dedup=False, to visually sanity-check how often/how much
#    that step actually changes things before touching the default here. ──
def _featurize_tracked(tracked, ff, getH, getNamed, huddle_number, fps, return_kept=False, dedup=True):
    opf, stats_cache = tracked['opf'], tracked['stats_cache']
    HSLOPE, HINTERCEPT = tracked['HSLOPE'], tracked['HINTERCEPT']
    IMG_H, IMG_W, n_new = tracked['IMG_H'], tracked['IMG_W'], tracked['n_new']

    # ── id stability (smoothing off by default) ──
    idc=defaultdict(int)
    for fi in opf:
        for oid in opf[fi]['out_obj_ids']: idc[int(oid)]+=1
    n_stable=sum(1 for n in idc.values() if n >= 0.9*len(opf))
    do_smooth = SMOOTH_POS and n_stable >= ID_STABILITY_MIN
    if SMOOTH_POS and not do_smooth:
        print(f"    id churn ({n_stable} stable of {len(idc)}) -> skipping per-track smoothing")

    # ── PASS A: positions (single estimator) + per-side body-blob silhouette ──
    per_frame={}; tracks=defaultdict(dict); blob_by_frame={}
    wide_state={}
    n_frag_total=0; n_extrap_total=0

    for fi,fname in enumerate(ff):
        H_fi=getH(fi); named_fi=getNamed(fi); per_frame[fi]=[]
        blob_by_frame[fi]={'left':dict(area=0,hull=np.nan,w=0,h=0),
                           'right':dict(area=0,hull=np.nan,w=0,h=0)}
        if fi not in opf: continue
        masks_fi=opf[fi]['out_binary_masks']; oids_fi=opf[fi]['out_obj_ids']
        kept=[mi for mi in range(len(masks_fi)) if stats_cache.get((fi,mi)) is not None]
        if not kept: continue
        raw_masks=[masks_fi[mi] for mi in kept]
        side_masks={'left':[], 'right':[]}

        for pos_idx, mi in enumerate(kept):
            st=stats_cache[(fi,mi)]; oid=int(oids_fi[mi])
            frag=is_fragment(st, IMG_H, HSLOPE, HINTERCEPT)   # flag, but KEEP for position
            if frag: n_frag_total+=1
            mw=st['x_max']-st['x_min']; mh=st['h']
            wide=wide_gate(oid, mw, mh, wide_state)
            other=[raw_masks[j] for j in range(len(kept)) if j!=pos_idx]
            other_stats=[stats_cache[(fi,kept[j])] for j in range(len(kept)) if j!=pos_idx]
            fx,fy,extrap=mask_foot_final(masks_fi[mi], IMG_H, IMG_W, other, HSLOPE, HINTERCEPT,
                                         wide=wide, st=st, other_stats=other_stats)
            if fx is None: continue
            if extrap: n_extrap_total+=1
            # No per-frame on-court filter here: once an oid is an
            # established tracked player (via the frame-0 seed or the
            # 30/45 scan injection -- the only ways in), she stays counted
            # for every frame she has a valid mask, even one where she's
            # momentarily off-court, rather than silently dropping out of
            # left_points/right_points for that frame. is_on_court is still
            # what gates WHO becomes a tracked player in the first place
            # (see _track_huddle_session); this is only about whether an
            # already-established player keeps being followed afterward.
            mx,my=project_point(fx,fy,H_fi)
            side='left' if mx<NET_X else 'right'
            per_frame[fi].append(dict(oid=oid, pos=(mx,my)))
            tracks[oid][fi]=(mx,my)
            side_masks[side].append((masks_fi[mi],st))       # keep mask+bbox for its side's blob

        for side, ms in side_masks.items():
            if not ms: continue
            # Crop to the union of the masks' cached bounding boxes before the
            # full-resolution OR-reduce/np.where -- same result (area/hull/w/h
            # are all translation-invariant), far fewer pixels to touch since
            # most of a full frame is empty background.
            y0=max(0, int(min(s['y_min'] for _,s in ms)))
            y1=min(IMG_H, int(max(s['y_max_true'] for _,s in ms))+1)
            x0=max(0, int(min(s['x_min'] for _,s in ms)))
            x1=min(IMG_W, int(max(s['x_max'] for _,s in ms))+1)
            blob_crop=np.logical_or.reduce([m[y0:y1,x0:x1].astype(bool) for m,_ in ms])
            ys2,xs2=np.where(blob_crop)
            hull=np.nan
            if len(xs2)>=3:
                try: hull=float(ConvexHull(np.column_stack([xs2,ys2])).volume)
                except: hull=np.nan
            blob_by_frame[fi][side]=dict(area=int(blob_crop.sum()), hull=hull,
                                         w=int(xs2.max()-xs2.min()),
                                         h=int(ys2.max()-ys2.min()))

    # ── ON-COURT-BY-45 FILTER: whole-trajectory decision, not per-frame. An
    #    oid that's genuinely on the court -- ON_COURT_BY_CUTOFF_MARGIN
    #    (0.05m), deliberately tight, NOT the usual 0.5m TRACK_MARGIN and NOT
    #    a hard 0m -- by frame 45 (the last legitimate detection point, see
    #    SCAN_FRAMES) is kept for its ENTIRE trajectory, including frames
    #    BEFORE that point (e.g. a player running on: frames of her still
    #    approaching are now included too, not just from the moment she's
    #    confirmed on-court onward) and frames after, even if she goes
    #    off-court again later (that per-frame drop was removed above). An
    #    oid that ISN'T on court by frame 45 -- whether she never reaches it
    #    at all (a coach/bystander planted off to the side the whole clip)
    #    or only reaches it later (a sub with no legitimate detection path
    #    this late, see SCAN_FRAMES) -- is dropped entirely, not just from
    #    the late frames: she's not followed at all, anywhere in the clip.
    LATE_ARRIVAL_CUTOFF_FRAME = max(SCAN_FRAMES)
    ON_COURT_BY_CUTOFF_MARGIN = 0.05
    not_on_court_by_cutoff_oids={
        oid for oid,fr in tracks.items()
        if not any(fi<=LATE_ARRIVAL_CUTOFF_FRAME and _court_bounds_ok(mx,my,getNamed(fi),margin=ON_COURT_BY_CUTOFF_MARGIN)
                   for fi,(mx,my) in fr.items())
    }
    for oid in not_on_court_by_cutoff_oids:
        for f in list(tracks[oid]):
            per_frame[f]=[r for r in per_frame.get(f,[]) if r['oid']!=oid]
        del tracks[oid]

    # ── PASS B: (optional smoothing) + features ──
    sm = smooth_tracks(tracks, SMOOTH_POS_WIN) if do_smooth else None
    rows=[]
    for fi in range(len(ff)):
        recs=per_frame.get(fi,[]); L,R=[],[]
        for r in recs:
            oid=r['oid']
            p = sm[oid][fi] if (do_smooth and fi in sm.get(oid,{})) else r['pos']
            (L if p[0]<NET_X else R).append((round(p[0],3), round(p[1],3)))
        if dedup:
            L,R = dedup_points(L), dedup_points(R)
        row=dict(set=SET_NUMBER, huddle_number=huddle_number, frame=fi,
                 time_s=round(fi/fps,3), fps=round(fps,2), smoothed=int(do_smooth))
        for pre,pts in [('left',L),('right',R)]:
            for k,v in feats(pts).items(): row[f'{pre}_{k}']=v
        row.pop('right_cx',None); row.pop('right_cy',None)
        bb=blob_by_frame.get(fi, {'left':dict(area=0,hull=np.nan,w=0,h=0),
                                   'right':dict(area=0,hull=np.nan,w=0,h=0)})
        for side in ('left','right'):
            row[f'{side}_blob_px']      = bb[side]['area']   # union silhouette area (px)
            row[f'{side}_blob_hull_px'] = bb[side]['hull']   # convex hull of the blob (px)
            row[f'{side}_blob_w']       = bb[side]['w']      # blob bbox width  (px)
            row[f'{side}_blob_h']       = bb[side]['h']      # blob bbox height (px)
        row['left_points']=json.dumps(L); row['right_points']=json.dumps(R)   # extrapolated (mx,my) per player
        rows.append(row)

    print(f"    feet: extrapolated={n_extrap_total}   fragments(kept)={n_frag_total}   injected={n_new}")
    df = pd.DataFrame(rows)
    if return_kept:
        kept_points_by_frame = {fi: {r['oid']: r['pos'] for r in per_frame.get(fi, [])} for fi in range(len(ff))}
        return df, kept_points_by_frame
    return df


# ── homography jitter check ──
# The 8 detected registration points feeding build_homography (net_far/
# net_close aren't detected directly -- they're the midpoint of the two
# attack-line points on their respective side, recomputed below to mirror
# run_pipeline's own logic).
JITTER_BASE_KEYS = BASE_COURT_KEYS

def jitter_named_points(named, jitter_px, rng):
    """Add uniform pixel noise to each detected registration point and recompute the derived net points."""
    jittered = dict(named)
    for k in JITTER_BASE_KEYS:
        v = named.get(k)
        if v is None:
            continue
        p = np.asarray(v, float).flatten()[:2]
        jittered[k] = p + rng.uniform(-jitter_px, jitter_px, size=2)
    alf, arf = jittered.get("attack_left_far"), jittered.get("attack_right_far")
    alc, arc = jittered.get("attack_left_close"), jittered.get("attack_right_close")
    jittered["net_far"] = (alf + arf) / 2 if (alf is not None and arf is not None) else None
    jittered["net_close"] = (alc + arc) / 2 if (alc is not None and arc is not None) else None
    return jittered


def run_homography_jitter_check(video_path, huddle_number, n_draws, jitter_px, out_dir, seed=None):
    """Track once, then jitter the calibration points and refit to gauge homography sensitivity."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print("    computing base calibration (probe-frame line detection)...")
    H_base, named_base = compute_static_homography(video_path)
    if H_base is None:
        print("    FAILED: no homography could be detected on this clip's probe frames "
              "-- abort jitter check")
        return None
    detected_keys = [k for k in JITTER_BASE_KEYS if named_base.get(k) is not None]
    print(f"    base calibration: {len(detected_keys)}/{FULL_REG_POINTS} registration points "
          f"detected -> {detected_keys}")

    FRAME_DIR = tempfile.mkdtemp(prefix=f"vb_jit_h{huddle_number}_")
    cap = cv2.VideoCapture(video_path); nf = 0
    while True:
        ret, fb = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(FRAME_DIR, f"{nf:05d}.jpg"), fb); nf += 1
    cap.release()
    ff = sorted(os.listdir(FRAME_DIR))

    scan_cands = {}
    for F in SCAN_FRAMES:
        F = min(F, len(ff) - 1)
        scan_cands[F] = detect_masks_at(FRAME_DIR, F, SAM3_PROMPT)
        print(f"    scan frame {F}: {len(scan_cands[F])} candidates")

    sid = video_predictor.handle_request(dict(type="start_session", resource_path=FRAME_DIR))["session_id"]
    try:
        tracked = _track_huddle_session(
            sid, FRAME_DIR, ff, lambda fi: H_base, lambda fi: named_base,
            huddle_number, fps, scan_cands,
        )
    finally:
        video_predictor.handle_request(dict(type="close_session", session_id=sid))
        shutil.rmtree(FRAME_DIR, ignore_errors=True); torch.cuda.empty_cache(); gc.collect()

    print(f"    tracking done -- reprojecting through {n_draws} jittered homographies "
          f"(no further SAM3 calls) ...")

    rng = np.random.default_rng(seed)
    import create_additional_features as caf

    feat_runs = []
    for run_i in range(1, n_draws + 1):
        jittered_named = jitter_named_points(named_base, jitter_px, rng)
        H_jit = build_homography(jittered_named)
        if H_jit is None:
            print(f"    draw {run_i}/{n_draws}: jittered points failed to fit a homography -- skipped")
            continue
        df = _featurize_tracked(tracked, ff, lambda fi, H=H_jit: H, lambda fi, n=jittered_named: n,
                                 huddle_number, fps)
        df.to_csv(out_dir / f"draw_{run_i}_raw.csv", index=False)
        feats_df = caf.huddle_formation_features(df, sides=("left", "right"))
        feats_df.insert(0, "draw", run_i)
        feat_runs.append(feats_df)

    if not feat_runs:
        print("\n=== jitter check: every draw failed, nothing to compare ===")
        return None

    all_feats = pd.concat(feat_runs, ignore_index=True)
    all_feats.to_csv(out_dir / "features_by_draw.csv", index=False)

    skip_cols = {"draw", "set", "huddle_number", "huddle_label"}
    numeric_cols = [c for c in all_feats.columns
                     if c not in skip_cols and pd.api.types.is_numeric_dtype(all_feats[c])]
    summary = all_feats[numeric_cols].agg(["mean", "std"]).T
    summary["cv"] = (summary["std"] / summary["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    summary.to_csv(out_dir / "jitter_summary.csv")

    pd.set_option("display.width", 160)
    print(f"\n=== homography jitter check: {len(feat_runs)}/{n_draws} successful draw(s), "
          f"+-{jitter_px}px noise on {len(detected_keys)} registration points ===")
    print(summary.round(4).to_string())

    # create_additional_features.py works entirely in feet/ft^2 (see its
    # METERS_TO_FEET), never meters - these columns (e.g. left_huddle_gyration_closeness_ft)
    # come back from caf.huddle_formation_features already converted, so the
    # jitter-noise threshold below is 0.05m expressed in feet, not 0.05 feet.
    GYRATION_JITTER_THRESHOLD_FT = 0.05 * 3.280839895013123
    gyration_cols = [c for c in summary.index if "gyration_closeness" in c]
    closeness_cols = [c for c in summary.index if "closeness" in c]
    print(f"\n--- gyration_closeness: SD below {GYRATION_JITTER_THRESHOLD_FT:.4f}ft (0.05m)? ---")
    for c in gyration_cols:
        sd = summary.loc[c, "std"]
        print(f"  {c}: SD={sd:.4f}ft -> {'YES' if sd < GYRATION_JITTER_THRESHOLD_FT else 'NO'}")
    print("\n--- all closeness metrics (mean +- SD, feet/ft^2) ---")
    for c in closeness_cols:
        print(f"  {c}: {summary.loc[c,'mean']:.4f} +- {summary.loc[c,'std']:.4f}")

    print(f"\nSaved per-draw raw tracking, per-draw features, and summary to {out_dir}")
    return summary


# ── per-game config, read from parameters.csv (joined by team-name tokens) ──
def load_game_config(matchup, parameters_csv):
    params = pd.read_csv(parameters_csv)
    params.columns = [c.strip() for c in params.columns]
    game_id = stimulus_map.find_by_team_tokens(matchup, list(params["game_id"]))
    prow = params.loc[params["game_id"] == game_id].iloc[0]
    return dict(
        static_camera=str(prow["camera_static"]).strip().upper() == "TRUE",
        number_sets=int(prow["number_sets"]),
    )


# ── one matchup: every set's huddles -> <features_out_root>/<matchup>/all_sets.csv ──
def process_game(matchup, drive_root, parameters_csv, sets_filter=None, features_out_root=DEFAULT_FEATURES_ROOT):
    global SET_NUMBER, STATIC_CAMERA, LAST_STATIC_H, LAST_STATIC_NAMED, FORCE_MANUAL_HOMOGRAPHY, FORCE_LOCKED_HOMOGRAPHY

    cfg = load_game_config(matchup, parameters_csv)
    STATIC_CAMERA = cfg["static_camera"]
    LAST_STATIC_H, LAST_STATIC_NAMED = None, None
    # Reset every call -- otherwise a manual-override/locked-homography matchup
    # processed earlier in the same run would leave these True for every
    # normal matchup after it.
    FORCE_MANUAL_HOMOGRAPHY = matchup in MANUAL_HOMOGRAPHY_MATCHUPS
    FORCE_LOCKED_HOMOGRAPHY = matchup in LOCKED_HOMOGRAPHY_SOURCE_HUDDLE

    huddle_dir = drive_root / matchup / "huddles"
    out_dir = features_out_root / matchup
    ckpt_dir = out_dir / "_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Legacy checkpoint location from before --features-out-root was split out
    # from --drive-root: <drive_root>/<matchup>/huddle_features/_checkpoints/.
    # Any matchup tracked before that change has its real, already-computed
    # checkpoints sitting there, not under ckpt_dir - checked as a fallback
    # below (and copied into ckpt_dir on the spot) so already-done work is
    # never silently redone just because the output location moved.
    legacy_ckpt_dir = drive_root / matchup / "huddle_features" / "_checkpoints"

    all_files = sorted(
        glob.glob(os.path.join(huddle_dir, "huddle_*_set_*.mp4")),
        key=lambda p: tuple(int(x) for x in re.search(r"huddle_(\d+)_set_(\d+)", p).groups()),
    )
    if not all_files:
        print(f"\n=== {matchup}: no huddle clips in {huddle_dir} (cut the huddle clips first) -- skipping ===")
        return

    sets_available = sorted({int(re.search(r"_set_(\d+)", p).group(1)) for p in all_files})
    sets_to_run = sets_filter if sets_filter else list(range(1, cfg["number_sets"] + 1))
    missing = [s for s in sets_to_run if s not in sets_available]
    if missing:
        print(f"\n=== {matchup}: no clips yet for set(s) {missing} -- skipping those ===")
    sets_to_run = [s for s in sets_to_run if s in sets_available]
    if not sets_to_run:
        print(f"\n=== {matchup}: none of the requested sets have clips yet -- skipping ===")
        return

    print(f"\n=== {matchup}: {len(all_files)} huddle clip(s) found, running set(s) {sets_to_run} "
          f"(static_camera={STATIC_CAMERA}) ===")

    if matchup in MANUAL_HOMOGRAPHY_MATCHUPS:
        # Automatic line-detection can't reliably find enough base points for
        # this matchup's camera framing -- use the hand-labeled points instead
        # of prescanning, and force STATIC_CAMERA so this single calibration
        # gets reused for every huddle (see prepare_huddle_frames's
        # STATIC_CAMERA branch).
        STATIC_CAMERA = True
        LAST_STATIC_H = build_homography(MANUAL_HOMOGRAPHY_POINTS)
        LAST_STATIC_NAMED = MANUAL_HOMOGRAPHY_POINTS
        print(f"    using manually-labeled homography ({n_named_base(LAST_STATIC_NAMED)}/"
              f"{FULL_REG_POINTS} base points) -> seeded as this matchup's fallback")
    elif matchup in LOCKED_HOMOGRAPHY_SOURCE_HUDDLE:
        # Automatic per-huddle detection intermittently wins the STATIC_CAMERA
        # ratchet with a confidently-wrong calibration for this matchup's
        # camera framing (see LOCKED_HOMOGRAPHY_SOURCE_HUDDLE's comment) --
        # source ONE real calibration from a huddle clip already confirmed
        # good and force STATIC_CAMERA so it's reused for every huddle instead
        # of competing with each huddle's own (unreliable) detection.
        STATIC_CAMERA = True
        source = LOCKED_HOMOGRAPHY_SOURCE_HUDDLE[matchup]
        if isinstance(source, tuple):
            # Borrowing a calibration from a DIFFERENT matchup's huddle
            # folder (same physical camera/court, no tracked output of this
            # matchup's own yet to pick a source huddle from).
            source_matchup, source_name = source
            source_path = drive_root / source_matchup / "huddles" / source_name
            source_label = f"{source_name} (borrowed from {source_matchup})"
        else:
            source_name = source
            source_path = huddle_dir / source_name
            source_label = source_name
        H, named = compute_static_homography(str(source_path))
        if H is None:
            raise RuntimeError(
                f"{matchup}: locked-homography source clip {source_label} "
                f"produced no homography -- pick a different confirmed-good huddle"
            )
        LAST_STATIC_H, LAST_STATIC_NAMED = H, named
        print(f"    using locked homography from {source_label} "
              f"({n_named_base(LAST_STATIC_NAMED)}/{FULL_REG_POINTS} base points) "
              f"-> forced for every huddle")
    elif STATIC_CAMERA:
        # The camera doesn't move for the whole matchup, so the single BEST
        # calibration found anywhere in the game is a valid fallback for EVERY
        # huddle, not just ones that come after it in processing order. Without
        # this upfront scan, a degraded (or totally failed) early huddle would
        # otherwise poison/starve the sequential LAST_STATIC_H chain for
        # everything that follows it until something better happens to show up
        # later -- e.g. huddle 1 fails outright, huddle 2 then has nothing to
        # fall back to and keeps its own weak detection, which then becomes the
        # "best so far" for huddle 3, etc. Cheap: compute_static_homography only
        # does the 3-probe-frame line detection, no SAM3 tracking. Deliberately
        # does NOT stop early at FULL_REG_POINTS (theoretical max, all 8 base
        # keys) -- some camera framings structurally never show 1-2 of the base
        # points in ANY frame (e.g. a corner permanently out of shot), so the
        # real achievable ceiling for a given matchup can be lower than 8 and
        # stopping early on that assumption would settle for less than the true
        # best available.
        print(f"    prescanning all {len(all_files)} clip(s) for the best calibration "
              f"to seed as this matchup's static-camera fallback...")
        for vp in all_files:
            H, named = compute_static_homography(vp)
            if H is not None and n_named_base(named) > n_named_base(LAST_STATIC_NAMED):
                LAST_STATIC_H, LAST_STATIC_NAMED = H, named
        if LAST_STATIC_NAMED is not None:
            print(f"    best calibration found: {n_named_base(LAST_STATIC_NAMED)}/{FULL_REG_POINTS} "
                  f"base points -> seeded as this matchup's fallback")
        else:
            print(f"    no clip in this matchup produced ANY homography during prescan -- "
                  f"no fallback available; each huddle will rely on its own detection only")

    for set_number in sets_to_run:
        SET_NUMBER = set_number
        files = [p for p in all_files if int(re.search(r"_set_(\d+)", p).group(1)) == set_number]
        for vp in files:
            hn = int(re.search(r"huddle_(\d+)_set", vp).group(1))
            ckpt = ckpt_dir / f"huddle_{hn}_set_{set_number}.csv"
            if not ckpt.is_file():
                legacy_ckpt = legacy_ckpt_dir / ckpt.name
                if legacy_ckpt.is_file():
                    shutil.copy2(legacy_ckpt, ckpt)
                    print(f"=== {matchup} set {set_number} huddle {hn}: found legacy checkpoint, "
                          f"migrated -> {ckpt} ===")
            if ckpt.is_file():
                print(f"=== {matchup} set {set_number} huddle {hn}: already done, skip ===")
                continue
            print(f"=== {matchup} set {set_number} huddle {hn}: {os.path.basename(vp)} ===")
            try:
                df = process_one_huddle(vp, hn)
                if df is not None:
                    df.to_csv(ckpt, index=False)
                    print(f"    saved {len(df)} frames x {df.shape[1]} cols -> {ckpt}")
            except Exception as e:
                print("    FAILED:", e)
                traceback.print_exc()

    done = sorted(
        glob.glob(os.path.join(ckpt_dir, "huddle_*_set_*.csv")),
        key=lambda p: tuple(int(x) for x in re.search(r"huddle_(\d+)_set_(\d+)", p).groups()),
    )
    if not done:
        print(f"=== {matchup}: nothing tracked yet, no CSV written ===")
        return
    big = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
    out_csv = out_dir / "all_sets.csv"
    big.to_csv(out_csv, index=False)
    print(f"=== {matchup}: saved {out_csv} ({len(big)} rows across {len(done)} huddle checkpoints) ===")


# ── prune-and-rerun: delete existing checkpoints whose STATIC_CAMERA calibration
#    re-detects fewer than --rerun-below-points base registration points, so the
#    normal process_game loop right after this reprocesses just those huddles --
#    now benefiting from the last-good-homography fallback in process_one_huddle
#    instead of the degraded per-huddle detection that produced them originally.
#    Cheap: compute_static_homography only does the 3-probe-frame line detection,
#    no SAM3 tracking session. ──
def prune_low_point_checkpoints(matchup, drive_root, parameters_csv, features_out_root, min_points):
    cfg = load_game_config(matchup, parameters_csv)
    if not cfg["static_camera"]:
        print(f"=== {matchup}: not STATIC_CAMERA, --rerun-below-points is a no-op -- skipping ===")
        return
    huddle_dir = Path(drive_root) / matchup / "huddles"
    ckpt_dir = Path(features_out_root) / matchup / "_checkpoints"
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "huddle_*_set_*.csv")))
    if not ckpts:
        print(f"=== {matchup}: no existing checkpoints under {ckpt_dir} -- nothing to prune ===")
        return
    n_pruned = 0
    for ckpt in ckpts:
        m = re.search(r"huddle_(\d+)_set_(\d+)", os.path.basename(ckpt))
        hn, sn = m.groups()
        vp = huddle_dir / f"huddle_{hn}_set_{sn}.mp4"
        if not vp.is_file():
            print(f"    huddle {hn} set {sn}: checkpoint exists but {vp} is missing -- skip")
            continue
        H, named = compute_static_homography(str(vp))
        n_base = n_named_base(named)
        if n_base < min_points:
            print(f"    huddle {hn} set {sn}: {n_base}/{FULL_REG_POINTS} base points "
                  f"-> deleting checkpoint, will rerun")
            os.remove(ckpt)
            n_pruned += 1
    print(f"=== {matchup}: pruned {n_pruned}/{len(ckpts)} checkpoint(s) with "
          f"< {min_points} base registration points ===")


def bad_huddle_fraction(matchup, features_out_root):
    """Fraction of checkpoints where one side was never tracked beyond 1 point."""
    ckpt_dir = Path(features_out_root) / matchup / "_checkpoints"
    ckpts = glob.glob(str(ckpt_dir / "huddle_*_set_*.csv"))
    if not ckpts:
        return 0.0
    n_bad = 0
    for ckpt in ckpts:
        df = pd.read_csv(ckpt, usecols=["left_n", "right_n"])
        if df["left_n"].max() <= 1 or df["right_n"].max() <= 1:
            n_bad += 1
    return n_bad / len(ckpts)


# ── auto-verify a BORROWED locked homography right after it runs, and
#    self-correct if it was wrong -- no manual check-then-rerun needed ──
def verify_and_retry_borrowed_homography(matchup, drive_root, parameters_csv,
                                          features_out_root, sets_filter,
                                          bad_fraction_threshold=0.5):
    """For borrowed calibrations, verify the result and retry on failure."""
    source = LOCKED_HOMOGRAPHY_SOURCE_HUDDLE.get(matchup)
    if not isinstance(source, tuple):
        return

    frac = bad_huddle_fraction(matchup, features_out_root)
    print(f"    borrowed-homography check for {matchup}: {frac:.0%} of huddles "
          f"show one-sided detection collapse (threshold {bad_fraction_threshold:.0%})")
    if frac <= bad_fraction_threshold:
        print(f"    looks consistent with {source[0]}'s camera -> keeping this run")
        return

    print(f"    borrowed calibration from {source[0]} looks WRONG for {matchup}'s "
          f"camera -> deleting this run's checkpoints and retrying with "
          f"{matchup}'s own auto-detection instead")
    ckpt_dir = Path(features_out_root) / matchup / "_checkpoints"
    for ckpt in glob.glob(str(ckpt_dir / "huddle_*_set_*.csv")):
        os.remove(ckpt)

    del LOCKED_HOMOGRAPHY_SOURCE_HUDDLE[matchup]
    try:
        process_game(matchup, drive_root, parameters_csv, sets_filter,
                     features_out_root=features_out_root)
    finally:
        LOCKED_HOMOGRAPHY_SOURCE_HUDDLE[matchup] = source

    frac2 = bad_huddle_fraction(matchup, features_out_root)
    print(f"    after retry with {matchup}'s own auto-detection: {frac2:.0%} of "
          f"huddles still show detection collapse")


if __name__ == "__main__":
    args = parse_args()

    if args.install_deps:
        install_dependencies()
        print("\nDependencies installed. Re-run this command without --install-deps "
              "in a fresh process (e.g. a new shell/Quest job) to load them cleanly.")
        sys.exit(0)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from scipy.interpolate import UnivariateSpline
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from skimage.morphology import skeletonize

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    hf_login(args.hf_token)

    model = load_sam3_image_model()

    patch_sam3_propagation_classifier()

    from sam3.model_builder import build_sam3_video_predictor
    video_predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    print("video predictor built")

    if args.debug_homography_frame:
        cap = cv2.VideoCapture(args.debug_homography_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        print("frame read ok:", ret, frame.shape if ret else None)
        fitted_lines, named = run_pipeline(frame)
        print(f"\n{len(fitted_lines)} fitted lines detected:")
        for ln in fitted_lines:
            print(f"  id={ln['id']} type={ln['type']} label={ln['label']} "
                  f"score={ln['score']:.3f} angle={ln['angle']:.1f}")
        print("\nnamed court points:")
        for k, v in named.items():
            print(f"  {k}: {v}")
        H = build_homography(named)
        print("\nhomography found:", H is not None, "n_named:", n_named(named))
        sys.exit(0)

    if args.repeat_huddle:
        repeat_path = str(args.repeat_huddle)
        m = re.search(r"huddle_(\d+)_set_(\d+)", os.path.basename(repeat_path))
        huddle_number = int(m.group(1)) if m else 1
        SET_NUMBER = int(m.group(2)) if m else 1
        if not m:
            print(f"    (couldn't parse huddle/set number from filename, "
                  f"defaulting to huddle {huddle_number} set {SET_NUMBER})")
        if args.matchup:
            cfg = load_game_config(args.matchup, args.parameters_csv)
            STATIC_CAMERA = cfg["static_camera"]

        out_dir = args.repeat_out_dir or (
            Path(repeat_path).resolve().parent.parent / "huddle_features" /
            "_repeatability" / Path(repeat_path).stem
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== repeatability check: {os.path.basename(repeat_path)} "
              f"x{args.repeat_n} runs (static_camera={STATIC_CAMERA}) -> {out_dir} ===")

        import create_additional_features as caf

        feat_runs = []
        for run_i in range(1, args.repeat_n + 1):
            print(f"\n--- run {run_i}/{args.repeat_n} ---")
            df = process_one_huddle(repeat_path, huddle_number)
            if df is None:
                print("    FAILED: no output for this run (e.g. no homography) -- skipped")
                continue
            df.to_csv(out_dir / f"run_{run_i}_raw.csv", index=False)
            feats_df = caf.huddle_formation_features(df, sides=("left", "right"))
            feats_df.insert(0, "run", run_i)
            feat_runs.append(feats_df)

        if not feat_runs:
            print("\n=== repeatability check: every run failed, nothing to compare ===")
            sys.exit(1)

        all_feats = pd.concat(feat_runs, ignore_index=True)
        all_feats.to_csv(out_dir / "features_by_run.csv", index=False)

        skip_cols = {"run", "set", "huddle_number", "huddle_label"}
        numeric_cols = [c for c in all_feats.columns
                         if c not in skip_cols and pd.api.types.is_numeric_dtype(all_feats[c])]
        summary = all_feats[numeric_cols].agg(["mean", "std"]).T
        summary["cv"] = (summary["std"] / summary["mean"].abs()).replace([np.inf, -np.inf], np.nan)
        summary = summary.sort_values("cv", ascending=False)
        summary.to_csv(out_dir / "variability_summary.csv")

        pd.set_option("display.width", 160)
        print(f"\n=== variability across {len(feat_runs)}/{args.repeat_n} successful "
              f"run(s), sorted by coefficient of variation (std / |mean|) ===")
        print(summary.round(4).to_string())
        print(f"\nSaved per-run raw tracking, per-run features, and this summary to {out_dir}")
        sys.exit(0)

    if args.jitter_homography:
        jitter_path = str(args.jitter_homography)
        m = re.search(r"huddle_(\d+)_set_(\d+)", os.path.basename(jitter_path))
        huddle_number = int(m.group(1)) if m else 1
        SET_NUMBER = int(m.group(2)) if m else 1
        if not m:
            print(f"    (couldn't parse huddle/set number from filename, "
                  f"defaulting to huddle {huddle_number} set {SET_NUMBER})")
        if args.matchup:
            cfg = load_game_config(args.matchup, args.parameters_csv)
            STATIC_CAMERA = cfg["static_camera"]
        if not STATIC_CAMERA:
            print("    jitter check only supports STATIC_CAMERA huddles (a single "
                  "per-video homography) -- this matchup's camera_static is FALSE, "
                  "aborting. Pass --matchup so this can be checked, if you didn't.")
            sys.exit(1)

        out_dir = args.jitter_out_dir or (
            Path(jitter_path).resolve().parent.parent / "huddle_features" /
            "_jitter" / Path(jitter_path).stem
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== homography jitter check: {os.path.basename(jitter_path)} "
              f"x{args.jitter_n} draws, +-{args.jitter_px}px -> {out_dir} ===")

        run_homography_jitter_check(
            jitter_path, huddle_number, args.jitter_n, args.jitter_px, out_dir,
            seed=args.jitter_seed,
        )
        sys.exit(0)

    rows = stimulus_map.load_rows(args.stimulus_map)
    rows = [r for r in rows if r.get("type", "").strip() == "huddle_video"]
    matchups = sorted(stimulus_map.group_by_matchup(rows))
    if args.matchup:
        tokens = [t.strip().lower() for t in args.matchup.split(",") if t.strip()]
        matchups = [m for m in matchups if any(t in m.lower() for t in tokens)]
        assert matchups, f"No matchup matched {args.matchup!r}"

    sets_filter = [int(s) for s in args.sets.split(",")] if args.sets else None

    print(f"Processing {len(matchups)} matchup(s): {matchups}")
    failed_matchups = []
    for matchup in matchups:
        try:
            if args.rerun_below_points is not None:
                prune_low_point_checkpoints(matchup, args.drive_root, args.parameters_csv,
                                             args.features_out_root, args.rerun_below_points)
            process_game(matchup, args.drive_root, args.parameters_csv, sets_filter,
                         features_out_root=args.features_out_root)
            verify_and_retry_borrowed_homography(matchup, args.drive_root, args.parameters_csv,
                                                  args.features_out_root, sets_filter)
        except Exception as e:
            # One matchup's setup failure (e.g. a bad LOCKED_HOMOGRAPHY_SOURCE_HUDDLE
            # clip) used to crash the whole job, silently abandoning every
            # matchup still queued after it -- happened for real, wasting the
            # remaining budget on games that had nothing wrong with them.
            # Log and move on instead.
            print(f"=== {matchup}: FAILED, skipping to next matchup -- {e} ===")
            traceback.print_exc()
            failed_matchups.append(matchup)

    if failed_matchups:
        print(f"\n=== {len(failed_matchups)} matchup(s) failed and were skipped: "
              f"{failed_matchups} ===")
