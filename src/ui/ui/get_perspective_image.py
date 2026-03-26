# core/get_perspective_image.py

import cv2
import numpy as np


# -------------------------
# Internal utility: warp to full or inner view
# -------------------------


def _warp_view(
    src_img: np.ndarray,
    H: np.ndarray,
    full: bool = True,
    border_value: tuple[int, int, int] = (255, 0, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp an image using a homography, either to a full bounding view
    or to an approximate inner cropped view.

    Parameters
    ----------
    src_img : np.ndarray
        Source image, shape (H, W, 3) or (H, W).
    H : np.ndarray
        Homography matrix, shape (3, 3).
    full : bool, optional
        If True, use full bounding box of the warped image.
        If False, use an approximate inner crop based on sorted corners.
    border_value : tuple[int, int, int], optional
        Border color used by cv2.warpPerspective for constant border.

    Returns
    -------
    warped : np.ndarray
        Warped image.
    H_shifted : np.ndarray
        Updated homography including the translation that moves the
        chosen bounding box's top-left corner to (0, 0).
    """
    h, w = src_img.shape[:2]

    # 1) Original image corners (x, y)
    corners = np.array(
        [[0, 0],
         [w, 0],
         [w, h],
         [0, h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # 2) Project corners with H
    warped_corners = cv2.perspectiveTransform(corners, H)  # (4,1,2)
    xs = warped_corners[:, 0, 0]
    ys = warped_corners[:, 0, 1]

    if full:
        # Use full bounding box that contains all warped corners
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
    else:
        # Approximate inner crop: use middle two values
        xs_sort = np.sort(xs)
        ys_sort = np.sort(ys)
        min_x, max_x = xs_sort[1], xs_sort[2]
        min_y, max_y = ys_sort[1], ys_sort[2]

    # 3) Output size
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    # 4) Translation: (min_x, min_y) → (0, 0)
    T = np.array(
        [[1.0, 0.0, -float(min_x)],
         [0.0, 1.0, -float(min_y)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_shifted = T @ H

    # 5) Final warp
    warped = cv2.warpPerspective(
        src_img,
        H_shifted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return warped, H_shifted

def _get_charuco_pts(src_img, debug=False):
    aruco = cv2.aruco

    # 1) Dictionary & board definition
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    square_length = 200
    marker_length = 120
    diamond_ids_for_board = np.array([1, 2, 3, 4], dtype=np.int32)

    board = aruco.CharucoBoard(
        (3, 3),
        square_length,
        marker_length,
        dictionary,
        diamond_ids_for_board,
    )

    charuco_detector = aruco.CharucoDetector(board)

    # 2) Detect diamonds and markers
    diamond_corners, diamond_ids, marker_corners, marker_ids = \
        charuco_detector.detectDiamonds(src_img)

    if diamond_corners is None or len(diamond_corners) == 0:
        raise RuntimeError("Failed to detect ChArUco diamond markers.")
    if marker_corners is None or len(marker_corners) < 4:
        raise RuntimeError("Less than 4 ArUco markers detected; 20-point matching requires 4 markers.")

    if debug:
        dbg = src_img.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
        aruco.drawDetectedDiamonds(dbg, diamond_corners, diamond_ids)
        cv2.imwrite("./test/detected_aruco.png", dbg)

    # 3) src_pts: diamond (4 points) + first 4 markers (4*4 points) = 20 points
    src_pts_list: list[np.ndarray] = []
    diamond_pts = diamond_corners[0].reshape(4, 2)
    src_pts_list.append(diamond_pts)
    for mc in marker_corners[:4]:
        src_pts_list.append(mc.reshape(4, 2))

    if debug:
        dbg2 = src_img.copy()
        for (x, y) in diamond_pts:
            cv2.circle(dbg2, (int(x), int(y)), 5, (0, 0, 255), -1)
        for mc in marker_corners[:4]:
            for (x, y) in mc.reshape(-1, 2):
                cv2.circle(dbg2, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.imwrite("./test/detected_20points.png", dbg2)

    return np.vstack(src_pts_list).astype(np.float32)  # (20, 2)

# -------------------------
# Main: perspective correction using ChArUco
# -------------------------
def get_perspective_img(
    src_img: np.ndarray,
    aruco_size_cm: float = 6.1,
    ref_marker_size_px: int = 300,
    debug: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute top-view (perspective corrected) image using a ChArUco board.

    Parameters
    ----------
    src_img : np.ndarray
        Input image containing the ChArUco board, shape (H, W, 3) or (H, W).
    aruco_size_cm : float, optional
        Physical size of the reference ArUco marker (edge length) in centimeters.
    ref_marker_size_px : int, optional
        Reference marker size in pixels (used to scale the destination coordinates).
    debug : bool, optional
        If True, save and show intermediate debug images.

    Returns
    -------
    warped : np.ndarray
        Perspective-corrected (top-view) image.
    pixel_scale : float
        Physical size per pixel, in centimeters per pixel.  
    H_shifted : np.ndarray
        Perspective Matrix.
    """
    
    src_pts = _get_charuco_pts(src_img)
    # 4) dst_pts: theoretical coordinates of the 20 points
    dst_pts = np.array(
        [
            [200.0, 100.0],
            [200.0, 200.0],
            [100.0, 200.0],
            [100.0, 100.0],

            [280.0, 120.0],
            [280.0, 180.0],
            [220.0, 180.0],
            [220.0, 120.0],

            [80.0, 120.0],
            [80.0, 180.0],
            [20.0, 180.0],
            [20.0, 120.0],

            [180.0, 20.0],
            [180.0, 80.0],
            [120.0, 80.0],
            [120.0, 20.0],

            [180.0, 220.0],
            [180.0, 280.0],
            [120.0, 280.0],
            [120.0, 220.0],
        ],
        dtype=np.float64,
    )

    dst_pts *= ref_marker_size_px / 300.0

    # 5) Homography estimation
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if H is None:
        raise RuntimeError("Failed to compute homography.")

    # 6) Warp (full view)
    warped, H_shifted = _warp_view(src_img, H, full=True)

    # 7) Pixel scale (cm/pixel)
    pixel_scale = aruco_size_cm / float(ref_marker_size_px)

    if debug:
        print(f"src shape: {src_img.shape}")
        print(f"warped shape: {warped.shape}")
        cv2.imshow("Origin Img (debug)", src_img)
        cv2.imshow("Top-View (debug)", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped, pixel_scale, H_shifted, src_pts


if __name__ == "__main__":
    SRC_IMG_PATH = "./data/pre_persepective.jpg"
    img = cv2.imread(SRC_IMG_PATH)
    if img is None:
        raise SystemExit("Error: image not found.")

    get_perspective_img(img, ref_marker_size_px=100, debug=True)