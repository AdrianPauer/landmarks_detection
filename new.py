import argparse
import json
import cv2


def redraw_points(state):
    crop_w = state["orig_w"] / state["zoom_level"]
    crop_h = state["orig_h"] / state["zoom_level"]

    state["center_x"] = max(crop_w / 2, min(state["center_x"], state["orig_w"] - crop_w / 2))
    state["center_y"] = max(crop_h / 2, min(state["center_y"], state["orig_h"] - crop_h / 2))

    x1 = int(state["center_x"] - crop_w / 2)
    y1 = int(state["center_y"] - crop_h / 2)
    x2 = int(x1 + crop_w)
    y2 = int(y1 + crop_h)

    crop_img = state["img_orig"][y1:y2, x1:x2].copy()
    img_display = cv2.resize(crop_img, (state["disp_w"], state["disp_h"]), interpolation=cv2.INTER_AREA)

    # Display point count status using state["target_points"]
    target = state["target_points"]
    status_color = (0, 255, 0) if len(state["points"]) == target else (0, 0, 255)
    cv2.putText(
        img_display, f"Points: {len(state['points'])}/{target}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2
    )

    for idx, (ox, oy) in enumerate(state["points"]):
        if x1 <= ox < x2 and y1 <= oy < y2:
            dx = int((ox - x1) / crop_w * state["disp_w"])
            dy = int((oy - y1) / crop_h * state["disp_h"])

            cv2.circle(img_display, (dx, dy), 4, (0, 0, 255), -1)
            cv2.putText(
                img_display, str(idx + 1), (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    cv2.imshow("Mark Points", img_display)


def click_event(event, x, y, flags, param):
    # Extract state from the param passed by OpenCV
    state = param
    target = state["target_points"]
    pan_thresh = state["pan_threshold"]

    crop_w = state["orig_w"] / state["zoom_level"]
    crop_h = state["orig_h"] / state["zoom_level"]
    x1 = state["center_x"] - crop_w / 2
    y1 = state["center_y"] - crop_h / 2

    orig_x = int(x1 + (x / state["disp_w"]) * crop_w)
    orig_y = int(y1 + (y / state["disp_h"]) * crop_h)

    if event == cv2.EVENT_LBUTTONDOWN:
        state["is_lbutton_down"] = True
        state["has_panned"] = False
        state["start_pan_x"], state["start_pan_y"] = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["is_lbutton_down"]:
            dx = x - state["start_pan_x"]
            dy = y - state["start_pan_y"]

            if abs(dx) > pan_thresh or abs(dy) > pan_thresh:
                state["has_panned"] = True

            if state["has_panned"]:
                state["center_x"] -= (dx / state["disp_w"]) * crop_w
                state["center_y"] -= (dy / state["disp_h"]) * crop_h
                state["start_pan_x"], state["start_pan_y"] = x, y
                redraw_points(state)

    elif event == cv2.EVENT_LBUTTONUP:
        state["is_lbutton_down"] = False
        if not state["has_panned"]:
            if len(state["points"]) < target:
                state["points"].append((orig_x, orig_y))
                print(f"Added point #{len(state['points'])}: ({orig_x}, {orig_y})")
                redraw_points(state)
            else:
                print(f"Maximum of {target} points reached! Press 's' to save or 'r' to undo.")

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            state["zoom_level"] = min(30.0, state["zoom_level"] * 1.25)
        else:
            state["zoom_level"] = max(1.0, state["zoom_level"] / 1.25)

        new_crop_w = state["orig_w"] / state["zoom_level"]
        new_crop_h = state["orig_h"] / state["zoom_level"]
        state["center_x"] = orig_x - (x / state["disp_w"]) * new_crop_w + new_crop_w / 2
        state["center_y"] = orig_y - (y / state["disp_h"]) * new_crop_h + new_crop_h / 2

        redraw_points(state)


def get_landmarks(image_path, output_path="points.json", max_dim=800, target_points=16):
    """
    Opens an interactive UI to mark exactly `target_points` on an image.

    Returns:
        list of tuples: The selected (x, y) coordinates if saved, or None if quit.
    """
    state = {
        "points": [],
        "target_points": target_points,
        "pan_threshold": 3,
        "zoom_level": 1.0,
        "center_x": 0.0,
        "center_y": 0.0,
        "is_lbutton_down": False,
        "has_panned": False,
        "start_pan_x": 0,
        "start_pan_y": 0,
        "img_orig": None,
        "orig_w": 0,
        "orig_h": 0,
        "disp_w": 0,
        "disp_h": 0
    }

    # Load original full-resolution image
    state["img_orig"] = cv2.imread(image_path)
    if state["img_orig"] is None:
        print(f"Error: Could not load image at '{image_path}'")
        return None

    state["orig_h"], state["orig_w"] = state["img_orig"].shape[:2]

    # Calculate display dimensions
    if max(state["orig_h"], state["orig_w"]) > max_dim:
        scale_factor = max_dim / float(max(state["orig_h"], state["orig_w"]))
        state["disp_w"] = int(state["orig_w"] * scale_factor)
        state["disp_h"] = int(state["orig_h"] * scale_factor)
    else:
        state["disp_w"], state["disp_h"] = state["orig_w"], state["orig_h"]

    # Initial center
    state["center_x"] = state["orig_w"] / 2
    state["center_y"] = state["orig_h"] / 2

    cv2.namedWindow("Mark Points")
    # Note: We pass `state` into the callback using the `param` argument
    cv2.setMouseCallback("Mark Points", click_event, param=state)

    print("\n--- INSTRUCTIONS ---")
    print(f"You MUST select exactly {target_points} points.")
    print("• Left Click         : Add landmark dot.")
    print("• Left Click & Drag  : Pan around zoomed image.")
    print("• Scroll Wheel       : Zoom in/out at cursor.")
    print("• Press 'r'          : Remove LAST point (undo).")
    print("• Press 'c'          : Clear ALL points.")
    print("• Press 's'          : Save and quit (ONLY works if 16 points are selected).")
    print("• Press 'q'          : Quit without saving.\n")

    redraw_points(state)
    final_points = None

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(state["points"]) == state["target_points"]:
                # Save to JSON
                if output_path:
                    data = {
                        "image": image_path,
                        "original_resolution": [state["orig_w"], state["orig_h"]],
                        "points": state["points"],
                    }
                    with open(output_path, "w") as f:
                        json.dump(data, f, indent=4)
                    print(f"Success! Saved exactly {state['target_points']} points to '{output_path}'.")

                # Store points to return
                final_points = state["points"]
                break
            else:
                print(
                    f"Cannot save: You currently have {len(state['points'])} points. You need exactly {state['target_points']}.")

        elif key == ord("r"):
            if len(state["points"]) > 0:
                removed = state["points"].pop()
                print(f"Removed last point: {removed}")
                redraw_points(state)  # Fixed missing state argument

        elif key == ord("c"):
            state["points"] = []
            print("Cleared all points.")
            redraw_points(state)  # Fixed missing state argument

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()
    return final_points


# # --- Allow script to be run from terminal directly ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Interactive tool to select and save dot coordinates from high-res images."
#     )
#     parser.add_argument("-i", "--image", required=True, help="Path to target image")
#     parser.add_argument("-o", "--output", default="points.json", help="Output JSON path")
#     parser.add_argument("--max-dim", type=int, default=800, help="Max window size")
#     args = parser.parse_args()
#
#     # Run the function and capture the points
#     returned_points = get_landmarks(args.image, args.output, args.max_dim)
#
#     if returned_points:
#         print("\nProcess finished successfully. Points collected:")
#         print(returned_points)