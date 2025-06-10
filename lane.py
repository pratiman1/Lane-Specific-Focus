import cv2
import math
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import time
import csv
from datetime import datetime, timedelta

# --- Configurable Variables ---
MODEL_PATH = ''  # Path to the YOLO model
CSV_FILE_PATH = ''  # Path to save the CSV file with coordinates
DESIRED_CLASS_ID = 0  # Actual class ID for 'Lane' in YOLOv8 model
REFERENCE_LINE_INTERVAL = 40  # Interval (in pixels) for green reference lines
TOLERANCE = 5  # Tolerance for detecting red lines within yellow contours
TRANSPARENCY = 0.5  # Transparency for blending original and annotated frames
FPS_FONT_SCALE = 0.6  # Font scale for displaying FPS
FONT_COLOR = (0, 255, 0)  # Font color for FPS display
LANE_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Lane colors for annotation

# --- Initialize YOLO model and CSV ---
model = YOLO(MODEL_PATH)
top_midpoint, bottom_midpoint = (-100, -100), (-100, -100)

# CSV File Setup
csv_header = ['Time', 'X', 'Y', 'Z', 'Frame']
with open(CSV_FILE_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)


# --- Helper Functions ---
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def get_current_ist_time():
    utc_time = datetime.utcnow()
    ist_time = utc_time + timedelta(hours=5, minutes=30)
    return ist_time.strftime('%Y-%m-%d %H:%M:%S.%f')


def convert_box_to_polygon(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)  # Extract coordinates
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Rectangle as polygon points


def draw_reference_lines(frame, interval=REFERENCE_LINE_INTERVAL, color=(0, 255, 0), thickness=2):
    height, width = frame.shape[:2]
    green_lines = []
    for y in range(0, height, interval):
        cv2.line(frame, (0, y), (width, y), color, thickness)
        green_lines.append(y)
    return green_lines


def draw_red_lines_within_yellow(frame, green_lines, yellow_contours, tolerance=TOLERANCE):
    height, width = frame.shape[:2]
    for y in green_lines:
        mask_img = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask_img, yellow_contours, -1, 255, thickness=cv2.FILLED)

        red_line_y = y
        if red_line_y - tolerance >= 0 and red_line_y + tolerance < height:
            left_bound = width
            right_bound = 0
            for i in range(width):
                if mask_img[red_line_y, i] == 255:
                    left_bound = min(left_bound, i)
                    right_bound = max(right_bound, i)

            if right_bound > left_bound:
                cv2.line(frame, (left_bound, red_line_y), (right_bound, red_line_y), (0, 0, 255), 2)


def compute_real_world_coordinates(x, y, depth_frame, intrinsics):
    dist = depth_frame.get_distance(x, y)
    if dist > 0:
        Xtemp = dist * (x - intrinsics.ppx) / intrinsics.fx
        Ytemp = dist * (y - intrinsics.ppy) / intrinsics.fy
        Ztemp = dist
        coordinates_text = f"({Xtemp:.3f}, {Ytemp:.3f}, {Ztemp:.3f})"
        return Xtemp, Ytemp, Ztemp, coordinates_text
    else:
        return None, None, None, "Invalid depth"


def draw_central_point(frame, green_lines, yellow_contours, depth_frame, intrinsics, frame_number):
    height, width = frame.shape[:2]
    for y in green_lines:
        mask_img = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask_img, yellow_contours, -1, 255, thickness=cv2.FILLED)

        red_line_y = y
        if red_line_y - 5 >= 0 and red_line_y + 5 < height:
            left_bound = width
            right_bound = 0
            for i in range(width):
                if mask_img[red_line_y, i] == 255:
                    left_bound = min(left_bound, i)
                    right_bound = max(right_bound, i)

            if right_bound > left_bound:
                midpoint_x = (left_bound + right_bound) // 2
                midpoint_y = red_line_y

                cv2.circle(frame, (midpoint_x, midpoint_y), 5, (0, 165, 255), -1)
                Xtemp, Ytemp, Ztemp, coordinates_text = compute_real_world_coordinates(midpoint_x, midpoint_y,
                                                                                       depth_frame, intrinsics)

                if coordinates_text != "Invalid depth":
                    cv2.putText(frame, coordinates_text, (midpoint_x + 10, midpoint_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    save_to_csv(cv2.getTickCount(), Xtemp, Ytemp, Ztemp, frame_number)


def calculate_area_in_region(frame, yellow_contours, green_lines):
    height, width = frame.shape[:2]
    area = 0
    for y in green_lines:
        mask_img = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask_img, yellow_contours, -1, 255, thickness=cv2.FILLED)

        left_bound, right_bound = width, 0
        for x in range(width):
            if mask_img[y, x] == 255:
                left_bound = min(left_bound, x)
                right_bound = max(right_bound, x)

        if right_bound > left_bound:
            area += (right_bound - left_bound)

    return area


def display_area(frame, area, position=None, font_scale=0.7, color=(255, 0, 0), thickness=2):
    if position is None:
        position = (frame.shape[1] - 200, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Area: {area} px", position, font, font_scale, color, thickness)
    return frame


def save_to_csv(tick_count, X, Y, Z, frame_number):
    timestamp = tick_count / cv2.getTickFrequency()
    ist_time = get_current_ist_time()
    X, Y, Z = round(X, 3), round(Y, 3), round(Z, 3)

    with open(CSV_FILE_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ist_time, X, Y, Z, frame_number])


def highlight_and_draw_central_lines(frame, depth_frame, intrinsics, transparency=TRANSPARENCY):
    annotated_frame = frame.copy()
    current_lane_color_index = 0

    green_lines = draw_reference_lines(annotated_frame)
    results = model(frame, verbose=False)

    yellow_contours = []
    for result in results:
        if result.masks is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            points1 = list(points[0])

            cv2.line(annotated_frame, tuple(map(int, top_midpoint)), tuple(map(int, bottom_midpoint)), (255, 0, 0), 5)
            cv2.fillPoly(annotated_frame, [np.array(points1, dtype=np.int32)], LANE_COLORS[1], 50)
            mask_img = np.zeros_like(frame[:, :, 0])
            cv2.fillPoly(mask_img, [np.array(points1, dtype=np.int32)], 255)

            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            yellow_contours.extend(contours)

    area = calculate_area_in_region(frame, yellow_contours, green_lines)
    annotated_frame = display_area(annotated_frame, area)

    draw_red_lines_within_yellow(annotated_frame, green_lines, yellow_contours)
    draw_central_point(annotated_frame, green_lines, yellow_contours, depth_frame, intrinsics, frame_number)

    result_frame = cv2.addWeighted(frame, 1 - transparency, annotated_frame, transparency, 0)
    return result_frame


def display_fps(frame, start_time, fps_font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, color=(0, 255, 0), thickness=2):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), fps_font, font_scale, color, thickness)
    return frame, end_time


# --- Main Execution ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frame_number = 0

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    start_time = time.time()
    result_frame = highlight_and_draw_central_lines(color_image, depth_frame, intrinsics)

    result_frame, start_time = display_fps(result_frame, start_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1
    cv2.imshow('Result', result_frame)

pipeline.stop()
cv2.destroyAllWindows()