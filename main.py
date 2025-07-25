import numpy as np

def iou(box1, box2):
    """
    Tính toán Intersection over Union (IoU) giữa hai hộp giới hạn.
    Box format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    # Tránh chia cho 0
    if union_area == 0:
        return 0
    return inter_area / union_area

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Thực hiện Non-Maximum Suppression (NMS) để loại bỏ các hộp giới hạn trùng lặp.
    boxes: mảng NumPy của các hộp giới hạn [[x1, y1, x2, y2], ...]
    scores: mảng NumPy của điểm tin cậy tương ứng
    iou_threshold: ngưỡng IoU để loại bỏ các hộp trùng lặp
    """
    if len(boxes) == 0:
        return []

    # Sắp xếp các hộp theo điểm tin cậy giảm dần
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_boxes = []
    
    while len(sorted_indices) > 0:
        # Lấy hộp có điểm tin cậy cao nhất
        best_box_idx = sorted_indices[0]
        keep_boxes.append(best_box_idx)
        
        # Loại bỏ hộp tốt nhất khỏi danh sách ứng viên
        sorted_indices = sorted_indices[1:]
        
        if len(sorted_indices) == 0:
            break

        # Tính toán IoU với tất cả các hộp còn lại
        ious = [iou(boxes[best_box_idx], boxes[idx]) for idx in sorted_indices]
        
        # Giữ lại các hộp có IoU dưới ngưỡng
        remaining_indices = np.where(np.array(ious) < iou_threshold)[0]
        sorted_indices = sorted_indices[remaining_indices]
        
    return [boxes[i] for i in keep_boxes], [scores[i] for i in keep_boxes]

def main():
    print("Mô phỏng One-Stage Object Detector")
    print("-" * 30)

    # Giả lập ảnh (ví dụ: kích thước 640x480)
    image_width, image_height = 640, 480

    # Giả lập các vật thể "ground truth" (thực tế) trên ảnh: [x1, y1, x2, y2, class_id, class_name]
    # Class ID: 0 = người, 1 = xe
    ground_truth_objects = [
        [50, 50, 150, 200, 0, "người"],
        [200, 100, 300, 250, 0, "người"],
        [400, 200, 550, 400, 1, "xe"]
    ]

    print("Vật thể thực tế (Ground Truth):")
    for obj in ground_truth_objects:
        print(f"  [{obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}] - {obj[5]}")
    print("-" * 30)

    # Giả lập các dự đoán "thô" từ một mô hình One-Stage (trước NMS)
    # Bao gồm các hộp trùng lặp và một số dự đoán có điểm thấp
    # Format: [x1, y1, x2, y2, score, class_id, class_name]
    raw_predictions = [
        # Dự đoán người 1 (tốt)
        [55, 55, 145, 195, 0.95, 0, "người"],
        # Dự đoán người 1 (hơi trùng)
        [60, 60, 160, 210, 0.88, 0, "người"],
        # Dự đoán người 2 (tốt)
        [205, 105, 295, 245, 0.92, 0, "người"],
        # Dự đoán người 2 (rất trùng, điểm thấp hơn)
        [200, 100, 300, 250, 0.70, 0, "người"],
        # Dự đoán xe (tốt)
        [405, 205, 545, 395, 0.90, 1, "xe"],
        # Dự đoán xe (điểm thấp, gần đúng nhưng có thể bị loại)
        [410, 210, 560, 410, 0.65, 1, "xe"],
        # Dự đoán sai, điểm thấp (có thể bị loại bởi ngưỡng điểm)
        [10, 10, 30, 30, 0.30, 0, "người"],
        # Dự đoán sai, điểm cao nhưng không có ground truth tương ứng (sẽ giữ lại nếu không có NMS)
        [500, 10, 550, 60, 0.80, 0, "người"] # "false positive"
    ]
    
    print("Dự đoán thô từ mô hình (trước NMS):")
    for pred in raw_predictions:
        print(f"  [{pred[0]}, {pred[1]}, {pred[2]}, {pred[3]}] - {pred[6]} (Score: {pred[4]:.2f})")
    print("-" * 30)

    # Tách các hộp và điểm số để truyền vào NMS
    boxes = np.array([p[:4] for p in raw_predictions])
    scores = np.array([p[4] for p in raw_predictions])
    class_ids = np.array([p[5] for p in raw_predictions])
    class_names = [p[6] for p in raw_predictions]

    # Ngưỡng điểm để lọc các dự đoán có điểm quá thấp
    score_threshold = 0.5
    high_confidence_indices = np.where(scores >= score_threshold)[0]

    filtered_boxes = boxes[high_confidence_indices]
    filtered_scores = scores[high_confidence_indices]
    filtered_class_ids = class_ids[high_confidence_indices]
    filtered_class_names = [class_names[i] for i in high_confidence_indices]

    print(f"Dự đoán sau khi lọc theo ngưỡng điểm ({score_threshold}):")
    if len(filtered_boxes) == 0:
        print("  Không có dự đoán nào.")
    else:
        for i in range(len(filtered_boxes)):
            print(f"  [{filtered_boxes[i][0]}, {filtered_boxes[i][1]}, {filtered_boxes[i][2]}, {filtered_boxes[i][3]}] - {filtered_class_names[i]} (Score: {filtered_scores[i]:.2f})")
    print("-" * 30)

    # Áp dụng Non-Maximum Suppression (NMS)
    iou_threshold = 0.5
    final_boxes, final_scores = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold)

    # Để có thể gán lại class_name cho các hộp đã qua NMS, chúng ta cần map lại.
    # Đây là một cách đơn giản để làm điều đó, trong thực tế sẽ phức tạp hơn.
    final_predictions_with_classes = []
    
    # Do cách NMS được thực hiện ở đây, chúng ta sẽ cần tìm lại class_id/name cho các final_boxes.
    # Trong một triển khai thực tế, NMS thường được thực hiện theo từng lớp hoặc lưu trữ thêm thông tin.
    # Ở đây, tôi sẽ đơn giản hóa bằng cách tìm lại class_name dựa trên hộp đã lọc ban đầu.
    for final_box, final_score in zip(final_boxes, final_scores):
        for i in range(len(filtered_boxes)):
            if np.array_equal(final_box, filtered_boxes[i]) and final_score == filtered_scores[i]:
                final_predictions_with_classes.append([
                    final_box[0], final_box[1], final_box[2], final_box[3],
                    final_score, filtered_class_ids[i], filtered_class_names[i]
                ])
                break

    print(f"Dự đoán cuối cùng sau Non-Maximum Suppression (IoU Threshold: {iou_threshold}):")
    if len(final_predictions_with_classes) == 0:
        print("  Không có dự đoán nào.")
    else:
        for pred in final_predictions_with_classes:
            print(f"  [{pred[0]}, {pred[1]}, {pred[2]}, {pred[3]}] - {pred[6]} (Score: {pred[4]:.2f})")
    print("-" * 30)

if __name__ == "__main__":
    main() 