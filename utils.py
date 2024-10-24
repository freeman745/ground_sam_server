def filter_large_small_bboxes(bboxes, labels, max_ratio, min_ratio, imgw, imgh):
    filtered_bboxes = []
    filtered_labels = []

    for b in range(len(bboxes)):
        i = bboxes[b]
        w = i[2] - i[0]
        h = i[3] - i[1]
        if w > max_ratio * imgw or h > max_ratio * imgh or w < min_ratio * imgw or h < min_ratio * imgh:
            continue
        filtered_bboxes.append(i)
        filtered_labels.append(labels[b])

    return filtered_bboxes, filtered_labels


def calculate_intersection_area(bbox1, bbox2):
    # get overlap
    x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

    # calculate area
    intersection_area = x_overlap * y_overlap
    return intersection_area


def filter_bboxes_by_overlap(bboxes, labels):
    filtered_bboxes = []
    filtered_labels = []

    for i, bbox1 in enumerate(bboxes):
        is_contained = False

        for j, bbox2 in enumerate(bboxes):
            if i == j:
                continue

            # calculate area
            intersection_area = calculate_intersection_area(bbox1, bbox2)

            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

            score1 = bbox1[-1]
            score2 = bbox2[-1]

            # get smaller one
            smaller_area = min(area1,area2)

            # calculate ratio
            overlap_ratio = intersection_area / smaller_area

            # >0.8
            if overlap_ratio > 0.8 and score1 < score2:
                is_contained = True
                break

        if not is_contained:
            filtered_bboxes.append(bbox1)
            filtered_labels.append(labels[i])

    return filtered_bboxes, filtered_labels