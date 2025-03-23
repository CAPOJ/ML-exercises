import torch


def evaluate_map(batch, results):
    device = 'cuda'
    targets, preds = [], []
    for target_masks, class_labels, seg_info in zip(
    batch["mask_labels"], batch["class_labels"], results
    ):
        target_masks = target_masks.to(torch.uint8)

        if target_masks.shape[0] == 0:
            target_labels = torch.tensor([], device=device)
        else:
            target_labels = class_labels

        pred_labels = torch.tensor([x["label_id"] for x in seg_info["segments_info"]], device=device)

        if pred_labels.shape == torch.Size([0]):
            continue

        pred_scores = torch.tensor([x["score"] for x in seg_info["segments_info"]], device=device)
        pred_masks = seg_info["segmentation"].to(torch.uint8)

        preds.append({"masks": pred_masks, "scores": pred_scores, "labels": pred_labels})
        targets.append({"masks": target_masks, "labels": target_labels})
    return preds, targets