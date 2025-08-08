import torch
import torch.nn as nn
import torch.nn.functional as F
import geometry


def generate_grid(
    num_points,
    batch_size=1,
    image_height=None,
    image_width=None,
    framestack=None,
    device="cuda",
):
    """
    Generate random matching points within image dimensions.

    Args:
        num_points (int): Number of points to generate
        batch_size (int): Batch size for generated points
        image_height (int, optional): Height of the image. Inferred from framestack if not provided
        image_width (int, optional): Width of the image. Inferred from framestack if not provided
        framestack (torch.Tensor, optional): Input framestack of shape [B, T, C, H, W]
        device (str): Device to place tensors on

    Returns:
        torch.Tensor: Random points of shape [batch_size, num_points, 2]
    """
    # Infer dimensions from framestack if provided
    if framestack is not None:
        image_height = framestack.shape[-2]
        image_width = framestack.shape[-1]
        batch_size = framestack.shape[0]

    # Validate inputs
    assert (
        image_height is not None and image_width is not None
    ), "Must provide either framestack or image dimensions"

    # Generate random coordinates
    x_coords = torch.rand(batch_size, num_points) * (image_width - 1)
    y_coords = torch.rand(batch_size, num_points) * (image_height - 1)

    # Stack coordinates and move to device
    points = torch.stack([x_coords, y_coords], dim=-1).to(device)

    return points


def mine_triplets_optimized(
    sourceembs,
    targetembs,
    margin=0.0,
    nth_candidate=1,
    target_mask=None,
    sourceembs_idx=None,
    targetembs_idx=None,
    are_matched=True,
):
    """
    Vectorized version of the 'triplets' function, returning the same
    triplets (possibly in a different order) but without explicit for-loops
    over batch or anchor indices.
    """

    B, C, S = sourceembs.shape

    # Normalize along the channel dimension
    src_norm = F.normalize(sourceembs, p=2, dim=1)  # (B, C, S)
    tgt_norm = F.normalize(targetembs, p=2, dim=1)  # (B, C, S)

    # Cosine similarities for src->tgt, then convert to L2 distances
    # shape (B, S, S)
    sim_src2tgt = torch.matmul(src_norm.transpose(1, 2), tgt_norm)  # (B, S, S)
    dist_src2tgt = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_src2tgt, min=0.0))

    # Cosine similarities for tgt->tgt, then convert to L2 distances
    # shape (B, S, S)
    sim_tgt2tgt = torch.matmul(tgt_norm.transpose(1, 2), tgt_norm)  # (B, S, S)
    dist_tgt2tgt = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_tgt2tgt, min=0.0))

    # We'll store the cosines in a single tensor to mimic the original sim_matrix
    # => same shape (B, S, S) as "simmatrix_tensor"
    # The old code returned just the src->tgt similarity
    sim_matrix = sim_src2tgt

    # target_mask => shape (B, 1, S) originally, we used mask_reduced = target_mask[:, 0, :].
    # So let’s replicate that logic:
    if target_mask is not None:
        mask_reduced = target_mask[:, 0, :].int()  # (B, S)
    else:
        mask_reduced = None

    # If no mask is given, everything is valid => set all True
    if mask_reduced is None:
        valid_targets = torch.ones((B, S), dtype=torch.bool, device=sourceembs.device)
    else:
        valid_targets = mask_reduced.bool()  # (B, S)

    # ---------- Determine positives ----------

    # We'll define a "pos_idx" for every anchor (B, S).
    # If are_matched=True, pos_idx[i] = i.  If false, pick the nth patch with distance < margin.
    anchor_grid = torch.arange(S, device=sourceembs.device).unsqueeze(0).expand(B, S)

    if are_matched:
        # shape (B, S)
        pos_idx = anchor_grid
        # We also skip anchors if the target at pos_idx is not valid
        # => i.e. skip if valid_targets[b, i] = false
        valid_pos_mask = valid_targets.gather(dim=1, index=pos_idx)
    else:
        # For each anchor row: pick the nth patch whose distance < margin
        # dist_src2tgt is (B, S, S). We want dist_src2tgt[b, i, :].
        # We'll mask with valid_targets[b, :]
        # Then pick the nth smallest distance among those < margin.
        # We'll do this by forcing invalid entries to +inf, then .topk(nth_candidate, largest=False).
        dist_for_pos = dist_src2tgt  # shape (B, S, S)
        # condition: distance < margin AND valid
        pos_condition = (dist_for_pos < margin).logical_and(
            valid_targets.unsqueeze(1)  # shape (B,1,S)
        )

        # replace invalid with +inf
        dist_for_pos_valid = torch.where(
            pos_condition, dist_for_pos, torch.full_like(dist_for_pos, float("inf"))
        )
        # topk along dim=2 to get the nth smallest
        # values, indices => shape (B, S, nth_candidate)
        values, indices = dist_for_pos_valid.topk(k=nth_candidate, dim=2, largest=False)
        # The nth candidate (if it exists) is:
        nth_vals = values[:, :, nth_candidate - 1]  # (B, S)
        pos_idx = indices[:, :, nth_candidate - 1]  # (B, S)
        # But if nth_vals is inf => no valid candidate => skip anchor
        valid_pos_mask = torch.isfinite(nth_vals)

    # ---------- Option A (anchor->target) for negatives ----------
    # For each anchor i: pick the nth patch j with dist_src2tgt[b, i, j] > margin AND valid.
    # Also skip j if j == pos_idx (since the old code discards the negative if it matches the positive).
    dist_for_negA = dist_src2tgt  # (B, S, S)
    negA_condition = (dist_for_negA > margin).logical_and(
        valid_targets.unsqueeze(1)  # shape (B,1,S)
    )

    # We also skip j == pos_idx => that’s a fancy per‐element condition:
    # shape (B, S, S) vs (B, S) => we do:
    # anchor_grid for j => shape (1,1,S) broadcast to (B,S,S)
    j_grid = torch.arange(S, device=sourceembs.device).view(1, 1, S)
    # compare j_grid != pos_idx => but pos_idx is (B,S). We can expand that to (B,S,1).
    pos_idx_3d = pos_idx.unsqueeze(-1)  # (B, S, 1)
    mask_j_not_pos = j_grid.expand(B, S, S) != pos_idx_3d

    negA_condition = negA_condition & mask_j_not_pos

    dist_for_negA_valid = torch.where(
        negA_condition, dist_for_negA, torch.full_like(dist_for_negA, float("inf"))
    )
    # shape (B, S, nth_candidate), after topk
    valsA, idxA = dist_for_negA_valid.topk(k=nth_candidate, dim=2, largest=False)
    # The chosen negative index for Option A
    negA_idx = idxA[:, :, nth_candidate - 1]  # (B, S)
    negA_dist = valsA[:, :, nth_candidate - 1]  # (B, S)
    valid_negA_mask = torch.isfinite(negA_dist)

    # ---------- Option B (positive->target) for negatives ----------
    # Need the distance from the chosen positive patch to all target patches.
    # That is dist_tgt2tgt[b, pos_idx, j].
    # We can gather row = pos_idx for each (b, anchor).
    # dist_tgt2tgt is (B, S, S), we want dist_tgt2tgt[b, pos_idx[b,i], j].
    # We'll do a 2D gather to get shape (B, S, S) again with "row = pos_idx[b,i]" for each anchor i.
    row_indices = pos_idx.clamp(min=0, max=S - 1)  # just to be safe
    # build a mesh of (b, i, j)
    b_grid = torch.arange(B, device=sourceembs.device).view(B, 1, 1).expand(B, S, S)
    i_grid = torch.arange(S, device=sourceembs.device).view(1, S, 1).expand(B, S, S)
    j_grid = torch.arange(S, device=sourceembs.device).view(1, 1, S).expand(B, S, S)
    # gather row = row_indices[b,i]
    # so for each (b,i), row = row_indices[b,i]. Then we vary j
    row_gather = row_indices.unsqueeze(-1).expand(B, S, S)  # shape (B,S,S)
    # index into dist_tgt2tgt with [b, row, j]
    dist_for_negB = dist_tgt2tgt[b_grid, row_gather, j_grid]  # (B, S, S)

    # Condition: dist_for_negB > margin AND valid AND j != pos_idx
    negB_condition = (dist_for_negB > margin).logical_and(valid_targets.unsqueeze(1))
    mask_j_not_posB = j_grid != row_gather  # row_gather is pos_idx
    negB_condition = negB_condition & mask_j_not_posB

    dist_for_negB_valid = torch.where(
        negB_condition, dist_for_negB, torch.full_like(dist_for_negB, float("inf"))
    )
    valsB, idxB = dist_for_negB_valid.topk(k=nth_candidate, dim=2, largest=False)
    negB_idx = idxB[:, :, nth_candidate - 1]  # (B, S)
    negB_dist = valsB[:, :, nth_candidate - 1]  # (B, S)
    valid_negB_mask = torch.isfinite(negB_dist)

    # ---------- Combine Option A and Option B ----------
    # We want the negative with the smaller distance if both are valid, else the valid one, else skip
    both_valid_mask = valid_negA_mask & valid_negB_mask
    onlyA_valid_mask = valid_negA_mask & ~valid_negB_mask
    onlyB_valid_mask = ~valid_negA_mask & valid_negB_mask

    # pick whichever is smaller where both are valid
    pickA_mask = both_valid_mask & (negA_dist <= negB_dist)
    pickB_mask = both_valid_mask & (negB_dist < negA_dist)

    final_neg_idx = torch.where(
        pickA_mask,
        negA_idx,
        torch.where(
            pickB_mask,
            negB_idx,
            torch.where(
                onlyA_valid_mask,
                negA_idx,
                torch.where(onlyB_valid_mask, negB_idx, torch.full_like(negA_idx, -1)),
            ),
        ),
    )

    final_neg_dist = torch.where(
        pickA_mask,
        negA_dist,
        torch.where(
            pickB_mask,
            negB_dist,
            torch.where(
                onlyA_valid_mask,
                negA_dist,
                torch.where(
                    onlyB_valid_mask,
                    negB_dist,
                    torch.full_like(negA_dist, float("inf")),
                ),
            ),
        ),
    )

    # We skip any anchor whose positive is invalid or whose final_neg_idx == -1, or final_neg_dist = inf
    valid_trip_mask = (
        valid_pos_mask & (final_neg_idx >= 0) & torch.isfinite(final_neg_dist)
    )

    # Flatten (b, anchor) to a single dimension of valid triplets
    # We’ll gather the anchor b,i => pos => neg for each valid slot
    # Build a linear index from (b, i)
    lin_bi = torch.arange(B * S, device=sourceembs.device)
    b_idx = lin_bi // S
    i_idx = lin_bi % S

    # We flatten valid_trip_mask => shape (B*S,)
    valid_trip_mask_flat = valid_trip_mask.view(-1)
    # Gather final_neg_idx => shape (B*S,) with the same flatten
    final_neg_idx_flat = final_neg_idx.view(-1)
    pos_idx_flat = pos_idx.reshape(-1)

    # Filter only valid ones
    valid_lin_bi = lin_bi[valid_trip_mask_flat]
    valid_b = b_idx[valid_trip_mask_flat]
    valid_i = i_idx[valid_trip_mask_flat]
    valid_pos = pos_idx_flat[valid_trip_mask_flat]
    valid_neg = final_neg_idx_flat[valid_trip_mask_flat]
    # Distances
    a2p_dist = dist_src2tgt.view(-1, S)[valid_lin_bi, valid_pos]  # shape (#triplets,)
    # final_neg_dist was shaped (B, S) => flatten => shape(B*S,)
    a2n_dist = final_neg_dist.view(-1)[valid_trip_mask_flat]

    # Gather embeddings from src (anchor) => shape (#triplets, C)
    # to gather: sourceembs[b, :, i]
    anchor_emb = sourceembs[valid_b, :, valid_i]
    # positive => targetembs[b, :, pos_idx]
    positive_emb = targetembs[valid_b, :, valid_pos]
    # negative => targetembs[b, :, valid_neg]
    negative_emb = targetembs[valid_b, :, valid_neg]

    # If we have index arrays to map
    if sourceembs_idx is not None:
        anchor_id = sourceembs_idx[valid_b, valid_i]
    else:
        anchor_id = valid_i
    if targetembs_idx is not None:
        positive_id = targetembs_idx[valid_b, valid_pos]
        negative_id = targetembs_idx[valid_b, valid_neg]
    else:
        positive_id = valid_pos
        negative_id = valid_neg

    # Build output
    out = {
        "anchor": anchor_emb,
        "positive": positive_emb,
        "negative": negative_emb,
        "a2p": a2p_dist,
        "a2n": a2n_dist,
        "sim_matrix": sim_matrix,  # shape (B, S, S), same as original
        "anchor_indices": anchor_id,
        "positive_indices": positive_id,
        "negative_indices": negative_id,
        "batch_indices": valid_b,  # which batch each triplet came from
    }
    return out
