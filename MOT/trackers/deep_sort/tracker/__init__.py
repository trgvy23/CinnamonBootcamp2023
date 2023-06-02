from .deep_sort import DeepSORT

__all__ = ['DeepSORT', 'build_tracker']


def build_tracker(weights, cfg, use_cuda=True):
    """
    Build a DeepSORT tracker instance.

    Args:
        weights (str): Path to the pre-trained model weights.
        cfg (dict): Configuration parameters for the tracker.
        use_cuda (bool): Whether to use CUDA for GPU acceleration.

    Returns:
        DeepSORT: An instance of the DeepSORT tracker.
    """
    return DeepSORT(weights,
                    max_dist=cfg['MAX_DIST'],
                    min_confidence=cfg['MIN_CONFIDENCE'],
                    nms_max_overlap=cfg['NMS_MAX_OVERLAP'],
                    max_iou_distance=cfg['MAX_IOU_DISTANCE'],
                    max_age=cfg['MAX_AGE'],
                    n_init=cfg['N_INIT'],
                    nn_budget=cfg['NN_BUDGET'],
                    use_cuda=use_cuda)
