import torch
import logging
# from .schedule_model import NeuralUCB0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, model_path: str):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # self.model = NeuralUCB0(input_dim=20)
            logger.info("Scheduler has been successfully initialized")
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {str(e)}", exc_info=True)
            self.model = None

    def infer(self, size, user, queue, latency) -> int:
        raise NotImplementedError("Scheduler has not been implemented yet.")
        assert len(size) == 2
        context = [size[0], size[1]]
        # context.extend([np.mean(roi_areas), np.std(roi_areas), np.max(roi_areas), np.min(roi_areas)])
        context.extend(user)
        context.extend(queue)
        context.extend(latency)
        return 0  # self.model.select_action(context)