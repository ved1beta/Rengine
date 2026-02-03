import torch
import random
import numpy as np
from pathlib import Path


def save_checkpoint(path,model,optimizer,update):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    torch.save(checkpoint, path)

def load_checkpoint(
    path,
    model,
    optimizer,
    device
):
    checkpoint = torch.load(path, weights_only=False ,map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
    if checkpoint["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng"]["cuda"]])

    np.random.set_state(checkpoint["rng"]["numpy"])
    random.setstate(checkpoint["rng"]["python"])

    return checkpoint["update"] + 1

