import json

def continue_overtrain_detector(training_file_path):
    with open(training_file_path, "r") as f:
        data = json.load(f)

    loss_disc_history, smoothed_loss_disc_history = data.get("loss_disc_history", []), data.get("smoothed_loss_disc_history", [])
    loss_gen_history, smoothed_loss_gen_history = data.get("loss_gen_history", []), data.get("smoothed_loss_gen_history", [])

    return smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history

def check_overtraining(
    smoothed_loss_history, 
    threshold, 
    epsilon=0.004
):
    if len(smoothed_loss_history) < threshold + 1: return False

    for i in range(-threshold, -1):
        if smoothed_loss_history[i + 1] > smoothed_loss_history[i]: 
            return True

        if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon: 
            return False

    return True

def update_exponential_moving_average(
    smoothed_loss_history, 
    new_value, 
    smoothing=0.987
):
    smoothed_value = (
        new_value 
        if not smoothed_loss_history else 
        (smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value)
    )      

    smoothed_loss_history.append(smoothed_value)
    return smoothed_value

def save_to_json(
    file_path, 
    loss_disc_history, 
    smoothed_loss_disc_history, 
    loss_gen_history, 
    smoothed_loss_gen_history
):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({
            "loss_disc_history": loss_disc_history, 
            "smoothed_loss_disc_history": smoothed_loss_disc_history, 
            "loss_gen_history": loss_gen_history, 
            "smoothed_loss_gen_history": smoothed_loss_gen_history
        }, f)

def overtraining_detector(
    smoothed_loss_disc_history, 
    current_loss_disc, 
    smoothed_loss_gen_history, 
    current_loss_gen, 
    overtraining_threshold
):
    smoothed_value_disc = update_exponential_moving_average(
        smoothed_loss_disc_history, 
        current_loss_disc
    )

    smoothed_value_gen = update_exponential_moving_average(
        smoothed_loss_gen_history, 
        current_loss_gen
    )

    is_overtraining_disc = check_overtraining(
        smoothed_loss_disc_history, 
        overtraining_threshold * 2
    )

    is_overtraining_gen = check_overtraining(
        smoothed_loss_gen_history, 
        overtraining_threshold, 
        0.01
    )

    consecutive_increases_disc = (consecutive_increases_disc + 1) if is_overtraining_disc else 0
    consecutive_increases_gen = (consecutive_increases_gen + 1) if is_overtraining_gen else 0

    return smoothed_value_disc, smoothed_value_gen, is_overtraining_disc, is_overtraining_gen, consecutive_increases_disc, consecutive_increases_gen