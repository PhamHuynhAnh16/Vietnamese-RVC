import json

def continue_overtrain_detector(training_file_path):
    """
    Loads historical raw and smoothed loss metrics for both the generator 
    and discriminator from a JSON monitoring log file.

    Args:
        training_file_path (str): Path to the target JSON log file.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            - smoothed_loss_gen_history: Historical smoothed generator losses.
            - loss_gen_history: Historical raw generator losses.
            - loss_disc_history: Historical raw discriminator losses.
            - smoothed_loss_disc_history: Historical smoothed discriminator losses.
    """

    with open(training_file_path, "r") as f:
        data = json.load(f)

    # Safely extract historical records defaulting to an empty list if not found
    loss_disc_history, smoothed_loss_disc_history = data.get("loss_disc_history", []), data.get("smoothed_loss_disc_history", [])
    loss_gen_history, smoothed_loss_gen_history = data.get("loss_gen_history", []), data.get("smoothed_loss_gen_history", [])

    return smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history

def check_overtraining(
    smoothed_loss_history, 
    threshold, 
    epsilon=0.004
):
    """
    Analyzes recent trends in smoothed loss history to flag potential overtraining.
    Overtraining is identified if the error consistently rises or fails to decline 
    by a significant margin (epsilon) across a given step window (threshold).

    Args:
        smoothed_loss_history (List[float]): Cumulative list of smoothed loss values.
        threshold (int): The number of recent trailing elements to verify.
        epsilon (float, optional): Minimal threshold required to signify 
            legitimate descent. Defaults to 0.004.

    Returns:
        bool: True if overtraining characteristics are detected, False otherwise.
    """

    # Not enough data points gathered yet to safely evaluate the lookback window
    if len(smoothed_loss_history) < threshold + 1: return False

    # Loop backwards through the historical window slice
    for i in range(-threshold, -1):
        # Flag immediately if a newer value is strictly higher than the previous one
        if smoothed_loss_history[i + 1] > smoothed_loss_history[i]: 
            return True

        # If the loss successfully drops by a value >= epsilon, it's a good step
        if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon: 
            return False

    # True represents a stagnant state where fluctuations remained under epsilon bounds
    return True

def update_exponential_moving_average(
    smoothed_loss_history, 
    new_value, 
    smoothing=0.987
):
    """
    Applies and appends an Exponential Moving Average (EMA) smoothing step onto 
    a tracking history array list.

    Args:
        smoothed_loss_history (List[float]): Cumulative list of historical smoothed losses.
        new_value (float): Incoming raw scalar value to smooth.
        smoothing (float, optional): Weight coefficient factor for past data. Defaults to 0.987.

    Returns:
        float: The newly updated EMA smoothed value.
    """

    # Initialize using the raw value directly if this is the first item in the list
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
    """
    Saves the training history arrays into an external JSON log file.

    Args:
        file_path (str): Destination path where the JSON file will be written.
        loss_disc_history (List[float]): Raw discriminator losses.
        smoothed_loss_disc_history (List[float]): Smoothed discriminator losses.
        loss_gen_history (List[float]): Raw generator losses.
        smoothed_loss_gen_history (List[float]): Smoothed generator losses.
    """

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
    overtraining_threshold,
    consecutive_increases_disc,
    consecutive_increases_gen
):
    """
    Orchestrates the overtraining check routines for an active optimization step. 
    Smooths new metrics, executes trend checks, and updates anomaly counters.

    Args:
        smoothed_loss_disc_history (List[float]): Historical smoothed discriminator loss list.
        current_loss_disc (float): Current step's raw discriminator loss.
        smoothed_loss_gen_history (List[float]): Historical smoothed generator loss list.
        current_loss_gen (float): Current step's raw generator loss.
        overtraining_threshold (int): Base step lookback threshold length window.
        consecutive_increases_disc (int): Current counter tracking consecutive overtraining anomalies in Discriminator.
        consecutive_increases_gen (int): Current counter tracking consecutive overtraining anomalies in Generator.

    Returns:
        Tuple[float, float, bool, bool, int, int]:
            - smoothed_value_disc: New EMA smoothed discriminator loss scalar.
            - smoothed_value_gen: New EMA smoothed generator loss scalar.
            - is_overtraining_disc: Flag checking if discriminator is overtraining.
            - is_overtraining_gen: Flag checking if generator is overtraining.
            - consecutive_increases_disc: Updated consecutive counter tracking for discriminator.
            - consecutive_increases_gen: Updated consecutive counter tracking for generator.
    """

    # 1. Update Exponential Moving Averages
    smoothed_value_disc = update_exponential_moving_average(
        smoothed_loss_disc_history, 
        current_loss_disc
    )

    smoothed_value_gen = update_exponential_moving_average(
        smoothed_loss_gen_history, 
        current_loss_gen
    )

    # 2. Check for overtraining signatures over specified lookback windows
    is_overtraining_disc = check_overtraining(
        smoothed_loss_disc_history, 
        overtraining_threshold * 2
    )

    is_overtraining_gen = check_overtraining(
        smoothed_loss_gen_history, 
        overtraining_threshold, 
        0.01
    )

    # 3. Update or reset the consecutive anomaly counters based on current step evaluation flags
    consecutive_increases_disc = (consecutive_increases_disc + 1) if is_overtraining_disc else 0
    consecutive_increases_gen = (consecutive_increases_gen + 1) if is_overtraining_gen else 0

    return smoothed_value_disc, smoothed_value_gen, is_overtraining_disc, is_overtraining_gen, consecutive_increases_disc, consecutive_increases_gen