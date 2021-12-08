import json
import modelop.utils as utils

logger = utils.configure_logger()


# modelop.init
def begin() -> None:
    """
    A function to set model-specific global variables used in ROI computations.
    """
    
    with open("modelop_parameters.json", "r") as parameters_file:
        modelop_parameters = json.load(parameters_file)
    
    ROI_parameters = modelop_parameters["monitoring"]["business_value"]["ROI"]
    logger.info("ROI parameters: %s", ROI_parameters)

    global amount_field, score_field
    global baseline_metrics, cost_multipliers
    global positive_class_label

    amount_field = ROI_parameters["amount_field"] # Column containing transaction amount
    score_field = ROI_parameters["score_field"] # Column containing model prediction
        
    # Classification metrics on baseline data
    baseline_metrics = ROI_parameters["baseline_metrics"]
    
    # ROI cost multipliers for each classification case
    cost_multipliers = ROI_parameters["cost_multipliers"]

    # Read and set label of positive class
    try:
        positive_class_label = modelop_parameters["monitoring"]["performance"]["positive_class_label"]
        logger.info("Label of Positive Class: %s", positive_class_label)
    except KeyError:
        raise KeyError("model parameters should define label of positive class!")


# modelop.metrics
def metrics(dataframe) -> dict:
    """
    A Function to compute projected ROI given a scored DataFrame.

    Args:
        dataframe (pd.DataFrame): Slice of Production data

    Yields:
        dict: Test Result containing projected roi metrics
    """

    # Compute projected ROI
    projected_roi = compute_projected_roi(dataframe)


    yield {
        "projected_roi": projected_roi,
        "amount_field": amount_field,
        "business_value": [
            {
                "test_name": "Projected ROI",
                "test_category": "business_value",
                "test_type": "projected_roi",
                "test_id": "business_value_projected_roi",
                "values": {
                    "projected_roi": projected_roi,
                    "amount_field": amount_field,
                    "baseline_metrics": baseline_metrics,
                    "cost_multipliers": cost_multipliers,
                },
            }
        ],
    }


def compute_projected_roi(data) -> float:
    """
    Helper function to compute projected ROI.

    Args:
        data (pd.DataFrame): Input DataFrame containing record_class

    Returns:
        float: projected ROI
    """
    
    projected_roi = 0
    for idx in range(len(data)):
        projected_roi += data.iloc[idx][amount_field] * (
            (data.iloc[idx][score_field] == positive_class_label)
            * (
                baseline_metrics["TPR"] * cost_multipliers["TP"]
                + (1 - baseline_metrics["TPR"] * cost_multipliers["FP"])
            )
            + (data.iloc[idx][score_field] != positive_class_label)
            * (
                baseline_metrics["TNR"] * cost_multipliers["TN"]
                + (1 - baseline_metrics["TNR"] * cost_multipliers["FN"])
            )
        )

    return round(projected_roi, 2)
