# Projected ROI Monitor
This ModelOp Center monitor computes **Return-on-Investment projections** given a slice of scored production data (containing model outputs) .

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Sample Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored is a **classification** model.
 - `BUSINESS_MODEL` has 
     - a **modelop parameters** file (`.json`) defining monitoring parameters, such as
     ```json
     {
         "monitoring": {
             "performance": {
                 "model_type": "classification",
                 "positive_class_label": <label_of_positive_class>
             },
             "business_value": {
                 "ROI": {
                     "amount_field": <name_of_amount_feature>,
                     "label_field": <name_of_ground_truth_feature>,
                     "score_field": <name_of_model_output>,
                     "baseline_metrics": {
                         "TNR": <true_negative_rate>,
                         "TPR": <true_positive_rate>
                     },
                     "cost_multipliers": {
                         "TP": <true_positive_cost_multiplier>,
                         "FP": <false_positive_cost_multiplier>,
                         "TN": <true_negative_cost_multiplier>,
                         "FN": <false_negative_cost_multiplier>
                     }
                 }
             }
         }
     }
     ```

## Execution
1. `init` function loads the `modelop_parameters.json` file and sets global monitoring variables.
2. `metrics` function computes projected ROI by calling the `compute_projected_roi` function.
3. Test results are appended to the list of `business_value` tests to be returned by the model.

## Monitor Output

```JSON
{
    "projected_roi": <projected_ROI_amount>,
    "amount_field": <name_of_amount_feature>,
    "business_value": [
        {
            "test_name": "Projected ROI",
            "test_category": "business_value",
            "test_type": "projected_roi",
            "test_id": "business_value_projected_roi",
            "values": {
                "projected_roi": <projected_ROI_amount>,
                "amount_field": <name_of_amount_feature>,
                "baseline_metrics": {
                    "TNR": <true_negative_rate>,
                    "TPR": <true_positive_rate>
                },
                "cost_multipliers": {
                    "TP": <true_positive_cost_multiplier>,
                    "FP": <false_positive_cost_multiplier>,
                    "TN": <true_negative_cost_multiplier>,
                    "FN": <false_negative_cost_multiplier>
                }
            }
        }
    ]
}
```