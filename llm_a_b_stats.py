import math
import numpy as np


def analyze_ab_test(
    control_outcomes: list,
    treatment_outcomes: list,
    confidence_level: float = 0.95,
    min_detectable_effect: float = 0.02
) -> dict:

    if not control_outcomes or not treatment_outcomes:
        return {}

    control_array = np.asarray(control_outcomes, dtype=float)
    treatment_array = np.asarray(treatment_outcomes, dtype=float)

    control_sample_size = len(control_array)
    treatment_sample_size = len(treatment_array)

    control_success_count = control_array.sum()
    treatment_success_count = treatment_array.sum()

    control_rate = control_success_count / control_sample_size
    treatment_rate = treatment_success_count / treatment_sample_size

    absolute_lift = treatment_rate - control_rate
    relative_lift_pct = (absolute_lift / control_rate * 100) if control_rate != 0 else np.inf

    significance_level = 1 - confidence_level
    z_critical_lookup = {
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758
    }
    critical_z_value = z_critical_lookup.get(round(confidence_level, 2), 1.9600)
    pooled_success_rate = (
        (control_success_count + treatment_success_count) /
        (control_sample_size + treatment_sample_size)
    )

    pooled_standard_error = np.sqrt(
        pooled_success_rate * (1 - pooled_success_rate) *
        (1 / control_sample_size + 1 / treatment_sample_size)
    )

    if pooled_standard_error == 0:
        z_statistic = 0.0 if absolute_lift == 0 else np.sign(absolute_lift) * np.inf
        p_value = 1.0 if absolute_lift == 0 else 0.0
    else:
        z_statistic = absolute_lift / pooled_standard_error
        normal_cumulative_probability = 0.5 * (
            1 + math.erf(abs(z_statistic) / np.sqrt(2))
        )
        p_value = 2 * (1 - normal_cumulative_probability)

    unpooled_standard_error = np.sqrt(
        (control_rate * (1 - control_rate) / control_sample_size) +
        (treatment_rate * (1 - treatment_rate) / treatment_sample_size)
    )

    confidence_interval_lower_bound = absolute_lift - critical_z_value * unpooled_standard_error
    confidence_interval_upper_bound = absolute_lift + critical_z_value * unpooled_standard_error

    statistically_significant = p_value < significance_level
    practically_significant = absolute_lift >= min_detectable_effect

    z_value_for_eighty_percent_power = 0.8416
    effect_size = abs(min_detectable_effect)

    if effect_size == 0:
        required_sample_size_per_group = 0
    else:
        treatment_rate_for_sample_size = min(max(control_rate + effect_size, 0.0), 1.0)
        average_rate_for_sample_size = (control_rate + treatment_rate_for_sample_size) / 2

        required_sample_size = (
            (
                critical_z_value * np.sqrt(
                    2 * average_rate_for_sample_size * (1 - average_rate_for_sample_size)
                ) +
                z_value_for_eighty_percent_power * np.sqrt(
                    control_rate * (1 - control_rate) +
                    treatment_rate_for_sample_size * (1 - treatment_rate_for_sample_size)
                )
            ) ** 2
        ) / (effect_size ** 2)
        required_sample_size_per_group = int(np.ceil(required_sample_size))

    current_sample_size_sufficient = (
        control_sample_size >= required_sample_size_per_group and
        treatment_sample_size >= required_sample_size_per_group
    )

    if not current_sample_size_sufficient or not statistically_significant:
        recommendation = "continue_testing"
    elif absolute_lift > 0 and practically_significant:
        recommendation = "launch_treatment"
    else:
        recommendation = "keep_control"

    return {
        "control_rate": round(float(control_rate), 4),
        "treatment_rate": round(float(treatment_rate), 4),
        "absolute_lift": round(float(absolute_lift), 4),
        "relative_lift_pct": round(float(relative_lift_pct), 4),
        "z_statistic": round(float(z_statistic), 4) if np.isfinite(z_statistic) else float(z_statistic),
        "p_value": round(float(p_value), 4),
        "confidence_interval": (
            round(float(confidence_interval_lower_bound), 4),
            round(float(confidence_interval_upper_bound), 4)
        ),
        "statistically_significant": statistically_significant,
        "practically_significant": practically_significant,
        "required_sample_size_per_group": required_sample_size_per_group,
        "current_sample_size_sufficient": current_sample_size_sufficient,
        "recommendation": recommendation
    }


if __name__ == "__main__":
 
    np.random.seed(42)
    control = list(np.random.binomial(1, 0.10, size=1000))    
    treatment = list(np.random.binomial(1, 0.12, size=1000))  

    result = analyze_ab_test(control, treatment, confidence_level=0.95, min_detectable_effect=0.02)

    print("A/B Test Results:")
    print("-" * 45)
    for key, value in result.items():
        print(f"  {key}: {value}")
