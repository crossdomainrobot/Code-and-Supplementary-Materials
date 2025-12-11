import os
import re
import numpy as np
import sys
from openai import OpenAI
import TextGrad as tg
from TextGrad.tasks import load_task

LOG_DIR = r"D:\Aresearch\闆溅璁烘枃\闆溅璁烘枃2\鏁版嵁\OPRO_distributed\22_comparision_SDS_textgrad"
os.makedirs(LOG_DIR, exist_ok=True)

STAGE_STEP_CONFIG = {
    "early": {
        "min_abs_step": 0.0,
        "max_abs_step": 4.0,
    },
    "mid": {
        "min_abs_step": 0.001,
        "max_abs_step": 1.0,
    },
    "late": {
        "min_abs_step": 0.001,
        "max_abs_step": 0.1,
    },
}



class LanguageGradientEngine:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def __call__(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
        )
        return resp.choices[0].message.content


def build_textgrad_task():
    try:
        return load_task("numeric-optimization")
    except Exception:
        return None



def compute_segment_distances(x_coord, y_coord, segment_points):
    n_segments = len(segment_points)
    distances = np.zeros(n_segments)
    for i in range(n_segments):
        start_idx = np.sum(segment_points[:i]) if i > 0 else 0
        end_idx = np.sum(segment_points[:i+1])
        dx = x_coord[end_idx - 1] - x_coord[start_idx]
        dy = y_coord[end_idx - 1] - y_coord[start_idx]
        distances[i] = np.sqrt(dx**2 + dy**2)
    return distances


def extract_first_float(text: str) -> float:
    pattern = r'-?\d+(?:\.\d+)?'
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"No float found in LLM output: {text}")
    return float(m.group(0))


def extract_stage_label(text: str) -> str:
    low = text.lower()
    if "late" in low:
        return "late"
    if "mid" in low or "middle" in low:
        return "mid"
    return "early"



def build_z_from_height_differences(x, y, segment_points, height_differences, H_target):
    n = len(x)
    z = np.zeros_like(x)

    cumulative_height = H_target - np.cumsum(
        np.concatenate(([0.0], height_differences.astype(float)))
    )

    start_idx = 0
    for i, seg_len in enumerate(segment_points):
        end_idx = start_idx + seg_len
        z[start_idx:end_idx] = np.linspace(
            cumulative_height[i],
            cumulative_height[i+1],
            seg_len
        )
        start_idx = end_idx

    return z


def compute_full_cost(
    x, y, segment_points, height_differences,
    L_target=1270,
    H_target=140,
    max_slope=0.204,
    mean_slope=0.116,
):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    height_differences = np.asarray(height_differences, dtype=float)
    n = len(x)

    z = build_z_from_height_differences(x, y, segment_points,
                                        height_differences, H_target)

    dx1 = np.zeros(n); dy1 = np.zeros(n); dz = np.zeros(n)
    ddx = np.zeros(n); ddy = np.zeros(n); ddz = np.zeros(n)
    curvature_2d = np.zeros(n); curvature_3d = np.zeros(n)
    curvature_difference = np.zeros(n)

    for i in range(1, n - 1):
        dx1[i] = (x[i+1] - x[i-1]) / 2
        dy1[i] = (y[i+1] - y[i-1]) / 2
        dz[i]  = (z[i+1] - z[i-1]) / 2
        ddx[i] = x[i+1] - 2 * x[i] + x[i-1]
        ddy[i] = y[i+1] - 2 * y[i] + y[i-1]
        ddz[i] = z[i+1] - 2 * z[i] + z[i-1]

    dx1[0],  dy1[0],  dz[0]  = x[1]-x[0],   y[1]-y[0],   z[1]-z[0]
    dx1[-1], dy1[-1], dz[-1] = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
    ddx[0],  ddy[0],  ddz[0] = dx1[1]-dx1[0], dy1[1]-dy1[0], dz[0]-dz[1]
    ddx[-1], ddy[-1], ddz[-1] = dx1[-1]-dx1[-2], dy1[-1]-dy1[-2], dz[-1]-dz[-2]

    for i in range(n):
        dx1_sq, dy1_sq, dz1_sq = dx1[i]**2, dy1[i]**2, dz[i]**2
        ddx_sq, ddy_sq, ddz_sq = ddx[i]**2, ddy[i]**2, ddz[i]**2

        norm_d  = dx1_sq + dy1_sq + dz1_sq
        norm_dd = ddx_sq + ddy_sq + ddz_sq
        dot_dd  = dx1[i]*ddx[i] + dy1[i]*ddy[i] + dz[i]*ddz[i]

        denom3d = (np.sqrt(norm_d)**3 + 1e-8)
        num3d   = norm_dd * norm_d - dot_dd**2
        if not np.isfinite(num3d) or num3d < 0:
            num3d = 0.0

        curvature_2d[i] = abs(dx1[i]*ddy[i] - ddx[i]*dy1[i]) / ((dx1_sq + dy1_sq)**1.5 + 1e-8)
        curvature_3d[i] = np.sqrt(num3d) / denom3d
        curvature_difference[i] = curvature_2d[i] - curvature_3d[i]

    curvature_difference_sum = np.sum(curvature_difference)

    distances1 = np.zeros(n)
    slope1 = np.zeros(n)
    for i in range(1, n):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dist = np.sqrt(dx**2 + dy**2)
        distances1[i] = distances1[i-1] + dist
        slope1[i] = (z[i-1] - z[i]) / (dist + 1e-8)
    slope_mean = np.sum(slope1) / n

    segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    total_length = np.sum(segment_lengths)
    current_height = np.sum(height_differences)

    def huber(r, delta):
        ar = np.abs(r)
        return np.where(ar <= delta, 0.5 * r**2, delta * (ar - 0.5 * delta))

    tau_L = 20.0
    L = total_length
    C_L_hard = (np.maximum(0.0, L - 1300.0) ** 2) / (1300.0 ** 2)
    C_L_soft = huber(L - 1270.0, tau_L) / (1270.0 ** 2)
    wL_hard, wL_soft = 1.0, 0.2
    C_L = wL_hard * C_L_hard + wL_soft * C_L_soft

    H = current_height
    C_H_hard = (np.maximum(0.0, 120.0 - H) ** 2) / (120.0 ** 2) \
             + (np.maximum(0.0, H - 150.0) ** 2) / (150.0 ** 2)
    band_dev = np.maximum(0.0, np.maximum(130.0 - H, H - 140.0))
    C_H_band = (band_dev ** 2) / (10.0 ** 2)
    wH_hard, wH_band = 1.0, 0.4
    C_H = wH_hard * C_H_hard + wH_band * C_H_band

    C_C = curvature_difference_sum / n

    smax = max_slope
    excess = np.maximum(0.0, np.abs(slope1) - smax)
    C_S = np.mean(excess ** 2) / (smax ** 2)

    C_S_S = slope_mean - mean_slope

    cost = 10000 * (
        0.25 * C_L
        + 0.35 * C_H
        + 0.04 * np.abs(C_C)
        + 0.35 * np.abs(C_S)
        + 0.22 * np.abs(C_S_S)
    )

    return float(cost), dict(
        total_length=total_length,
        current_height=current_height,
        C_L=C_L,
        C_H=C_H,
        C_C=C_C,
        C_S=C_S,
        C_S_S=C_S_S,
        slope_mean=slope_mean,
        curvature_difference_sum=curvature_difference_sum,
        C_L_hard=C_L_hard,
        C_L_soft=C_L_soft,
        C_H_hard=C_H_hard,
        C_H_band=C_H_band,
    )



def build_system_prompt():
    system_prompt = (
        " (Background) In bobsleigh races, athletes must steer the sled along a predefined track as quickly as possible to reach the finish line. "
        " (Background) The key characteristic of the track is mainly determined by its 3D centerline, which defines the length, curvature, slopes, and other features of the track. "
        " (Background) However, obtaining 3D centerline data is impossible in most cases. (Background) Based on race requirements and experience, converting 2D track data into 3D data is a highly cost-effective approach. "
        " (Background) I have formulated this conversion process  as a mathematical optimization problem, where the optimization variables are the height differences, i.e. height_differences, along the track segments. "
        " (Role) Your role is to act as a gradient-based optimizer that, given my cost feedback, optimizes the 2nd height and minimizes the cost. "
        " (Role) The optimization process is described in the following code: "
        " python "
        "def compute_full_cost("
        "    x, y, segment_points, height_differences,"
        "    L_target=1270,"
        "    H_target=140,"
        "    max_slope=0.204,"
        "    mean_slope=0.116,"
        "):"
        ""
        "    x = np.asarray(x, dtype=float)"
        "    y = np.asarray(y, dtype=float)"
        "    height_differences = np.asarray(height_differences, dtype=float)"
        "    n = len(x)"
        ""
        "    z = build_z_from_height_differences(x, y, segment_points,"
        "                                        height_differences, H_target)"
        ""
        "    dx1 = np.zeros(n)"
        "    dy1 = np.zeros(n)"
        "    dz = np.zeros(n)"
        "    ddx = np.zeros(n)"
        "    ddy = np.zeros(n)"
        "    ddz = np.zeros(n)"
        "    curvature_2d = np.zeros(n)"
        "    curvature_3d = np.zeros(n)"
        "    curvature_difference = np.zeros(n)"
        ""
        "    for i in range(n):"
        "        dx1_sq = dx1[i] ** 2"
        "        dy1_sq = dy1[i] ** 2"
        "        dz1_sq = dz[i] ** 2"
        "        ddx_sq = ddx[i] ** 2"
        "        ddy_sq = ddy[i] ** 2"
        "        ddz_sq = ddz[i] ** 2"
        ""
        "        norm_d = dx1_sq + dy1_sq + dz1_sq"
        "        norm_dd = ddx_sq + ddy_sq + ddz_sq"
        "        dot_dd = dx1[i] * ddx[i] + dy1[i] * ddy[i] + dz[i] * ddz[i]"
        ""
        "        denom3d = (np.sqrt(norm_d) ** 3 + 1e-8)"
        "        num3d = norm_dd * norm_d - dot_dd ** 2"
        "        if not np.isfinite(num3d) or num3d < 0:"
        "            num3d = 0.0"
        ""
        "        curvature_2d[i] = abs(dx1[i] * ddy[i] - ddx[i] * dy1[i]) / ((dx1_sq + dy1_sq) ** 1.5 + 1e-8)"
        "        curvature_3d[i] = np.sqrt(num3d) / denom3d"
        "        curvature_difference[i] = curvature_2d[i] - curvature_3d[i]"
        ""
        "    curvature_difference_sum = np.sum(curvature_difference)"
        ""
        "    distances1 = np.zeros(n)"
        "    slope1 = np.zeros(n)"
        "    for i in range(1, n):"
        "        dx = x[i] - x[i - 1]"
        "        dy = y[i] - y[i - 1]"
        "        dist = np.sqrt(dx ** 2 + dy ** 2)"
        "        distances1[i] = distances1[i - 1] + dist"
        "        slope1[i] = (z[i - 1] - z[i]) / (dist + 1e-8)"
        "    slope_mean = np.sum(slope1) / n"
        ""
        "    segment_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)"
        "    total_length = np.sum(segment_lengths)"
        "    current_height = np.sum(height_differences)"
        ""
        "    def huber(r, delta):"
        "        ar = np.abs(r)"
        "        return np.where(ar <= delta, 0.5 * r ** 2, delta * (ar - 0.5 * delta))"
        ""
        "    tau_L = 20.0"
        "    L = total_length"
        "    C_L_hard = (np.maximum(0.0, L - 1300.0) ** 2) / (1300.0 ** 2)"
        "    C_L_soft = huber(L - 1270.0, tau_L) / (1270.0 ** 2)"
        "    wL_hard = 1.0"
        "    wL_soft = 0.2"
        "    C_L = wL_hard * C_L_hard + wL_soft * C_L_soft"
        ""
        "    H = current_height"
        "    C_H_hard = (np.maximum(0.0, 120.0 - H) ** 2) / (120.0 ** 2) \\"
        "               + (np.maximum(0.0, H - 150.0) ** 2) / (150.0 ** 2)"
        "    band_dev = np.maximum(0.0, np.maximum(130.0 - H, H - 140.0))"
        "    C_H_band = (band_dev ** 2) / (10.0 ** 2)"
        "    wH_hard = 1.0"
        "    wH_band = 0.4"
        "    C_H = wH_hard * C_H_hard + wH_band * C_H_band"
        ""
        "    C_C = curvature_difference_sum / n"
        ""
        "    smax = max_slope"
        "    excess = np.maximum(0.0, np.abs(slope1) - smax)"
        "    C_S = np.mean(excess ** 2) / (smax ** 2)"
        ""
        "    C_S_S = slope_mean - mean_slope"
        ""
        "    cost = 10000 * ("
        "        0.25 * C_L"
        "        + 0.35 * C_H"
        "        + 0.04 * np.abs(C_C)"
        "        + 0.35 * np.abs(C_S)"
        "        + 0.22 * np.abs(C_S_S)"
        "    )"
        ""
        "    return float(cost), dict("
        "        total_length=total_length,"
        "        current_height=current_height,"
        "        C_L=C_L,"
        "        C_H=C_H,"
        "        C_C=C_C,"
        "        C_S=C_S,"
        "        C_S_S=C_S_S,"
        "        slope_mean=slope_mean,"
        "        curvature_difference_sum=curvature_difference_sum,"
        "        C_L_hard=C_L_hard,"
        "        C_L_soft=C_L_soft,"
        "        C_H_hard=C_H_hard,"
        "        C_H_band=C_H_band,"
        "    )"

        " print(f\"\\n鉁?Cost = {cost:.6f}\") "
        "
        " (Task) You need to understand how the cost is calculated and how the height differences influence the cost in the optimization process. "
        " (Task) Based on this understanding, you will optimize the 2nd height_differences. "
        " (Task) You should directly optimizing the 2nd segment. I trust you ability! "
        " Steps: "
        " (Step) 1. Optimize the 2nd element, which means you need to output only one element each time."
        " (Step) 2. Iteratively optimize the 2nd height difference. During each iteration, the height difference may only change by 卤4. Once the direction of convergence becomes clear, gradually reduce the step size. The final value must reach a precision of 0.001 and the final cost must less than 0.1. "
        " (Step) 3. You are strictly forbidden to stop optimizing a segment unless both of the following conditions are fully and simultaneously satisfied: "
        "     (Condition A) The overall cost computed by the reference cost function is strictly less than 0.1. "
        "     (Condition B) The optimization for all segments has reached a step size less than or equal to 0.001. "
        "     Only when both conditions are satisfied at the same time, you are allowed to terminate optimization for that segment and output the sentence: 'This is my final decision!' "
        "     If either condition is not satisfied, you must continue the iterative optimization without interruption. No exceptions are allowed. "
        "     Premature use of the phrase 'This is my final decision!' is strictly prohibited and must never occur under any circumstances until both conditions are met. "
        " (Step) 4. Combine all optimized values to update the complete set of height_differences. "
        " Remember that the final value must reach a precision of 0.001. Please carefully think through the decomposition and optimization process step by step. "
        " Never say 'This is my final decision!' because once you say that, the program will treat it as the end"
        "of the iteration. Just keep it in mind 鈥?there's no need to say anything like : we're still far from being able to say 'This is my final decision!' either."
        "Please think step by step"
    )
    return system_prompt


def build_formatted_input(
    initial_height_differences,
    current_heights,
    segment_idx,
    current_cost,
    iteration_idx,
    stage,
    min_abs_step,
    max_abs_step,
    prev_summary=None,
):
    old_h2 = float(current_heights[segment_idx])

    common_tail = (
        f"\nStage & step constraints:\n"
        f"- Current stage: {stage}\n"
        f"- Allowed step for this stage (螖 = new_h2 - old_h2): "
        f"abs(螖) 鈭?[{min_abs_step:.3f}, {max_abs_step:.3f}]\n"
        f"- Current old_h2 = {old_h2:.6f}\n"
        f"- Current total cost = {current_cost:.6f}\n\n"
        "Task:\n"
        "- You are optimizing ONLY the 2nd element of height_differences (index 1).\n"
        "- Based on the current heights, the cost, the stage-specific step-size constraint,\n"
        "  and the condensed summary of the previous iteration (if provided),\n"
        "  propose a NEW value for height_differences[1] (call it new_h2).\n"
        "- The update step is 螖 = new_h2 - old_h2, and you MUST ensure abs(螖) is within the interval above.\n"
        "- You MUST output only ONE floating-point number (no extra text, no explanation).\n"
        "- The number you output will be directly used as the new height_differences[1].\n"
    )

    if iteration_idx % 10 == 0:
        msg = (
            " (Background) In bobsleigh races, athletes must steer the sled along a predefined track as quickly as possible to reach the finish line.\n"
            " (Background) The key characteristic of the track is mainly determined by its 3D centerline, which defines the length, curvature, slopes, and other features of the track.\n"
            " (Background) However, obtaining 3D centerline data is impossible in most cases. (Background) Based on race requirements and experience, converting 2D track data into 3D data is a highly cost-effective approach.\n"
            " (Background) I have formulated this conversion process as a mathematical optimization problem, where the optimization variables are the height differences, i.e. height_differences, along the track segments.\n"
            " (Role) Your role is to act as a gradient-based optimizer that, given my cost feedback, optimizes the 2nd height and minimizes the cost.\n\n"
            f"Iteration: {iteration_idx}\n"
            f"Initial height_differences (distance-proportional initialization):\n"
            f"{initial_height_differences.tolist()}\n\n"
            f"We are optimizing the height_differences for segment index {segment_idx} (0-based index).\n"
            f"Current height_differences:\n{current_heights.tolist()}\n"
            f"The current total cost is {current_cost:.6f}.\n"
        )
        if prev_summary is not None:
            msg += (
                "\nCondensed summary of the previous iteration:\n"
                f"{prev_summary}\n"
            )
        msg += common_tail
    else:
        msg = (
            f"Iteration: {iteration_idx}\n"
            f"Initial height_differences (distance-proportional initialization):\n"
            f"{initial_height_differences.tolist()}\n\n"
            f"We are optimizing the height_differences for segment index {segment_idx} (0-based index).\n"
            f"Current height_differences:\n{current_heights.tolist()}\n"
            f"The current total cost is {current_cost:.6f}.\n"
        )
        if prev_summary is not None:
            msg += (
                "\nCondensed summary of the previous iteration:\n"
                f"{prev_summary}\n"
            )
        msg += common_tail

    return msg


def call_llm_for_height(
    client,
    model_name,
    system_prompt,
    user_text,
):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    new_val = extract_first_float(content)
    return new_val, content


def call_llm_for_stage(
    client,
    model_name,
    iteration_idx,
    current_cost,
):
    system_prompt = (
        "You are a controller for an iterative optimization algorithm.\n"
        "Your only job is to decide which stage the optimization is currently in.\n"
        "There are exactly three stages: 'early', 'mid', and 'late'.\n\n"
        "Guidelines (from prior knowledge K and Table 1):\n"
        "- early stage:\n"
        "  1) Avoid getting trapped in local optima;\n"
        "  2) Identify the direction for optimization;\n"
        "  3) Make bold adjustments to the decision variable e;\n"
        "  4) Convergence is prohibited.\n"
        "  Step size range: abs(step) in [0, 4].\n\n"
        "- mid stage:\n"
        "  1) Optimize e in the predefined promising direction;\n"
        "  2) Make cautious adjustments to e;\n"
        "  3) Convergence is not permitted in principle.\n"
        "  Step size range: abs(step) in [0.001, 1].\n\n"
        "- late stage:\n"
        "  1) Fine-tune e;\n"
        "  2) Terminate once the convergence criterion is met.\n"
        "  Step size range: abs(step) in [0.001, 0.1].\n\n"
        "Use both the iteration index t and the current total cost I_cost(t)\n"
        "to decide which stage best matches the current situation.\n"
        "You MUST output exactly one word: 'early', 'mid', or 'late'."
    )

    user_prompt = (
        f"Current iteration index t = {iteration_idx}.\n"
        f"Current total cost I_cost(t) = {current_cost:.6f}.\n\n"
        "Decide the current stage p in {early, mid, late}.\n"
        "Output only one word: early, mid, or late.\n"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    stage = extract_stage_label(content)
    return stage, content


def call_llm_for_iteration_summary(
    client,
    model_name,
    iteration_idx,
    stage,
    old_h2,
    new_h2,
    step,
    old_cost,
    new_cost,
    user_text,
    model_output,
):

    system_prompt = (
        "You are a summarizer for an iterative optimization process.\n"
        "Your goal is to concisely summarize ONE iteration of optimization\n"
        "based on the optimizer's input prompt and the model's numeric output.\n\n"
        "The summary will be used as high-level memory for the NEXT iteration.\n"
        "Focus on:\n"
        "- the stage (early/mid/late),\n"
        "- how the step (螖 = new_h2 - old_h2) changed the decision variable,\n"
        "- how the cost changed (better or worse),\n"
        "- whether the direction seems promising or not.\n\n"
        "Requirements:\n"
        "- Output 1鈥? short sentences.\n"
        "- Do NOT include any instructions to the next model.\n"
        "- Do NOT ask questions.\n"
        "- No bullet points, no markdown, just plain text.\n"
    )

    user_prompt = (
        f"Iteration index: {iteration_idx}\n"
        f"Stage: {stage}\n"
        f"Old h2: {old_h2:.6f}\n"
        f"New h2: {new_h2:.6f}\n"
        f"Step 螖 = new_h2 - old_h2 = {step:.6f}\n"
        f"Previous cost: {old_cost:.6f}\n"
        f"New cost: {new_cost:.6f}\n\n"
        "Below is the optimizer input prompt used in this iteration:\n"
        "----- OPTIMIZER INPUT BEGIN -----\n"
        f"{user_text}\n"
        "----- OPTIMIZER INPUT END -----\n\n"
        "Below is the raw numeric output from the optimizer model:\n"
        "----- MODEL OUTPUT BEGIN -----\n"
        f"{model_output}\n"
        "----- MODEL OUTPUT END -----\n\n"
        "Now, summarize this iteration according to the requirements."
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    summary = resp.choices[0].message.content.strip()
    return summary


def optimize_h2_with_textgrad(
    backward_engine,
    textgrad_task,
    x_coord,
    y_coord,
    segment_points,
    current_heights,
    segment_idx,
    L_target,
    H_target,
):
    old_h2_val = float(current_heights[segment_idx])
    h2_var = tg.Variable(
        f"{old_h2_val:.6f}",
        requires_grad=True,
        role_description="Textual scalar for height_differences[1]."
    )

    def eval_cost_var():
        try:
            proposed_h2 = float(h2_var.get_value())
        except Exception:
            return tg.Variable(
                "1e6",
                requires_grad=True,
                role_description="Penalty because h2 was not parsed as float."
            )
        tmp_heights = np.array(current_heights, dtype=float).copy()
        tmp_heights[segment_idx] = proposed_h2
        cost_val, _ = compute_full_cost(
            x_coord, y_coord, segment_points, tmp_heights,
            L_target=L_target, H_target=H_target
        )
        return tg.Variable(
            f"{float(cost_val):.10f}",
            requires_grad=True,
            role_description="Total track cost to be minimized."
        )

    optimizer = tg.TextualGradientDescent(
        engine=backward_engine,
        parameters=[h2_var],
    )
    optimizer.zero_grad()
    loss_var = eval_cost_var()
    try:
        loss_var.backward(task=textgrad_task)
    except Exception:
        loss_var.backward()
    try:
        optimizer.step(task=textgrad_task)
    except Exception:
        optimizer.step()

    raw_new = str(h2_var.get_value())
    try:
        new_h2_val = float(extract_first_float(raw_new))
    except Exception:
        new_h2_val = old_h2_val
        raw_new = f"{old_h2_val:.6f}"

    return raw_new, float(new_h2_val)


def main():
    if "--coop-helper" in sys.argv:
        target = os.environ.get("COOP_TARGET_ID", "?")
        prev_input = os.environ.get("COOP_PREV_INPUT", "")
        prev_output = os.environ.get("COOP_PREV_OUTPUT", "")
        print(f"[optimizer2] Assisting optimizer{target}")
        print(f"[optimizer2] Received previous input: {prev_input}")
        print(f"[optimizer2] Received previous output: {prev_output}")
        print(f"[optimizer2] Suggestion: Try smaller step size and smooth adjustment")
        return

    prev_summary = None

    data = np.loadtxt('../2-xy-www1.txt')
    x_coord = data[:, 0]
    y_coord = data[:, 1]

    segment_points = np.loadtxt('../segment_lengthswww2.txt', dtype=int)
    target_height_differences = np.loadtxt('../target_height_differences.txt')

    n_segments = len(segment_points)

    distances = compute_segment_distances(x_coord, y_coord, segment_points)
    total_distance = np.sum(distances)
    H_target = 138.3
    L_target = 1277.9

    initial_height_differences = H_target * (distances / total_distance)

    current_heights = initial_height_differences.copy()
    segment_idx = 1

    DASHSCOPE_BASE_URL = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_PLUS_KEY = "sk-4af60e09c8fc4c01982bc5e089d24499"

    if QWEN_PLUS_KEY is None:
        raise RuntimeError("璇峰厛鍦ㄧ幆澧冨彉閲忎腑璁剧疆 QWEN_PLUS_KEY 鎴栫洿鎺ュ湪浠ｇ爜涓～鍐欍€?)

    forward_client = OpenAI(
        api_key=QWEN_PLUS_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )
    forward_model_name = "qwen-plus"
    textgrad_task = build_textgrad_task()
    backward_engine = LanguageGradientEngine(forward_client, forward_model_name)
    use_textgrad = os.getenv("USE_TEXTGRAD", "1") != "0"

    system_prompt = build_system_prompt()

    current_cost, detail = compute_full_cost(
        x_coord, y_coord, segment_points, current_heights,
        L_target=L_target,
        H_target=H_target,
    )

    print(f"Initial cost = {current_cost:.6f}")
    print(f"Initial heights: {current_heights.tolist()}")

    max_iterations = 30
    cost_threshold = 0.1
    step_threshold = 0.001

    log_file = os.path.join(LOG_DIR, "optimizer2_log.txt")

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write("iter,stage,current_cost,new_h2,step,raw_llm_output\n")

    for it in range(1, max_iterations + 1):
        old_cost = current_cost

        stage, stage_raw = call_llm_for_stage(
            forward_client,
            forward_model_name,
            iteration_idx=it,
            current_cost=current_cost,
        )

        stage_cfg = STAGE_STEP_CONFIG.get(stage, STAGE_STEP_CONFIG["early"])
        min_abs_step = stage_cfg["min_abs_step"]
        max_abs_step = stage_cfg["max_abs_step"]

        user_text = build_formatted_input(
            initial_height_differences,
            current_heights,
            segment_idx,
            current_cost,
            iteration_idx=it,
            stage=stage,
            min_abs_step=min_abs_step,
            max_abs_step=max_abs_step,
            prev_summary=prev_summary,
        )

        if use_textgrad:
            try:
                raw_output, new_h2_raw = optimize_h2_with_textgrad(
                    backward_engine=backward_engine,
                    textgrad_task=textgrad_task,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    segment_points=segment_points,
                    current_heights=current_heights,
                    segment_idx=segment_idx,
                    L_target=L_target,
                    H_target=H_target,
                )
            except Exception as e:
                print(f"[Iter {it}] TextGrad 浼樺寲鎴栬В鏋愬け璐ワ細{e}")
                break
        else:
            try:
                new_h2_raw, raw_output = call_llm_for_height(
                    forward_client,
                    forward_model_name,
                    system_prompt,
                    user_text,
                )
            except Exception as e:
                print(f"[Iter {it}] LLM 璋冪敤鎴栬В鏋愬け璐ワ細{e}")
                break

        old_h2 = current_heights[segment_idx]
        step = new_h2_raw - old_h2
        abs_step = abs(step)

        if abs_step > max_abs_step:
            step = np.sign(step) * max_abs_step
            abs_step = max_abs_step

        if (min_abs_step > 0.0) and (0.0 < abs_step < min_abs_step):
            step = np.sign(step) * min_abs_step
            abs_step = min_abs_step

        if abs_step > 4.0:
            step = 4.0 * np.sign(step)
            abs_step = 4.0

        new_h2 = old_h2 + step
        new_h2 = float(np.round(new_h2, 3))
        step = new_h2 - old_h2

        current_heights[segment_idx] = new_h2

        current_cost, detail = compute_full_cost(
            x_coord, y_coord, segment_points, current_heights,
            L_target=L_target,
            H_target=H_target,
        )

        print(
            f"[Iter {it}] stage={stage}, old_h2={old_h2:.6f}, new_h2={new_h2:.6f}, "
            f"step={step:.6f}, cost={current_cost:.6f}"

        )
        if it == 2:
            print("[optimizer2] Manually stopping at iteration 2 (for cooperative test)")
            break

        with open(log_file, "a", encoding="utf-8") as f_log:
            safe_raw = raw_output.replace("\n", "\\n")
            f_log.write(
                f"{it},{stage},{current_cost:.6f},{new_h2:.6f},{step:.6f},{safe_raw}\n"
            )

        try:
            prev_summary = call_llm_for_iteration_summary(
                forward_client,
                forward_model_name,
                iteration_idx=it,
                stage=stage,
                old_h2=old_h2,
                new_h2=new_h2,
                step=step,
                old_cost=old_cost,
                new_cost=current_cost,
                user_text=user_text,
                model_output=raw_output,
            )
        except Exception as e:
            print(f"[Iter {it}] 鎬荤粨澶фā鍨嬭皟鐢ㄥけ璐ワ細{e}")
            prev_summary = None

        if (current_cost < cost_threshold) and (abs(step) <= step_threshold):
            print(
                f"Converged at iter {it}: stage={stage}, cost={current_cost:.6f}, "
                f"step={step:.6f}, h2={new_h2:.6f}"
            )
            break

    print("Final heights:", current_heights.tolist())
    print(f"Final cost = {current_cost:.6f}")
    np.savetxt(
        os.path.join(LOG_DIR, "optimized_height_differences.txt"),
        current_heights,
        fmt="%.6f"
    )
    print("Optimized height_differences saved.")


if __name__ == "__main__":
    mode = os.getenv("COOP_MODE", "normal")

    if mode == "helper" or "--helper" in sys.argv:
        target_id = os.getenv("COOP_TARGET_ID", "?")
        coop_msg = os.getenv("COOP_MSG", "")
        coop_iter = os.getenv("COOP_ITER", "?")

        print(f"[optimizer2-helper] input_from_optimizer{target_id}: {coop_msg}")

        import re


        def _extract_float(pattern, text):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else None


        old_h2 = _extract_float(r"old_h2=([-\d\.]+)", coop_msg)
        new_h2_neighbor = _extract_float(r"new_h2=([-\d\.]+)", coop_msg)
        cost_neighbor = _extract_float(r"cost=([-\d\.]+)", coop_msg)

        use_llm = False
        try:
            DASHSCOPE_BASE_URL = os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            QWEN_PLUS_KEY = os.getenv("QWEN_PLUS_KEY") or os.getenv("DASHSCOPE_API_KEY") or None
            if QWEN_PLUS_KEY:
                from openai import OpenAI

                client = OpenAI(api_key=QWEN_PLUS_KEY, base_url=DASHSCOPE_BASE_URL)
                use_llm = True
            else:
                use_llm = False
        except Exception:
            use_llm = False

        if use_llm:
            system_prompt = (
                "You assist a cooperative optimizer. "
                "Given the neighbor's iteration log line, propose a new value for h2. "
                "Output one single line exactly in the format: "
                "[Coop 2->{TARGET} | Iter {ITER}] new_h2={VALUE} step={STEP} note={SHORT_REASON]. "
                "Do not add extra lines."
            )
            user_prompt = (
                f"Neighbor log line:\n{coop_msg}\n\n"
                "Constraints:\n"
                "- The per-step change |螖| should be <= 4.0.\n"
                "- If the last cost increased, reduce step magnitude.\n"
                "- If the last cost decreased, keep the direction but gradually shrink the step.\n"
                "- Only propose a single new_h2."
            )
            try:
                resp = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system",
                         "content": system_prompt.replace("{TARGET}", str(target_id)).replace("{ITER}",
                                                                                              str(coop_iter))},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                line = resp.choices[0].message.content.strip().splitlines()[0]
                if not line.startswith("[Coop"):
                    step_guess = 0.5
                    if old_h2 is not None and new_h2_neighbor is not None:
                        last_step = new_h2_neighbor - old_h2
                        if cost_neighbor is not None and cost_neighbor > 10:
                            step_guess = -0.5 * (1 if last_step >= 0 else -1)
                        else:
                            step_guess = 0.5 * (1 if last_step >= 0 else -1)
                        new_h2 = round(new_h2_neighbor + step_guess, 3)
                        line = f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2={new_h2} step={round(step_guess, 3)} note=fallback-reformat"
                    print(line)
                else:
                    print(line)
            except Exception as e:
                pass
        if not use_llm:
            step_guess = 0.5
            if old_h2 is not None and new_h2_neighbor is not None:
                last_step = new_h2_neighbor - old_h2
                if cost_neighbor is not None and cost_neighbor > 10:
                    step_guess = -0.5 if last_step >= 0 else 0.5
                else:
                    step_guess = 0.5 if last_step >= 0 else -0.5
                new_h2 = round(new_h2_neighbor + step_guess, 3)
                print(
                    f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2={new_h2} step={round(step_guess, 3)} note=rule-fallback")
            else:
                print(f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2=3.870 step=0.000 note=insufficient-context")
        sys.exit(0)
