import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import fire
from cycler import cycler

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.prop_cycle"] = cycler(
    "color", ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
)
mpl.rcParams["axes.axisbelow"] = True

FIG_HALF_WIDTH = (5, 3)


def pre_plot(figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.grid(linestyle=":")


def post_plot(fp_output):
    for spine in ("top", "right", "bottom", "left"):
        plt.gca().spines[spine].set_visible(False)
    plt.savefig(fp_output, bbox_inches="tight")
    plt.close()


def effects_R(
    dir_input: str = "data",
    dir_output: str = "plots",
    pval_cutoff: float = 0.05,
    group: str = "extracted_results",
    outcome: str = "click",
    cos_sim_cutoff: float = 0.85,
):

    os.makedirs(dir_output, exist_ok=True)

    if group == "extracted_results":
        treatments = [
            "direct_answer",
            "featured_snippet",
            "top_image_carousel",
            "top_stories",
            "knowledge_panel_rhs",
        ]

    elif group == "google_services":
        treatments = [
            "local_results",
            "images",
            "map_results",
            "scholarly_articles",
            "shopping_ads",
            "ad",
            "videos",
        ]

    xlabels = {
        "time_elapsed": "Time on Page Difference (seconds)",
        "click": "CTR Difference",
        "organic_click_1p": "Organic CTR to 1st-Party Domains Difference",
        "organic_click_3p": "Organic CTR to 3rd-Party Domains Difference",
    }

    coefs = []
    ci_lower = []
    ci_upper = []
    pvals = []
    for trt in treatments:

        trt_dir = "direct_answer_combined" if trt == "direct_answer" else trt
        fp = os.path.join(
            dir_input,
            trt_dir,
            f"att_{outcome}_{cos_sim_cutoff}.csv",
        )
        data = pd.read_csv(fp)
        coefs.append(data["estimate"].values[0])
        ci_lower.append(data["conf.low"].values[0])
        ci_upper.append(data["conf.high"].values[0])
        pvals.append(data["p.value"].values[0] < pval_cutoff)

    data = pd.DataFrame(
        {
            "estimate": np.concatenate((coefs, ci_lower, ci_upper)),
            "Component": np.tile(
                [t.replace("_", "-").replace("-rhs", "") for t in treatments], 3
            ),
            f"Significant (p<{pval_cutoff})": np.tile(pvals, 3),
        }
    )

    pre_plot(FIG_HALF_WIDTH)
    ax = sns.pointplot(
        x="estimate",
        y="Component",
        data=data,
        join=False,
        hue=f"Significant (p<{pval_cutoff})",
    )

    ymin, ymax = ax.get_ylim()
    ax.set(xlabel=xlabels[outcome])
    ax.vlines(0, ymin, ymax, colors="#e42536", linestyles="dashed")
    if outcome != "time_elapsed":
        ax.get_legend().remove()

    cos_sim_cutoff = str(cos_sim_cutoff).replace(".", "")
    fp_output = os.path.join(dir_output, f"{group}_{outcome}.pdf")
    post_plot(fp_output)


def all_effects_plots():
        
    outcomes = {
        "extracted_results": ["click", "time_elapsed"],
        "google_services": ["organic_click_1p", "organic_click_3p"],
    }

    for group in outcomes:
        for outcome in outcomes[group]:
            effects_R(outcome=outcome, group=group)


def evalue(
    dir_input: str = "data",
    fp_output: str = "plots/evalue.pdf",
    cos_sim_cutoff: float = 0.85,
):
    
    extracted_trt = [
        "direct_answer",
        "featured_snippet",
        "knowledge_panel_rhs",
        "top_image_carousel",
        "top_stories",
    ]
    google_trt = [
        "map_results",
        "local_results",
        "images",
        "scholarly_articles",
        "shopping_ads",
        "videos",
        "ad",
    ]
    trt_dict = {0: extracted_trt, 1: google_trt}
    out_dict = {
        0: ["click", "time_elapsed"],
        1: ["organic_click_1p", "organic_click_3p"],
    }
    out_map = {
        "click": "clicks",
        "time_elapsed": "time",
        "organic_click_1p": "1p clicks",
        "organic_click_3p": "3p clicks",
    }

    pre_plot((4, 3))

    data = []
    for i in range(2):
        for trt in trt_dict[i]:
            for out in out_dict[i]:
                trt_dir = "direct_answer_combined" if trt == "direct_answer" else trt
                evalue_fp = os.path.join(
                    dir_input, trt_dir, f"evalue_{out}_{cos_sim_cutoff}.csv"
                )
                evalue = pd.read_csv(evalue_fp)
                evalue.columns = ["Unmeasured Confounding", "E-Value"]
                
                if len(evalue) == 2:
                    continue

                evalue = evalue[evalue["Unmeasured Confounding"] != "tip_point"]

                evalue["Effect"] = trt.replace("_", "-") + ", " + out_map[out]
                evalue["E-Value Type"] = evalue["Unmeasured Confounding"].apply(
                    lambda t: "E-Value"
                    if t.startswith("tip")
                    else "Observed Covariate\nE-Value"
                )
                evalue = evalue.replace(
                    {
                        "tip_ci": "needed to tip \nconfidence interval",
                        "no_comps": "same size as \ncomponent covariates",
                        "no_qry": "same size as \nquery covariates",
                        "no_topics": "same size as \ntopic covariates",
                        "no_behavior": "same size as \nbehavior covariates",
                    }
                )
                data.append(evalue)
                
    data = pd.concat(data)
    sns.scatterplot(
        x="E-Value",
        y="Effect",
        hue="E-Value Type",
        style="E-Value Type",
        size="E-Value Type",
        style_order=["Observed Covariate\nE-Value", "E-Value"],
        size_order=["E-Value", "Observed Covariate\nE-Value"],
        data=data,
        legend=True,
    )
    post_plot(fp_output)


if __name__ == "__main__":
    fire.Fire()
