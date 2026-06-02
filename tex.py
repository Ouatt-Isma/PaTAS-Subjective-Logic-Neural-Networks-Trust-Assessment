import os
import shutil
from collections import defaultdict

RESULTS_DIR = "results"
OUTPUT_DIR = "latex"
TEX_FILE = "filelist.tex"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Copy PDFs, tracking original folder name for captions ────────────
# Each entry: (dest_filename, original_folder_name, pdf_stem)
all_files = []

for folder_name in sorted(os.listdir(RESULTS_DIR)):
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".pdf"):
            continue
        src = os.path.join(folder_path, filename)
        safe_folder = folder_name.replace(",", "_").replace(" ", "_")
        pdf_stem = filename[:-4]
        dest_name = f"{safe_folder}__{pdf_stem}.pdf"
        shutil.copy2(src, os.path.join(OUTPUT_DIR, dest_name))
        all_files.append((dest_name, folder_name, pdf_stem))

print(f"Copied {len(all_files)} PDF(s) to '{OUTPUT_DIR}/'")

# ── Step 2: Caption helper ───────────────────────────────────────────────────
def make_caption(orig_folder):
    name = orig_folder
    for prefix in ("NN_Train_", "PTAS_Eval_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    ps = None
    if "_PathSize_" in name:
        name, ps = name.rsplit("_PathSize_", 1)
    eps = None
    if "_eps_" in name:
        name, eps = name.rsplit("_eps_", 1)
    caption = name.replace("_", r"\_")
    if eps is not None:
        caption += f", $\\varepsilon$={eps}"
    if ps and ps != "None":
        caption += f", ps={ps}"
    return caption


# ── Step 3: Parse PTAS folder into (dataset, trust_combo) for grouping ───────
def parse_ptas(folder_name):
    """Returns (dataset, hidden, trust_combo, eps, path_size)."""
    name = folder_name[len("PTAS_Eval_"):]
    ps = "None"
    if "_PathSize_" in name:
        name, ps = name.rsplit("_PathSize_", 1)
    name, eps = name.rsplit("_eps_", 1)
    # name = "{dataset}_{hidden[_hidden]}_{trust_combo}"
    tokens = name.split("_", 1)
    dataset = tokens[0]
    rest = tokens[1]
    # consume leading numeric tokens as hidden size
    rtokens = rest.split("_")
    i = 0
    while i < len(rtokens) and rtokens[i].isdigit():
        i += 1
    hidden = "_".join(rtokens[:i])
    trust = "_".join(rtokens[i:])
    return dataset, hidden, trust, eps, ps


def fmt_trust_label(trust):
    """Human-readable trust combo for section headings."""
    parts = trust.split("_", 1)
    def fmt(t):
        return f"({t.replace(',', ', ')})" if "," in t else t
    if len(parts) == 2:
        return f"{fmt(parts[0])} -- {fmt(parts[1])}"
    return trust


# ── Step 4: Categorise files ─────────────────────────────────────────────────
nn_by_type = defaultdict(list)   # pdf_stem -> [(dest_name, orig_folder)]
# PTAS: nested dict  dataset -> trust_combo -> [(dest_name, orig_folder, eps, ps)]
ptas_grouped = defaultdict(lambda: defaultdict(list))

for dest_name, orig_folder, pdf_stem in all_files:
    if orig_folder.startswith("NN_Train_"):
        nn_by_type[pdf_stem].append((dest_name, orig_folder))
    elif orig_folder.startswith("PTAS_Eval_"):
        dataset, hidden, trust, eps, ps = parse_ptas(orig_folder)
        ptas_grouped[dataset][trust].append((dest_name, orig_folder, eps, ps))

# Sort PTAS entries within each trust group by eps value numerically
for dataset in ptas_grouped:
    for trust in ptas_grouped[dataset]:
        ptas_grouped[dataset][trust].sort(key=lambda x: float(x[2]))


# ── Step 5: Write LaTeX ──────────────────────────────────────────────────────
PLOT_TYPE_LABELS = {
    "accuracy_evolution":   "Accuracy Evolution",
    "ipta_table2a":         "IPTA -- Table 2a",
    "ipta_trust_2a_vs_2c":  "IPTA -- Trust 2a vs 2c",
}

def write_nn_group(f, entries, section_label):
    """9 subfloats per figure, 3-column grid."""
    n = len(entries)
    total_figs = (n + 8) // 9
    for i, (dest_name, orig_folder) in enumerate(entries):
        if i % 9 == 0:
            f.write("\\begin{figure}[htbp]\n\\centering\n")
        caption = make_caption(orig_folder)
        path = f"{OUTPUT_DIR}/{dest_name}".replace("\\", "/")
        f.write(f"\\subfloat[{{{caption}}}]{{%\n")
        f.write(f"  \\includegraphics[width=0.3\\textwidth]{{{path}}}%\n")
        f.write("}\n")
        if (i + 1) % 3 == 0:
            f.write("\\\\\n")
        if (i + 1) % 9 == 0 or (i + 1) == n:
            grp = i // 9 + 1
            fig_cap = section_label if total_figs == 1 else f"{section_label} ({grp}/{total_figs})"
            f.write(f"\\caption{{{fig_cap}}}\n\\end{{figure}}\n\n")


def write_ptas_figure(f, dest_name, orig_folder):
    caption = make_caption(orig_folder)
    path = f"{OUTPUT_DIR}/{dest_name}".replace("\\", "/")
    f.write("\\begin{figure}[htbp]\n\\centering\n")
    f.write(f"  \\includegraphics[width=0.8\\textwidth]{{{path}}}\n")
    f.write(f"\\caption{{{caption}}}\n\\end{{figure}}\n\n")


with open(TEX_FILE, "w") as f:

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 – NN Training Results
    # ════════════════════════════════════════════════════════════════════════
    f.write("\\section{NN Training Results}\n\n")

    for plot_type in sorted(nn_by_type.keys()):
        entries = nn_by_type[plot_type]
        label = PLOT_TYPE_LABELS.get(plot_type, plot_type.replace("_", " ").title())
        f.write(f"\\subsection{{{label}}}\n\n")
        write_nn_group(f, entries, label)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 – PTAS Evaluation Results
    # ════════════════════════════════════════════════════════════════════════
    f.write("\\section{PTAS Evaluation Results}\n\n")

    DATASET_LABELS = {"cancer": "Cancer Dataset", "mnist": "MNIST Dataset"}

    for dataset in sorted(ptas_grouped.keys()):
        ds_label = DATASET_LABELS.get(dataset, dataset.capitalize())
        f.write(f"\\subsection{{{ds_label}}}\n\n")

        for trust in sorted(ptas_grouped[dataset].keys()):
            trust_label = fmt_trust_label(trust)
            f.write(f"\\subsubsection{{{trust_label}}}\n\n")
            for dest_name, orig_folder, eps, ps in ptas_grouped[dataset][trust]:
                write_ptas_figure(f, dest_name, orig_folder)

print(f"LaTeX written to '{TEX_FILE}'")
print(f"  NN plot types : {dict((k, len(v)) for k, v in nn_by_type.items())}")
print(f"  PTAS datasets : { {ds: list(ptas_grouped[ds].keys()) for ds in ptas_grouped} }")
