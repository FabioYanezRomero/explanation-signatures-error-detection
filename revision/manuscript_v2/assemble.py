#!/usr/bin/env python3
"""Assemble main_v2.tex from the submitted main.tex and the revised section files."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "main.tex").read_text(encoding="utf-8")

def between(text, start, end, start_inclusive=True):
    i = text.index(start); j = text.index(end, i + len(start))
    return (i if start_inclusive else i + len(start)), j

def read(name):
    return (HERE / name).read_text(encoding="utf-8")

front = read("frontmatter_v2.tex")
meth = read("methodology_v2.tex")
res = read("results_v2.tex")
concl = read("conclusions_v2.tex")

# --- pieces from the revised files
new_title = re.search(r"\\title\{.*?\}", front, re.S).group(0)
new_abstract = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", front, re.S).group(0)
new_keyword = re.search(r"\\begin\{keyword\}.*?\\end\{keyword\}", front, re.S).group(0)
new_intro = front[front.index(r"\section{Introduction}"):]
new_meth = meth[meth.index(r"\section{Methodology}"):]
new_exp = ""  # Section 4 merged into "Experiments and Results" (results_v2.tex) on 2026-09-22
new_results = res[res.index(r"\section{Experiments and Results}"):]

out = src
# title / abstract / keywords
out = out.replace(re.search(r"\\title\{.*?\}", out, re.S).group(0), new_title, 1)
out = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: new_abstract, out, count=1, flags=re.S)
out = re.sub(r"\\begin\{keyword\}.*?\\end\{keyword\}", lambda m: new_keyword, out, count=1, flags=re.S)
# section 1
i, j = between(out, r"\section{Introduction}", r"\section{Related Work}")
out = out[:i] + new_intro + "\n" + out[j:]
# section 2: condensed related work (keeps every citation; includes the error-detection paragraph)
related = read("related_v2.tex")
i, j = between(out, r"\section{Related Work}", r"\section{Methodology}")
out = out[:i] + related[related.index(r"\section{Related Work}"):] + "\n\n" + out[j:]
# section 3 (complete) and section 4
i, j = between(out, r"\section{Methodology}", r"\section{Experiments}")
out = out[:i] + new_meth + "\n\n" + out[j:]
i, j = between(out, r"\section{Experiments}", r"\section{Results and Discussion}")
out = out[:i] + new_exp + out[j:]
# section 5, keeping Table 2
tab2_i, tab2_j = between(src, r"\begin{table}[H]" + "\n\\centering\n\\caption{Classification performance for BERT teacher", r"\end{table}")
table2 = src[tab2_i:tab2_j] + r"\end{table}"
new_results = new_results.replace("%% Table 2 (tab:performance_comparison) unchanged.", table2)
i, j = between(out, r"\section{Results and Discussion}", r"\section{Conclusion and Future Work}")
out = out[:i] + new_results + "\n\n" + out[j:]
# section 6
i, j = between(out, r"\section{Conclusion and Future Work}", r"\section*{CRediT authorship contribution statement}")
out = out[:i] + concl + "\n\n" + out[j:]
# figure environments 3-5 (unchanged) re-inserted from main.tex at the markers of results_v2.tex
def figure_block(label):
    k = src.index("\\label{" + label + "}")
    i = src.rindex("\\begin{figure}", 0, k); j = src.index("\\end{figure}", k) + len("\\end{figure}")
    return src[i:j]
for lab in ("fig:insertion_auc", "fig:deletion_auc", "fig:progression_combined"):
    marker = "%%FIGURE_BLOCK:" + lab + "%%"
    assert out.count(marker) <= 1, marker
    if marker in out:
        out = out.replace(marker, figure_block(lab))
# figures live in figures/
for f in ("Figure_3.png", "Figure_4.png", "Figure_5a.png", "Figure_5b.png"):
    out = out.replace("{" + f + "}", "{figures/" + f + "}")
# figure captions 3-5
out = out.replace("Error and correctness detection rates plotted across insertion AUC thresholds, indicating the mean per dataset.",
                  "Error detection rate (errors with insertion AUC below the threshold) and correctness retention rate (correct predictions at or above it) across insertion-AUC thresholds, per explainer, averaged over graph types; gold-label errors.")
out = out.replace("Error and correctness detection rates plotted across deletion AUC thresholds, indicating the mean per dataset.",
                  "Error detection rate and correctness retention rate across deletion-AUC thresholds, per explainer, averaged over graph types; gold-label errors.")
out = out.replace("Progression drop concentration for the top k features across the evaluated datasets and explainability modules.",
                  "Mean cumulative confidence drop when the top-\\(k\\) elements are removed (a) or when only the top-\\(k\\) elements are kept (b), for correct and incorrect predictions, per explainer, averaged over graph types.")
out = out.replace("Maskout Progression drop concentration.", "Mask-out progression.").replace("Sufficiency Progression drop concentration.", "Sufficiency progression.")
# bibliography files
out = out.replace(r"\setlength{\bibsep}{2pt plus 0.5pt minus 0.5pt}", r"\setlength{\bibsep}{0pt}")  # 35-page limit: gap between entries, not line spacing
assert out.count(r"\setlength{\bibsep}{0pt}") == 1
out = out.replace(r"\bibliography{references}", r"\bibliography{references,references_additions}")
# AI declaration placeholder
i, j = between(out, "During the preparation of this work, no generative AI", r"\bibliographystyle{elsarticle-num}")
out = out[:i] + "During the preparation of this work the authors used Claude Code (Anthropic) in order to audit the analysis code, to write the scripts that generate the tables and figures from the experimental results, and to draft and edit the text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.\n\n\n" + out[j:]
# Table 2: surrogates retrained with node features from the fine-tuned teacher (2026-09-19); weighted P/R/F1
tab2_rows = {
    'AG News': {'Syntactic': ('.982', '.982', '.982', '.936', '.937', '.936'), 'Constituency': ('.982', '.982', '.982', '.935', '.935', '.935'),
                'Skip-grams': ('.982', '.982', '.982', '.937', '.937', '.937'), 'Window': ('.983', '.983', '.983', '.937', '.937', '.937')},
    'SST-2': {'Syntactic': ('.969', '.969', '.969', '.904', '.904', '.904'), 'Constituency': ('.981', '.981', '.981', '.911', '.911', '.911'),
              'Skip-grams': ('.978', '.978', '.978', '.911', '.911', '.911'), 'Window': ('.978', '.979', '.978', '.913', '.913', '.913')},
}
i0 = out.index(r"\caption{Classification performance for BERT teacher"); i1 = out.index(r"\end{table}", i0)
block = out[i0:i1]; j = block.index(r"\textbf{SST-2}")
parts = [block[:j], block[j:]]
for part_i, ds in enumerate(('AG News', 'SST-2')):
    for label, vals in tab2_rows[ds].items():
        pat = re.compile(r"(& \\textbf\{" + re.escape(label) + r"\}\s*&)[^\\]*\\\\")
        parts[part_i], n = pat.subn(lambda m: m.group(1) + ' ' + ' & '.join(vals) + r' \\', parts[part_i])
        assert n == 1, (ds, label, n)
out = out[:i0] + ''.join(parts) + out[i1:]
# Table 2: the two side-by-side minipages overflow in the double-spaced review layout; scale each tabular to its minipage
i0 = out.index(r"\caption{Classification performance for BERT teacher"); i1 = out.index(r"\end{table}", i0)
blk = out[i0:i1].replace(r"\begin{tabular}{@{}llcccccc@{}}", r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}llcccccc@{}}").replace(r"\end{tabular}", r"\end{tabular}}")
assert blk.count(r"\resizebox") == 2
out = out[:i0] + blk + out[i1:]
# tables and figures regenerated from the fine-tuned-feature run
out = out.replace(r"\input{tables/", r"\input{tables_ft/").replace("{figures/Figure_", "{figures_ft/Figure_")
# Times (elsarticle [times] + \\usepackage{times}) needs the T1 encoding to resolve its bold and italic shapes; without it XeTeX/tectonic falls back to Latin Modern regular (no bold)
out = out.replace("\\usepackage[utf8]{inputenc}", "\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}", 1)
# funding identifier: allow line breaks (the \url form overflows the margin)
out = out.replace(r"\url{MCIN/AEI/10.13039/501100011033}", r"MCIN\slash AEI\slash 10.13039\slash 501100011033")
out = out.replace("(2022/TL22/00215334)", r"(2022\slash TL22\slash 00215334)").replace("CIPROM/2021/021", r"CIPROM\slash 2021\slash 021")
out = out.replace("This work was supported by the University of Alicante", r"\begin{sloppypar}This work was supported by the University of Alicante").replace(r"CIPROM\slash 2021\slash 021.", r"CIPROM\slash 2021\slash 021.\end{sloppypar}")
# 35-page limit: single spacing inside tables and figures only; body text keeps the 'review' spacing
out = out.replace(r"\begin{document}", "\\usepackage{etoolbox}\n\\AtBeginEnvironment{table}{\\linespread{1}\\selectfont}\n\\AtBeginEnvironment{figure}{\\linespread{1}\\selectfont}\n\n\\begin{document}", 1)
(HERE / "main_v2.tex").write_text(out, encoding="utf-8")
print("main_v2.tex written:", len(out.splitlines()), "lines")
