#!/usr/bin/env python3
"""
Generate LaTeX table (.tex) files from the summary CSV files in results/summary/.
This ensures that the empty stub files are fully populated and synchronized with
the ground-truth experimental data.
"""
import pandas as pd
from pathlib import Path

SUMMARY_DIR = Path("results/summary")

def escape_latex(val):
    if not isinstance(val, str):
        return val
    # Map Unicode markers to standard LaTeX alternatives or text
    val = val.replace("✓", "\\checkmark")
    val = val.replace("✗", "$\\times$")
    val = val.replace("⚠", "$\\triangle$")
    val = val.replace("—", "---")
    val = val.replace("∞", "$\\infty$")
    val = val.replace("%", "\\%")
    return val

def generate_table1():
    csv_path = SUMMARY_DIR / "table1_main_results.csv"
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Main Experimental Results of Centralized vs. Federated Learning Configurations on MIMIC-IV}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{AUROC} & \\textbf{Brier Score} & \\textbf{ECE} & \\textbf{Recall} & \\textbf{Precision} & \\textbf{Clinical Status} \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        model = row["Model"]
        auroc = f"{row['AUROC']:.4f}"
        brier = f"{row['Brier Score']:.4f}" if isinstance(row['Brier Score'], (int, float)) and not pd.isna(row['Brier Score']) else escape_latex(str(row['Brier Score']))
        ece = f"{row['ECE']:.4f}" if isinstance(row['ECE'], (int, float)) and not pd.isna(row['ECE']) else escape_latex(str(row['ECE']))
        recall = f"{row['Recall']*100:.1f}\\%" if isinstance(row['Recall'], (int, float)) else escape_latex(str(row['Recall']))
        precision = f"{row['Precision']*100:.1f}\\%" if isinstance(row['Precision'], (int, float)) else escape_latex(str(row['Precision']))
        status = escape_latex(row["Clinical Status"])
        
        tex.append(f"{model} & {auroc} & {brier} & {ece} & {recall} & {precision} & {status} \\\\")
        
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table1_main_results.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def generate_table2():
    csv_path = SUMMARY_DIR / "table2_scalability.csv"
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Federated Learning Scalability Across Expanding ICU Client Network}",
        "\\label{tab:scalability}",
        "\\begin{tabular}{rcccc}",
        "\\toprule",
        "\\textbf{\\# Clients} & \\textbf{AUROC} & \\textbf{Training Time (s)} & \\textbf{Throughput (samples/sec)} & \\textbf{AUROC Loss (\\%)} \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        clients = int(row["# Clients"])
        auroc = f"{row['AUROC']:.4f}"
        time = f"{row['Training Time (s)']:.2f}"
        throughput = f"{row['Throughput (samples/sec)']:.1f}"
        loss = f"{row['AUROC Loss (%)']:.3f}\\%"
        
        tex.append(f"{clients} & {auroc} & {time} & {throughput} & {loss} \\\\")
        
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table2_scalability.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def generate_table3():
    csv_path = SUMMARY_DIR / "table3_aggregation.csv"
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Comparison of Federated Aggregation Strategies on Test AUROC}",
        "\\label{tab:aggregation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Aggregation Strategy} & \\textbf{Test AUROC} & \\textbf{AUROC Loss (\\%)} \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        strategy = row["Aggregation Strategy"]
        auroc = f"{row['Test AUROC']:.4f}"
        loss = f"{row['AUROC Loss (%)']:.2f}\\%"
        
        tex.append(f"{strategy} & {auroc} & {loss} \\\\" )
        
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table3_aggregation.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def generate_table4():
    # Update table4 to use the 30-seed trials we just ran!
    csv_path = Path("results/trials/PHASE5_STATISTICAL_SUMMARY.csv")
    if not csv_path.exists():
        csv_path = SUMMARY_DIR / "table4_statistical.csv"
    
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Statistical Validation Across Independent Trials}",
        "\\label{tab:statistical_validation}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{N} & \\textbf{AUROC (Mean)} & \\textbf{AUROC (SEM)} & \\textbf{Recall (Mean)} & \\textbf{Recall (SEM)} \\\\",
        "\\midrule"
    ]
    
    # Check if we are using the new PHASE5_STATISTICAL_SUMMARY format
    if "AUROC_mean" in df.columns:
        for _, row in df.iterrows():
            config = escape_latex(row["Configuration"])
            n = int(row["N"])
            auroc_mean = f"{row['AUROC_mean']:.4f}"
            auroc_sem = f"{row['AUROC_sem']:.4f}"
            recall_mean = f"{row['Recall_mean']*100:.2f}\\%"
            recall_sem = f"{row['Recall_sem']*100:.2f}\\%"
            tex.append(f"{config} & {n} & {auroc_mean} & {auroc_sem} & {recall_mean} & {recall_sem} \\\\")
    else:
        # Fallback to the original table4_statistical format
        for _, row in df.iterrows():
            strategy = escape_latex(row["Strategy"])
            runs = int(row["# Runs"])
            mean = f"{row['Mean AUROC']:.4f}"
            std = f"{row['Std Dev']:.4f}"
            low = f"{row['95% CI (Low)']:.4f}"
            high = f"{row['95% CI (High)']:.4f}"
            tex.append(f"{strategy} & {runs} & {mean} & {std} & {low} & {high} \\\\")
            
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table4_statistical.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def generate_table5():
    csv_path = SUMMARY_DIR / "table5_privacy.csv"
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Privacy Budget (\\$\\varepsilon\\$) vs. Model Utility and Calibration Trade-offs}",
        "\\label{tab:privacy}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Privacy Budget (\\$\\varepsilon\\$)} & \\textbf{AUROC} & \\textbf{Recall} & \\textbf{Brier Score} & \\textbf{Clinical Viability} \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        eps = escape_latex(str(row["Privacy Budget (ε)"]))
        auroc = f"{row['AUROC']:.3f}" if isinstance(row['AUROC'], (int, float)) else escape_latex(str(row['AUROC']))
        recall = f"{row['Recall']*100:.1f}\\%" if isinstance(row['Recall'], (int, float)) else escape_latex(str(row['Recall']))
        brier = f"{row['Brier Score']:.3f}" if isinstance(row['Brier Score'], (int, float)) else escape_latex(str(row['Brier Score']))
        viability = escape_latex(row["Clinical Viability"])
        
        tex.append(f"{eps} & {auroc} & {recall} & {brier} & {viability} \\\\")
        
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table5_privacy.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def generate_table6():
    csv_path = SUMMARY_DIR / "table6_robustness.csv"
    df = pd.read_csv(csv_path)
    
    tex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Model Robustness under Byzantine Client Failure Scenarios}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Attack Scenario} & \\textbf{Byzantine Clients} & \\textbf{Fraction} & \\textbf{AUROC} & \\textbf{Recall} & \\textbf{Status} \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        scenario = escape_latex(row["Attack Scenario"])
        clients = int(row["Byzantine Clients"])
        fraction = escape_latex(row["Fraction"])
        auroc = f"{row['AUROC']:.4f}"
        recall = f"{row['Recall']*100:.1f}\\%" if isinstance(row['Recall'], (int, float)) else escape_latex(str(row['Recall']))
        status = escape_latex(row["Status"])
        
        tex.append(f"{scenario} & {clients} & {fraction} & {auroc} & {recall} & {status} \\\\")
        
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_path = SUMMARY_DIR / "table6_robustness.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Generated: {out_path}")

def main():
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()
    generate_table6()

if __name__ == "__main__":
    main()
