# When Should a Collector Stop Buying Blind Packs?

In this project, I study the cost of collecting all eight designs in the **DEATH NOTE – My Dear Light** badge series. A collector can buy random packs, resell duplicates, and purchase missing designs on the secondary market. I compare stopping rules to understand the tradeoff between average cost and financial risk.

I start with a fixed number of draws, then consider a decision based on which designs are still missing. I use a Bellman recursion to check the dynamic rule, and simulation and sensitivity analysis to examine its risks and limitations.

## Report

The [full report](report/Report.pdf) explains the model development, derivations, results, and limitations. The code in this repository provides the numerical analysis and figure generation used alongside the report.

The [LaTeX source package](report/Report_LaTeX.zip) contains the report source, references, figures, and tables. To compile it, extract the package and use `main.tex` with **XeLaTeX** and **Biber**. The package can also be uploaded as a separate Overleaf project. Compiling the report is optional; running the Python analysis does not require LaTeX.

The report contains a fixed set of results and figures. Running the code creates files in `outputs/`; it does not automatically replace the figures inside the LaTeX package. If the parameters change, those new results should be treated separately from the report baseline.

## Run the project

Use Python 3.9 or later. Unzip the package, then open a terminal in the folder containing this README and `run_final.py`. For example, if the extracted folder is named `Blind_Draw_Model_English`:

```bash
cd Blind_Draw_Model_English
```

Install the three dependencies and run:

```bash
python -m pip install -r requirements.txt
python run_final.py
```

The script reads the Excel workbook, prepares the usable price records, and generates English figures and result tables in `outputs/`. The default run uses 100,000 trials per randomized strategy and 2,000 price-bootstrap resamples.

For a shorter run:

```bash
python run_final.py --quick
```

This uses 10,000 trials and 500 bootstrap resamples. Both modes use the same `outputs/` folder and overwrite files with the same names. Run the full mode last when preparing report figures. Each run also rebuilds the prepared files in `data/`, so do not edit those generated files directly.

On systems where Python is invoked as `python3`, replace `python` with `python3` in these commands. Installing dependencies normally requires an internet connection; the analysis itself runs offline once they are installed. No API key or LaTeX installation is needed. A full run may take several minutes. Successful completion prints the key results and `Output directory: outputs/`.

## Files

```text
README.md                       Project overview and instructions
DEATH_NOTE_My_Dear_Light.xlsx     Original English data workbook
run_final.py                    One-command analysis and plotting
prepare_data.py                 Workbook-to-model data preparation
requirements.txt                Python dependencies
report/Report.pdf               Full modeling report
report/Report_LaTeX.zip          LaTeX source with figures and references
src/gacha_model/                Model and plotting modules
data/                           Prepared price records and audit summary
```

The workbook is included unchanged. Its worksheet headings and data fields are in English. Original listing titles and source-language product names remain as recorded, so some source cells retain Chinese or Japanese text. They are not used as plot labels or program messages. The Python files, instructions, console messages, and generated results are in English.

`prepare_data.py` uses the recorded inclusion flags and numeric unit prices. Of 134 raw records, 125 are usable. It also prepares a separate 122-record check that excludes the same-design repeated-link pair MKT-068/MKT-075 and the unresolved date in MKT-126. It does not edit the workbook or silently repair that date.

## Main baseline results

| Strategy | Expected net cost (CNY) |
|---|---:|
| Secondary-market purchase only | 366.58 |
| Fixed two draws | 357.59 |
| Fixed three draws | 357.31 |
| Dynamic stopping | 356.72 |
| Draw until complete | 571.95 |

The dynamic rule has an expected stopping time of 2.619 draws and agrees with Bellman actions across all 256 missing-item states. What matters here is that its small average-cost advantage does not make it the lowest-risk choice. For fixed two draws, exact enumeration gives a 3.4375% chance of final net cost exceeding CNY 400, compared with 6.25% for purchase outlay before resale. I discuss this distinction in the report rather than choosing a strategy from its average cost alone.

## Parameters and interpretation

Manual settings are in `src/gacha_model/config.py`. The baseline uses equal draw probabilities, a CNY 39.8 draw price, and a 0.60 resale success probability. The eight purchase prices are fixed values copied from the workbook medians for this report. Reading or changing the workbook does not automatically replace these configuration prices. The workbook supplies the price observations for Bootstrap and the separate record-exclusion check. Draw and resale parameters are scenario assumptions, not measured transaction probabilities.

This is an eight-design case study. You can adjust draw cost, resale success, transaction costs and the budget reference in the configuration. General consistency checks still run after a change; checks against the report's known numerical values run only when the original economic inputs are used. Changing a parameter does not have to preserve a fixed optimum of three draws.

Keep the eight-design structure and equal baseline probabilities for the one-command analysis. The collect-all closed form requires equal probabilities; the separate probability-sensitivity module evaluates unequal probabilities for fixed and dynamic stopping. The fixed model searches counts from zero to `fixed_k_max` (15 by default). When changing prices substantially, check that the optimum has not reached this search limit. Fixed purchase prices and workbook observations describe the same snapshot by default; if changing the price assumptions, interpret them separately from the workbook-based checks.

The dynamic model minimizes expected net cost. It does not solve a hard cash-budget constraint. Listing prices are a convenience sample rather than completed transactions. Bootstrap describes variation within that sample and cannot remove selection bias or seller dependence. Source links and screenshot references in the workbook do not establish that every original listing or screenshot has been independently verified.

Each run exports 15 main figures and 11 appendix figures in PNG, SVG, and PDF, together with CSV tables, selected LaTeX tables, and a numerical summary. Outputs are excluded from Git by default because they can be regenerated. Numerical results were checked using NumPy 1.26.4 and Matplotlib 3.8.4; appearance can vary with software versions.
