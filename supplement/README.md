# Online Supplementary Material

**Paper**: Edge-Intelligent Video Analytics: A Systematic Survey of Architectures, Applications, and Deployment Strategies Across Smart City Verticals  
**Journal**: ACM Computing Surveys (CSUR)  
**Authors**: Mukhiddin Toshpulatov, Wookey Lee, Jinsoo Cho

---

## Contents

| File | Description |
|------|-------------|
| `appendix_benchmark_methodology.tex` | Full EdgeVA benchmarking methodology: hardware specs, software versions, measurement protocol, pipeline-timing breakdown, confidence intervals, reproducibility checklist |

## How to compile

```bash
cd supplement
pdflatex appendix_benchmark_methodology.tex
bibtex appendix_benchmark_methodology
pdflatex appendix_benchmark_methodology.tex
pdflatex appendix_benchmark_methodology.tex
```

Requires the parent `AntVision_refs.bib` and `csur_extra.bib` files,
or update the `\bibliography{}` path to point to your local copies.

## Relation to main paper

The appendix documents all measurement protocols supporting:
- **Table 2** (CPU vs GPU inference benchmarks, same-machine)
- **Figure 5** (pipeline block diagram with per-stage latencies)
- **Section 6.4** (end-to-end pipeline timing)

Benchmark scripts, raw CSVs, and model exports are in `../experiments/`.
