PYTHON ?= /home/eva/code/ma6216/industrial-visual-anomaly-detection/.venv/bin/python

.PHONY: all check tables figures clean
all:
	command -v latexmk >/dev/null || (echo "latexmk not on PATH" && false)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check: all
	@$(PYTHON) -c 'import re, sys; text=open("main.log").read(); bad=[line for line in text.splitlines() if re.search(r"undefined|Citation .* undefined", line, re.I) or (re.search(r"Overfull \\hbox \(([\d.]+)pt too wide\)", line) and float(re.search(r"Overfull \\hbox \(([\d.]+)pt too wide\)", line).group(1)) > 5)]; print("\n".join(bad) if bad else "PASS: no undefined references/citations or overfull hboxes > 5pt"); sys.exit(bool(bad))'

tables:
	$(PYTHON) ../scripts/paper_tables.py

figures:
	MPLCONFIGDIR=/tmp/ma6216-paper-matplotlib $(PYTHON) ../scripts/paper_figures.py

clean:
	latexmk -C
