# Build the paper. Requires `latexmk` + `pdflatex` + `bibtex` (TeX Live).
# On macOS:
#   brew install --cask basictex   # ~100 MB, includes latexmk and bibtex
#   sudo tlmgr update --self && sudo tlmgr install booktabs natbib hyperref \
#                                              microtype titlesec geometry
# Or upload paper/ to Overleaf to compile online without installing TeX Live.

MAIN = main
TEX  = $(MAIN).tex

.PHONY: all clean watch
all:
	latexmk -pdf -bibtex -interaction=nonstopmode $(TEX)

watch:
	latexmk -pdf -bibtex -pvc -interaction=nonstopmode $(TEX)

clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc \
	      sections/*.aux
