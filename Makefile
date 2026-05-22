.PHONY: all verify figures pdf clean

all: verify figures pdf

verify:
	python3 verify_claims.py -v
	python3 regenerate_tables.py --verbose

figures:
	python3 generate_figures.py
	python3 analyze_kfac_prediction.py

pdf:
	cd paper && pdflatex -interaction=nonstopmode spectral_alignment.tex
	cd paper && pdflatex -interaction=nonstopmode spectral_alignment.tex

clean:
	cd paper && rm -f spectral_alignment.{aux,bbl,blg,log,out}
