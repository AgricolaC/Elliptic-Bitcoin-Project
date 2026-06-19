RCSFLAGS= -l
MAIN=paper-graphnet
TEXPARTS= \
	000abstract.tex		\
	010intro.tex		\
	020related.tex		\
	030proposed.tex	\
	040experiments.tex	\
	050discussion.tex	\
	060conclusion.tex	\
	070appendix.tex

BIBPARTS= *.bib
LPR = pspr # or change to 'qpr -q ps3'
POSTSCRIPT= dvips -P cmz -t letter  #dvips -f
FIGPARTS = FIG

$(MAIN).pdf: $(MAIN).tex $(TEXPARTS)
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

try: $(MAIN).tex
	pdflatex $(MAIN)

print: $(MAIN).ps
	$(LPR) $(MAIN).ps

checkout:
	cvs checkout $(RCSFLAGS) $(MAIN).tex $(TEXPARTS)

checkin:
	cvs checkin $(RCSFLAGS) -f $(MAIN).tex $(TEXPARTS)

clean:
	\rm -f *.dvi $(MAIN).ps *.bbl *.aux *.blg *.fdb_latexmk *.fls *.synctex.gz *.log *.blg *.toc all.tar* uu *~ *.bak

spotless: clean
	\rm -f $(MAIN).ps

all.tar:
	tar cvfh all.tar makefile $(MAIN).tex $(TEXPARTS) $(BIBPARTS) \
	$(FIGPARTS) README

uu: all.tar
	gzip all.tar
	uuencode all.tar.gz all.tar.gz > uu

bundle.all: $(TEXPARTS) 
	bundle  README $(TEXPARTS) $(BIBPARTS) makefile > bundle.all

zip:
	cmd /C "pkzip $(MAIN).zip -P $(MAIN).pdf Makefile *.cls *.tex *.bib FIG/*.eps"
