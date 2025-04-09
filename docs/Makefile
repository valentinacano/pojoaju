SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: clean html

clean:
	@echo "🧽 Borrando documentación generada..."
	rm -rf $(BUILDDIR)

html:
	@echo "📄 Generando documentación HTML..."
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
