default:
	echo "No-op."
install:
	pip install -e .
documentation:
	pdoc --html ki --template-dir docs/template/ --force
	cp html/ki/* docs/
	rm -rf html/
clean:
	rm -rf docs/*.html
	rm -rf *.egg-info
	rm -rf htmlcov/
test:
	KITEST=1 coverage run -m pytest -vv -s tests/
unit:
	KITEST=1 coverage run -m pytest -vv -s tests/test_ki.py tests/test_parser.py
integration:
	KITEST=1 coverage run -m pytest -vv -s tests/test_integration.py
report:
	coverage html
