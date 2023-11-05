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
	KITEST=1 coverage run -m pytest -W ignore::DeprecationWarning -vv -s tests/
	./tests/test_subtree.sh
unit:
	KITEST=1 coverage run -m pytest -W ignore::DeprecationWarning -vv -s tests/test_ki.py tests/test_parser.py
integration:
	KITEST=1 coverage run -m pytest -W ignore::DeprecationWarning -vv -s tests/test_integration.py
report:
	coverage html
isolate:
	sed -z -i 's/def test/@pytest.mark.skip\ndef test/g' tests/*.py
unisolate:
	sed -z -i 's/@pytest.mark.skip\n//g' tests/*.py
count:
	find ki/*.py -type f -exec python3 -m tokenize {} \; | wc -l
