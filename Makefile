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
	coverage run -m pytest tests/
report:
	coverage html
