default:
	echo "No-op."
install:
	pip install -r requirements.txt
documentation:
	pdoc --html ki --force
	cp docs/1f95e.svg html/ki/
