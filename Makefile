default:
	echo "No-op."
install:
	pip install -r requirements.txt
documentation:
	pdoc --html ki --template-dir docs/template/ --force
	cp docs/u1F367-shavedice.svg html/ki/
clean:
	rm -rf html/
