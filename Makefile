install:
	python3 -m pip install --user -e .
	echo 'export PATH=$$HOME/.local/bin:$$PATH' >> ~/.bashrc
	. ~/.bashrc