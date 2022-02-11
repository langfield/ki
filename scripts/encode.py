with open("example.py", "r", encoding="UTF-8") as src_file:
    src: str = src_file.read()
    print(src.encode())
