"""Collect licensed, authentic-color historical Vietnam still photographs."""
import sys

from collect_wikimedia_vietnam import main


if __name__ == "__main__":
    main(["--historical", *sys.argv[1:]])
