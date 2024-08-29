import os
import argparse
import jsonlines

def parse_arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infilepath", type=str, default="")
    parser.add_argument("--outfilepath", type=str, default="")
    return parser.parse_args()

args = parse_arg_main()

def jsonl2txt(infilepath, outfilepath):
    with open(infilepath, "r") as f_in, open(outfilepath, "w") as f_out:
        for item in jsonlines.Reader(f_in):
            item["stego"] = item["stego"].replace(" ", "<br /><br />")
            item["stego"] = item["stego"].replace(" ", "<br />")
            f_out.write(item["stego"] + "\n")


if __name__ == '__main__':
    jsonl2txt(args.infilepath, args.outfilepath)
