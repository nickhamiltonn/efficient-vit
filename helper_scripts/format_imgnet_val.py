import os

FILE_PREFIX = "ILSVRC2012_val_"
FILE_POSTFIX = ".JPEG"
VAL_LABELS_FILE = "imagenet_2012_validation_synset_labels.txt"

def main():
    with open(VAL_LABELS_FILE) as fp:
        file_index = 1
        for line in fp:
            curr_file = FILE_PREFIX + f"{file_index:08d}" + FILE_POSTFIX
            if not os.path.exists(f"imagenet/val/{curr_file}"):
                print(f"{curr_file}: Not found")
                raise FileNotFoundError()

            if not os.path.isdir(f"imagenet/val/{line}"):
                os.mkdir(f"imagenet/val/{line}")
            os.rename(f"imagenet/val/{curr_file}", f"imagenet/val/{line}/{curr_file}")
            
            file_index += 1
                    
    print("Completed")


if __name__ == "__main__":
    main()