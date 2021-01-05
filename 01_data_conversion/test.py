import struct
import argparse

def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    args = parser.parse_args()

    fb=open(args.file,'rb')
    while True:
                # read each line of the file
                record=fb.read(696)

                # break when the line was empty
                if(len(record) < 696): break

                # unpack the line
                line = list(struct.unpack('=87d',record))
                print(line[-5:])

if __name__ == '__main__':
	main()
