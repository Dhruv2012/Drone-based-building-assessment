import argparse
from ground_normal import GroundNormal

parser = argparse.ArgumentParser("Distance between adjacent buildings")
parser.add_argument("-i", "--imagestxt", help="images.txt file generated from colmap", required=True, type=str)
parser.add_argument("-s", "--segmented", help="path to segmented ground plane images", required=True, type=str)
parser.add_argument("-p", "--points3d", help="points3d.txt file generated from colmap", required=True, type=str)

def main():
    global args
    args = parser.parse_args()
    ground_normal = GroundNormal(args.imagestxt, args.points3d, args.segmented)
    # print(args.imagestxt, args.segmented, args.points3d)

    print(ground_normal.GroundNormalCoefficients())

if __name__ == "__main__":
    main()