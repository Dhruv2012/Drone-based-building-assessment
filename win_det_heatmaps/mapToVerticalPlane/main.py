from mapToVerticalPlane import *

if __name__ == '__main__':
    coord_dir = "../sample_seq_data"

    imgPath = "../sample_seq_data/001_new/images"
    coordFilePath = os.path.join(coord_dir, 'coordinatesFromPostProcessing-1_new-shufflenet.csv')

    ##  TESTING ONLY SINGLE SEQUENCE
    imgPath2 = "../sample_seq_data/002_new/images"
    coordFilePath2 = os.path.join(coord_dir, 'coordinatesFromPostProcessing-2_new-shufflenet.csv')

    imgPath3 = "../sample_seq_data/003_new/images"
    coordFilePath3 = os.path.join(coord_dir, 'coordinatesFromPostProcessing-3_new-shufflenet.csv')

    verticalPlane = np.zeros((1500,10000,3),np.uint8)

    depth = 750 ## Depth to the building(in cm)

    offset_ramp_001 = 15 #cm for seq 001
    offset_ramp_002 = 81 #cm for seq 002
    offset_ramp_003 = 0 #cm for seq 003

    mapToVerticalPlane = MapToVerticalPlane(focalLength = 920)

    seq_num = 1
    final_building_boxes, windowCount1, storeyCount1, avgStoreyHeights1 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath, coordFilePath, depth, offset_ramp_001, seq_num)
    seq_num = 2
    final_building_boxes, windowCount2, storeyCount2, avgStoreyHeights2 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath2, coordFilePath2, depth, offset_ramp_002, seq_num)
    seq_num = 3
    final_building_boxes, windowCount3, storeyCount3, avgStoreyHeights3 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath3, coordFilePath3, depth, offset_ramp_003, seq_num)
    
    mapToVerticalPlane.plotBoxes(np.copy(verticalPlane), final_building_boxes)

    print('Total window count of all seqs:', str(mapToVerticalPlane.windowCount))