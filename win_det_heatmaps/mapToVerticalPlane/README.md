## Mapping Windows to Vertical Plane
### Sample Data
-> Sample data of 3 seqs is provided in [sample_seq_data](https://github.com/Dhruv2012/Drone-based-building-assessment/tree/resolveNMS/win_det_heatmaps/sample_seq_data). It contains images, log files and CSV File containing window coordinates obtained from post-processing module.

### How to run?

* **mapToVerticalPlane.ipynb**: \
This notebook can be used for more of a visualization purpose as to how the module gets the final mapped windows on the plane along with the intermediate results. It maps windows from each image to a vertical plane and finds window/storey count and gets a rough estimate of avg. height difference between consecutive storeys in a building(which then can be further used to scale up 3D model of the building).

    -> Takes sample seq data(3 seqs) of Bakul building from [sample_seq_data](https://github.com/Dhruv2012/Drone-based-building-assessment/tree/resolveNMS/win_det_heatmaps/sample_seq_data) and maps them onto imaginary vertical plane. \
    -> In case, one wants to try out a new seq, enter input params after cell "**Initialize Variables Here**".

* **mapToVerticalPlane.py**: \
A python class for mapping windows to Vertical Plane. The code snippet below shows how to use the class.


    ```
    # focal Length = 920 in pixels
    mapToVerticalPlane = MapToVerticalPlane(focalLength = 920)

    # Imaginary vertical plane for mapping     
    verticalPlane = np.zeros((1500,10000,3),np.uint8)

    # Paths to img seq and coord CSV file
    coord_dir = "../sample_seq_data"
    imgPath = "../sample_seq_data/001_new/images"
    coordFilePath = os.path.join(coord_dir, 'coordinatesFromPostProcessing-1_new-shufflenet.csv')

    depth = 750 ## Depth to the building(in cm)

    # Needed if UAV takes off from some height above the ground
    offset_ramp_001 = 15 #cm for seq 001

    seq_num = 1
    final_building_boxes, windowCount1, storeyCount1, avgStoreyHeights1 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath, coordFilePath, depth, offset_ramp_001, seq_num)

    # Plotting final mapped Boxes
    mapToVerticalPlane.plotBoxes(np.copy(verticalPlane), final_building_boxes)

    ```

    -> Directly run the [main.py](https://github.com/Dhruv2012/Drone-based-building-assessment/tree/resolveNMS/win_det_heatmaps/mapToVerticalPlane/main.py) to visualize results of 3 sample sequences.