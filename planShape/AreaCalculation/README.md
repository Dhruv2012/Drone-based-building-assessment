## Area Calculation

input_to_findArea: Contains sample images(masks) of some buildings\
find_area.py:  Calculates the area from the building mask obtained by semantic segmentation\
  -> Input: 
  building mask, depth(D) to the building and Focal length(f in pixels)\
  -> First calculates contour area from the mask in pixels\
  -> Then scales up by (D/f)^2

Assumption:\
The images are captured orthogonally from UAV in order to avoid any perspective correction.

