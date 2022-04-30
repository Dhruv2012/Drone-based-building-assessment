# Post-processing module
A python class integrating Template matching and Non-maxima suppression employed to account for the failure cases of window detection model.
This class is called during model inference ([infer.py](../infer.py)), where it generates a CSV file containing window coordinates for the entire vertical sequence.

<p align="center"><img src="../../readme_images/newtm.drawio.png" width=500%" height="70%"/></p>
<p align = "center">Model inference(left), Horizontal Patch
(middle), Template (right)</p>

<p align="center"><img src="../../readme_images/newio.drawio.png" width=500%" height="70%"/></p>
<p align = "center">Post processing results</p>

As shown in the fig. above, the post processing module detects all windows successfully.