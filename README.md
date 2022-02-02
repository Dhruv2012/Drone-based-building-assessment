# Drone-based-building-assessment
### Readme Update(In Progress)

### Project - "Identification of salient structuaral elements from buildings using UAVs"
The project aims to extract information such as planShape/Area, Storey and window count, their height and so on of the buildings from the camera feed through drone. It is a part of IHub - Mobility Research at IIIT Hyderabad.

#### Objectives of Pilot Study:
To identify salient structural elements in buildings from RGB images captured using a UAV.
1) Number of windows
2) Number of storeys
3) Storey height (Uniform/varied storey heights)
4) Building Plan estimation

#### Dataset:
We have made our own custom dataset by capturing buildings on IIIT-H campus through a drone. In addition, we have also used the open-source zju_facade dataset to train our models.

<p>
  <img src="readme_images/Dataset.drawio.png" width="45%" height="90%"/>     
</p>
For window detection task (The ground truths are bounding boxes shown in red)

<br />
<br />
<p>
  <img src="readme_images/RoofTopDataset.png" width="45%" height="90%"/>
</p>

For plan-shape/area (The ground truths are segmented masks in white)

### Progress till now:
Please find our current progress in the following presentation.
<a href="https://docs.google.com/presentation/d/1SGP0-3pb7mIS0CNtp-2dJ66QOJx7wwM2IqhtbE_4YXs/edit?usp=sharing" target="_blank">Presentation Link</a>


### Project Team:
Dhruv Patel - Project Associate, Robotics Research Centre(RRC), IIIT Hyderabad \
Shivani Chepuri - MS Student, IIIT Hyderabad \
Sarvesh Thakur - MEng Robotics, University of Maryland 

Advisors: \
Prof. Madhava Krishna (Head, RRC, IIIT Hyderabad) \
Dr. Harikumar Kandath (Assistant Professor, IIIT-Hyderabad) \
Dr. Ravi Kiran Sarvadevabhatla (Assistant Professor, IIIT-Hyderabad)
