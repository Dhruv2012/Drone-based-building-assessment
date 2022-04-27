# Template Matching module

Template matching module is used to account for model's failure. \
-> However the only condition here is the model should be able to detect atleast 1 window in each horizontal patch, after which the template matching module will take the detected window(from model) as template and run it over the entire horizontal patch to find the remaining undetected windows. \
-> Non-maxima suppression module is also employed to remove the multiple instances giving a unique detection for each window in the frame.


Inferred Images(from trained models on zju_facade+iiitH dataset) which need templateMatching
Vindhya_002_000320(resnet, mobilenet)
Bakul_003_000470(resnet, mobilenet)
Bakul_002_000970(resnet, shuflenet)
Bakul_002_000920(resnet)
