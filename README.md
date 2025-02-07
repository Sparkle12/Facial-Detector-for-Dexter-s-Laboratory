# Facial Detector for Dexter's Laboratory
  Built a facial detector algorithm to detect characters from the popular cartoon Dexter's Laboratory. The system can detect all the faces in the image, as well as the faces of certain characters like Dexter, DeeDee, Dexter's Mom and his Dad. The algorithm is based on a multiscale sliding window approach that uses CNNs to classify it as a face or character. The CNNs were trained using examples from all the scales.
  For more information on how the algorithm works please refer to the pdf inside the repo.
## Results
  To quantify the performance of the algorithm I used mean average precision. For the detection of all the faces in the image I obtained a mAP of 0.926 and for the detections of the 4 characters I obtained a mAP of 0.9595.
