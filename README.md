# Video-matting refinement

- Library (use pip to install)

  + scikit-image
  + numpy
  + av
  + matplotlib

- Run

  Static background

  `python path_to_video.mp4`

  Moving background

  Change parameter 'motion' to True in line 43 of main.py, then run

  `python path_to_video.mp4`


- Data structure

  Put path_to_video.mp4 and path_to_video_matte.mp4 in the same folder. The matte video should be output of MODNet (or other approximate matte to be refined) If the refinement runs successfully, output video will be saved in the same folder. (if motion=True, no error report will be print in log and the err video will be empty)

