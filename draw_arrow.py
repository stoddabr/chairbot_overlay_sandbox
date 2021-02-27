import cv2
import numpy as np
import math 

def drawArrow(img, coords, orientation, color=(255,0,0), thickness = 10):
  """ draw an arrow overlay on the image at the coordinates """

  x, y = coords
  x = int(x)
  y = int(y)
  # points for an arrow centered at coords pointing right (zero degree angle)
  delta = 50  # TODO tune for full sized image
  left_pt = np.array([x,y-delta])
  right_pt = np.array([x,y+delta])
  tip_pt = np.array([x+delta,y])  

  # TODO rotate based on orientation
  # https://stackoverflow.com/questions/7953316/rotate-a-point-around-a-point-with-opencv 

  triangle_cnt = np.array( [left_pt, right_pt, tip_pt] )

  # Using cv2.polylines() method 
  # Draw a Blue polygon with  
  # thickness of 1 px 
  isClosed = True
  return cv2.drawContours(img, [triangle_cnt], 0, color, thickness)


def _test_on_blank_image():
  # execute only if run as a script
  height = 100
  width = 100
  blank_image = np.zeros((height,width,3), np.uint8)
  
  arrow_image = drawArrow(blank_image, (height/2, width/2), 30)
  cv2.imshow('image', arrow_image)
  cv2.waitKey()


def _test_on_real_image():
  # execute only if run as a script
  image = cv2.imread('images/chairbot_noAR_1.png')
  height, width, _ = image.shape
  
  arrow_image = drawArrow(image, (height/2, width/2), 30)
  cv2.imshow('image', arrow_image)
  cv2.waitKey()


def _find_chairbots(image):
  """ returns the chairbot location and orientation in an image """
  # detectMarkers returns: (corners, ids, rejectedImgPoints)

  dictionary = cv2.aruco.getPredefinedDictionary( cv2.aruco.DICT_4X4_50 )
  corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
  found_chairbots = []
  # Checks all fiducials in ditionary
  if len(corners) > 0:
    # fids corners, findex fiducial index
    for (fids, index) in zip(corners, ids):
      for corner in fids: # pt => point number
        try:
          fid = int(index[0]) # defined fiducial id number
          # exclude fiducial ids outside of expected range
          if (fid >= 0 and fid <= 5):  # chairbot max number is 5
            # ll contains (x, y) coordinate of the middle of fiducial
            midcords = (corner[0] + corner[1] +corner[2] +corner[3]) \
                /4 # average sum of 4 corners

            # calculate angle from origin to fiducial center
            # average of the top two fiducial corners
            topcords = (corner[0] + corner[3]) / 2
            # average of the bottome two fiducial corners
            botcords = (corner[1] + corner[2]) / 2
            # Difference between top and bottom
            ydeltacords = topcords - botcords
            # Tangent of the y and the x
            theta = math.atan2(ydeltacords[1], ydeltacords[0])
            # Changes theta from radians to positive degrees (0 to 360 rotating counter-clockwise)
            degree = theta * (180 / math.pi) + 180
            found_chairbots.append([midcords, degree ])
        except IndexError:
          print('IndexError thrown and passed while looping through aruco markers')
          pass
  return found_chairbots

def _test_on_chairbots():
  # execute only if run as a script
  image = cv2.imread('images/chairbot_noAR_1.png')
  height, width, _ = image.shape
  chairs = _find_chairbots(image)
  print('chairs found: ',chairs)

  for chair in chairs:
    chair_x, chair_y = chair[0]
    chair_angle = chair[1]
    arrow_image = drawArrow(image, (chair_x, chair_y), chair_angle)
  cv2.imshow('image', arrow_image)
  cv2.waitKey()





if __name__ == "__main__":
  _test_on_chairbots()


