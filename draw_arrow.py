import cv2
import numpy as np
import math
import path_planner

#color = Blue, Green, Red

def blend_images():
  alpha = 0.5

  # [load]
  src1 = cv2.imread('images/chairbot_noAR_4.png')
  src2 = cv2.imread('images/chairbot_noAR_4_adjust.png')
  # [blend_images]
  beta = (1.0 - alpha)
  im_blend = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
  # [display]
  cv2.imshow('im_blend', im_blend)
  cv2.waitKey()
  # [display]
  cv2.destroyAllWindows()

def draw_real_path(img, coord_one, coord_two, color, thickness = 10):
  x, y, z = coord_one
  x = int(x)
  y = int(y)
  z = int(z)
  a, b, c = coord_two
  a = int(a)
  b = int(b)
  c = int(c)
  cv2.line(img, (x, y), (a, b), color, thickness)


def red_circle(img, center, color=(0, 0, 255), thickness = 10):
  cv2.circle(img, center, 34, color, thickness)

def rotate_pts(x1, y1, orientation, origin):
  #pt (x0, y0) will always be the origin
  x0, y0 = origin

  orientation = (orientation - 180) / (180/math.pi)

  #angle a is the angle the pt will rotate around the orgin
  x2 = ((x1 - x0) * math.cos(orientation)) - ((y1 - y0) * math.sin(orientation)) + x0
  y2 = ((x1 - x0) * math.sin(orientation)) + ((y1 - y0) * math.cos(orientation)) + y0

  update1 = int(math.ceil(x2))
  update2 = int(math.ceil(y2))
  updated_coords = update1, update2

  return updated_coords

def drawArrow(img, coords, orientation, color=(218,224,64), thickness = 3, delta=50, offset=15):
  """ draw an arrow overlay on the image at the coordinates """

  x, y = coords
  x = int(x)
  y = int(y)

  # points for an arrow centered at coords pointing right (zero degree angle)
  left_x = x+offset
  left_y = y-delta
  right_x = x+offset
  right_y = y+delta
  tip_x = x+delta+offset
  tip_y = y

  origin = (x, y)

  left = rotate_pts(left_x, left_y, orientation, origin)
  right = rotate_pts(right_x, right_y, orientation, origin)
  tip = rotate_pts(tip_x, tip_y, orientation, origin)


  left_pt = np.array([left])
  right_pt = np.array([right])
  tip_pt = np.array([tip])

  triangle_cnt = np.array( [left_pt, right_pt, tip_pt] )

  # Using cv2.polylines() method
  # Draw a Blue polygon with
  # thickness of 1 px
  isClosed = True
  return cv2.drawContours(img, [triangle_cnt], -1, color, thickness)



def _test_on_blank_image():
  # execute only if run as a script
  height = 100
  width = 100
  blank_image = np.zeros((height,width,3), np.uint8)

  arrow_image = drawArrow(blank_image, (height/2, width/2), 30)
  #cv2.imshow('image', arrow_image)
  cv2.waitKey()


def _test_on_real_image():
  # execute only if run as a script
  image = cv2.imread('images/chairbot_noAR_4.png')
  height, width, _ = image.shape

  arrow_image = drawArrow(image, (height/2, width/2), 30)
  #cv2.imshow('image', arrow_image)
  cv2.waitKey()

def _test_on_real_image_smallgrid():  # note: untested as of 4/23
  # execute only if run as a script
  image = cv2.imread('images/chairbot_noAR_4.png')
  height, width, _ = image.shape
  
  heightR = int(math.ceil(height / 10.0)) * 10
  widthR = int(math.ceil(width / 10.0)) * 10

  botlocations = _find_chairbots(image)
  count = 0
  bots = []
  
  # this should do the following but cleaner/easier-to-read syntax
  # for bot in botlocations:
  #  bots.append(bot[0]/10)
    
  while count <= len(botlocations)-1:
    bots.append(botlocations[count][0]/10)
    count+=1
    
  # this should do the following but more robust to variable numbers of robots
  #for bot in bots:
  #  botx = int(bot[0])
  #  boty = int(bot[1])
  
  bot1X = int(bots[0][0])
  bot1Y = int(bots[0][1])
  bot2X = int(bots[1][0])
  bot2Y = int(bots[1][1])

  print("Height: ", heightR)
  print("Width: ", widthR)
  
  smallgrid = np.zeros((int(heightR/10),int(widthR/10)))
  for i in range(10):
    for j in range(10):
      smallgrid[i+bot1Y,j+bot1X] = 255
      smallgrid[i+bot2Y,j+bot2X] = 255
  cv2_imshow(smallgrid)
  print(smallgrid)


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
  image = cv2.imread('images/chairbot_noAR_4.png')
  height, width, _ = image.shape
  chairs = _find_chairbots(image)
  print('chairs found: ',chairs)


  for chair in chairs:
    chair_x, chair_y = chair[0]
    chair_angle = chair[1]
    arrow_image = drawArrow(image, (chair_x, chair_y), chair_angle)


  paths = path_planner.main()

  for path in paths:
    if len(path) == 1:
      #draw red circle here
      red_circle(image, path[0]) #coords of chair passed in 
    else:
      for i in range(len(path)-1):
        draw_real_path(image, path[i], path[i+1], (50, i*10 + 10, 108))
    

  #draws regular image without opacit/blur effect
  #cv2.imshow('image', arrow_image)
  cv2.imwrite('images/chairbot_noAR_4_adjust.png', image)

  #blend images
  blend_images()

  cv2.waitKey()


if __name__ == "__main__":
  TEST_ON_CHAIRBOTS = True
  TEST_ON_REAL_IMAGE = True
  
  if TEST_ON_CHAIRBOTS:
    _test_on_chairbots()

  if TEST_ON_REAL_IMAGE:
    image = cv2.imread('images/chairbot_noAR_4.png')
    #cv2_imshow(image)
    print(_find_chairbots(image))
    _test_on_real_image()