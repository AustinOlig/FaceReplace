import cv2
import numpy as np
import dlib
import argparse

# setting up commandline arguments to control functionality
parser = argparse.ArgumentParser(description='Face Replace!')
parser.add_argument('--debug', help='show debugging features', action="store_true")
parser.add_argument('--webcam', help='use webcam as resulting image', action="store_true")
parser.add_argument('--mixedclone', help='use mixed cloning technique', action="store_true")
parser.add_argument('--face', help='face to be replaced')
parser.add_argument('--replace', help='replacement face')
args = parser.parse_args()

while(True):
    # show debugging features
    if args.debug:
        print("Debugging mode enabled...")
    # use webcam as resulting image
    if args.webcam:
        print("Loading webcam...")
        # initialize video capture device
        cap = cv2.VideoCapture(0)
        ret, img2 = cap.read()
    else:
        print("Loading images...")
        img2 = cv2.imread(args.face)
        
    # use mixed cloning technique
    if args.mixedclone:
        print("Mixed cloning technique enabled...")

    # load images
    img1 = cv2.imread(args.replace)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    face_warped = np.zeros_like(img2)

    # set up face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat") # trained model data (via Davis King https://github.com/davisking/dlib-models)

    # detect face and facial landmarks for both images
    faces1 = face_detector(img1_gray)
    for face in faces1:
        # predict point locations and store (x, y) locations of facial landmarks
        face1_landmarks = face_predictor(img1_gray, face)
        face1_landmark_points = []
        for n in range(0, 68):
            x = face1_landmarks.part(n).x
            y = face1_landmarks.part(n).y
            face1_landmark_points.append((x, y))
            if args.debug:
                cv2.circle(img1, (x, y), 3, (0, 255,0), -1) # draw landmarks of face 1
    faces2 = face_detector(img2_gray)
    for face in faces2:
        # predict point locations and store (x, y) locations of facial landmarks
        face2_landmarks = face_predictor(img2_gray, face)
        face2_landmark_points = []
        for n in range(0, 68):
            x = face2_landmarks.part(n).x
            y = face2_landmarks.part(n).y
            face2_landmark_points.append((x, y))
            if args.debug:
                cv2.circle(img2, (x, y), 3, (0, 255,0), -1) # draw landmarks of face 2

    # get convex hull of first facial landmark points and create image mask
    face1_convexhull = cv2.convexHull(np.array(face1_landmark_points, np.int32))
    face1_mask = np.zeros_like(img1_gray)
    cv2.fillConvexPoly(face1_mask, face1_convexhull, 255)
    if args.debug:
       cv2.polylines(img1, [face1_convexhull], True, (255, 0, 0), 2) # show convex hull of face 1

    # perform Delaunay subdivision using landmark points of first face
    rect = cv2.boundingRect(face1_convexhull)
    del_subdivision = cv2.Subdiv2D(rect)
    del_subdivision.insert(face1_landmark_points)
    # perform triangulation of each subdivision
    subdivisions = []
    for triangle in del_subdivision.getTriangleList():
        # Triangulation of each subdivision by connecting points of each triangle
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        # create index information for each point in subdivision by matching landmark 
        # points locations to the subdivision point locations
        pt1_index = np.where((np.array(face1_landmark_points, np.int32) == pt1).all(axis=1))
        pt2_index = np.where((np.array(face1_landmark_points, np.int32) == pt2).all(axis=1))
        pt3_index = np.where((np.array(face1_landmark_points, np.int32) == pt3).all(axis=1))
        # store sets of indices from each point of the subdivision to reference
        # the index location for each subdivision
        subdivision_indices = [pt1_index[0][0], pt2_index[0][0], pt3_index[0][0]]
        subdivisions.append(subdivision_indices)

    # perform triangulation for both faces based on 
    # gathered subdivision index information of the first image
    for subdivision_indices in subdivisions:
        # triangulation of image 1 face
        face1_pt1 = face1_landmark_points[subdivision_indices[0]]
        face1_pt2 = face1_landmark_points[subdivision_indices[1]]
        face1_pt3 = face1_landmark_points[subdivision_indices[2]]
        # isolate working set indices of face 1
        face1_triangle = np.array([face1_pt1, face1_pt2, face1_pt3], np.int32)
        face1_rect = cv2.boundingRect(face1_triangle)
        x1, y1, w1, h1 = face1_rect
        if args.debug:
            cv2.rectangle(img1, (x1, y1), (x1+w1, y1+h1), (0, 255, 0))
        # crop triangulated area and create mask
        face1_triangle_cropped = img1[y1:y1+h1, x1:x1+w1]
        face1_triangle_mask = np.zeros((h1, w1), np.uint8)
        # triangle points with respect to bounding rectangle
        face1_masking_points = np.array([[face1_pt1[0] - x1, face1_pt1[1] - y1],
                                         [face1_pt2[0] - x1, face1_pt2[1] - y1],
                                         [face1_pt3[0] - x1, face1_pt3[1] - y1]], np.int32)
        cv2.fillConvexPoly(face1_triangle_mask, face1_masking_points, 255)
        # extract triangulated area using triangle mask
        face1_triangle_cropped = cv2.bitwise_and(face1_triangle_cropped, face1_triangle_cropped, mask=face1_triangle_mask)

        # triangulation of image 2 face
        face2_pt1 = face2_landmark_points[subdivision_indices[0]]
        face2_pt2 = face2_landmark_points[subdivision_indices[1]]
        face2_pt3 = face2_landmark_points[subdivision_indices[2]]
        # isolate working set indices of face 2
        face2_triangle = np.array([face2_pt1, face2_pt2, face2_pt3], np.int32)
        face2_rect = cv2.boundingRect(face2_triangle)
        x2, y2, w2, h2 = face2_rect
        if args.debug:
            cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0))
        # crop triangulated area and create mask
        face2_triangle_cropped = img2[y2:y2+h2, x2:x2+w2]
        face2_triangle_mask = np.zeros((h2, w2), np.uint8)
        # triangle points with respect to bounding rectangle
        face2_masking_points = np.array([[face2_pt1[0] - x2, face2_pt1[1] - y2],
                                         [face2_pt2[0] - x2, face2_pt2[1] - y2],
                                         [face2_pt3[0] - x2, face2_pt3[1] - y2]], np.int32)
        cv2.fillConvexPoly(face2_triangle_mask, face2_masking_points, 255)
        # extract triangulated area using triangle mask
        face2_triangle_cropped = cv2.bitwise_and(face2_triangle_cropped, face2_triangle_cropped, mask=face2_triangle_mask)

        if args.debug:
            # draw triangulation of face 1
            cv2.line(img1, face1_pt1, face1_pt2, (0, 0, 255))
            cv2.line(img1, face1_pt2, face1_pt3, (0, 0, 255))
            cv2.line(img1, face1_pt3, face1_pt1, (0, 0, 255))
            # draw triangulation of face 2
            cv2.line(img2, face2_pt1, face2_pt2, (0, 0, 255))
            cv2.line(img2, face2_pt2, face2_pt3, (0, 0, 255))
            cv2.line(img2, face2_pt3, face2_pt1, (0, 0, 255))

        # warp cropped triangle from face 1 to match triangle from face 2 using affine transformation
        map_matrix = cv2.getAffineTransform(np.float32(face1_masking_points), np.float32(face2_masking_points))
        face1_warped_triangle = cv2.warpAffine(face1_triangle_cropped, map_matrix, (w2, h2))
        # place warped face 1 triangle into same location space of face 2 triangle within final image
        triangle_area = face_warped[y2:y2+h2, x2:x2+w2]
        face_warped[y2:y2+h2, x2:x2+w2] = cv2.add(triangle_area, face1_warped_triangle)

    # replacing face
    face_warped_gray = cv2.cvtColor(face_warped, cv2.COLOR_BGR2GRAY)
    face2_mask = cv2.threshold(face_warped_gray, 1, 255, cv2.THRESH_BINARY)[1]
    face2_background = cv2.threshold(face_warped_gray, 1, 255, cv2.THRESH_BINARY_INV)[1]
    face2_background = cv2.bitwise_and(img2, img2, mask=face2_background)
    face2_convexhull = cv2.convexHull(np.array(face2_landmark_points, np.int32))
    cv2.fillConvexPoly(face2_mask, face2_convexhull, 255)
    x3, y3, w3, h3 = cv2.boundingRect(face2_convexhull)
    center=(int((x3 + x3 + w3) / 2), int((y3 + y3 + h3) / 2))
    face_replace = cv2.add(face2_background, face_warped)

    if args.debug:
        # show convex hull of face 2
        cv2.polylines(img2, [face2_convexhull], True, (255, 0, 0), 2) 
        # extract face from image 1 using mask
        face1 = cv2.bitwise_and(img1, img1, mask=face1_mask)
        # extract face from image 2 using mask
        face2 = cv2.bitwise_and(img2, img2, mask=face2_mask)

    if args.mixedclone:
        seamless_face_replace = cv2.seamlessClone(face_replace, img2, face2_mask, center, cv2.MIXED_CLONE)
    else:
        seamless_face_replace = cv2.seamlessClone(face_replace, img2, face2_mask, center, cv2.NORMAL_CLONE)

    # show images, image masks, faces
    if args.debug:
        cv2.imshow("Replacement Face: Face", face1)
        cv2.imshow("Replacement Face: Mask", face1_mask)
        cv2.imshow("Base Face: Face", face2)
        cv2.imshow("Base Face: Mask", face2_mask)
        cv2.imshow("Face Replace: Non-mixed", face_replace)


    cv2.imshow("Replacement Face", img1)
    cv2.imshow("Base Face", img2)
    cv2.imshow("Face Replace", seamless_face_replace)
    
    print("Hit 'q' to 'quit'")

    if args.webcam:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    else:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cv2.destroyAllWindows()