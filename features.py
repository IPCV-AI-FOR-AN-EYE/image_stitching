import numpy 
import cv2 as opencv


def computekeypointsandfeatures(inputim, featuretype="ORB"):
    if featuretype == "ORB":
        md = opencv.ORB_create(nfeatures = 5000)
    if featuretype == "SIFT":
        featureobject = opencv.xfeatures2d.SIFT_create()
    featureobject = opencv.xfeatures2d.SIFT_create()
    (detectedkeypoints, detectedfeatures) = featureobject.detectAndCompute(inputim, None)
    detectedkeypoints = numpy.float32([keypoint.pt for keypoint in detectedkeypoints])
    return (detectedkeypoints, detectedfeatures)


def computefeaturematches(detectedfeatures1, detectedfeatures2, featuretype = "BruteForce"):
    if featuretype == "BruteForce":
        matchobject = opencv.DescriptorMatcher_create("BruteForce")
    if featuretype == "Flann":
        FLANN_INDEX_KDTREE = 0
        p1 = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        p2 = dict(checks = 50)
        matchobject = opencv.FlannBasedMatcher(p1, p2)
    matches = matchobject.knnMatch(detectedfeatures1, detectedfeatures2, 2)
    return matches


def retainmatches(matches, lowesratio):   
    retainedmatches = []
    for match in matches:
        if len(match) == 2: 
            if match[0].distance < match[1].distance * lowesratio:
                retainedmatches.append((match[0].trainIdx, match[0].queryIdx))
    return retainedmatches


def warp2images(im1, im2, homographymatrix):
    finalshape = im1.shape[1] + im2.shape[1]
    stitchedimage = opencv.warpPerspective(im1, homographymatrix, (finalshape, im1.shape[0]))
    return stitchedimage


def computehomographymatrix(detectedkeypoints1, detectedkeypoints2,threshmax):
    (H, status) = opencv.findHomography(detectedkeypoints1, detectedkeypoints2, opencv.RANSAC, threshmax)
    return (H, status)


def computekeypointmatches(detectedkeypoints1, detectedkeypoints2, detectedfeatures1, detectedfeatures2, lowesratio, threshmax):
    matches = computefeaturematches(detectedfeatures1, detectedfeatures2);
    retainedmatches = retainmatches(matches,lowesratio)
    if len(retainedmatches) > 4:
        detectedkeypoints1 = numpy.float32([detectedkeypoints1[k] for (j, k) in retainedmatches])
        detectedkeypoints2 = numpy.float32([detectedkeypoints2[l] for (l, m) in retainedmatches])
        (homography, status) = computehomographymatrix(detectedkeypoints1, detectedkeypoints2, threshmax)
        return (retainedmatches, homography, status)
    else:
        return None


def stitch2images(images, lowesratio=0.75, matched=False, threshmax=4.0):
    (im2, im1) = images
    (detectedkeypoints1, detectedfeatures1) = computekeypointsandfeatures(im1)
    (detectedkeypoints2, detectedfeatures2) = computekeypointsandfeatures(im2)
    matches, homographymatrix, status = computekeypointmatches(detectedkeypoints1, detectedkeypoints2, detectedfeatures1, detectedfeatures2, lowesratio, threshmax)
    stitchedimage = warp2images(im1, im2, homographymatrix)
    stitchedimage[0:im2.shape[0], 0:im2.shape[1]] = im2
    if matched:
        points_image = pointsmatched(im1, im2, detectedkeypoints1, detectedkeypoints2, matches, status)
        return (stitchedimage, points_image)
    return stitchedimage


def pointsmatched( im1, im2, detectedkeypoints1, detectedkeypoints2, matches, status):
    (height1, width1) = im1.shape[:2]
    points_image = returnpoints(im1,im2)    
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            pointsim1 = (int(detectedkeypoints1[queryIdx][0]), int(detectedkeypoints1[queryIdx][1]))
            pointsim2 = (int(detectedkeypoints2[trainIdx][0]) + width1, int(detectedkeypoints2[trainIdx][1]))
            opencv.line(points_image, pointsim1, pointsim2, (0, 255, 0), 1)

    return points_image


def returnpoints(im1, im2):
    (height1, width1) = im1.shape[:2]
    (height2, width2) = im2.shape[:2]
    points_image = numpy.zeros((max(height1, height2), width1 + width2, 3), dtype="uint8")
    points_image[0:height1, 0:width1] = im1
    points_image[0:height2, width1:] = im2
    return points_image




