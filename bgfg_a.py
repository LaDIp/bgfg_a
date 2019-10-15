import cv2
import numpy as np

class bgfg_config(object):
    #speed config: resize_ratio: int, morphology_open: T/F, morphology_close: T/F, connected_component: T/F, hole_filling: T/F, shadow: T/F
    speeds = {
        'slowest':  {'resize_ratio': 1,     'morphology_open': True,    'morphology_close': True,   'connected_component': True,    'hole_filling': True,   'shadow': True,},
        'slow':     {'resize_ratio': 2,     'morphology_open': True,    'morphology_close': False,  'connected_component': True,    'hole_filling': False,  'shadow': True,},
        'normal':   {'resize_ratio': 1,     'morphology_open': True,    'morphology_close': True,   'connected_component': False,   'hole_filling': False,  'shadow': True,},
        'fast':     {'resize_ratio': 1.5,   'morphology_open': True,    'morphology_close': True,   'connected_component': False,   'hole_filling': False,  'shadow': True,},
        'faster':   {'resize_ratio': 2,     'morphology_open': True,    'morphology_close': True,   'connected_component': False,   'hole_filling': False,  'shadow': True,},
        'fastest':  {'resize_ratio': 4,     'morphology_open': True,    'morphology_close': False,  'connected_component': False,   'hole_filling': False,  'shadow': True,},
        'flash':    {'resize_ratio': 8,     'morphology_open': False,   'morphology_close': False,  'connected_component': False,   'hole_filling': False,  'shadow': False,},
        'pure':     {'resize_ratio': 1,     'morphology_open': False,   'morphology_close': False,  'connected_component': False,   'hole_filling': False,  'shadow': True,},

        # for testing
        '1': {'resize_ratio': 4,     'morphology_open': True,    'morphology_close': True,   'connected_component': False,    'hole_filling': True,   'shadow': False,},
        '2': {'resize_ratio': 4,     'morphology_open': True,    'morphology_close': False,   'connected_component': False,    'hole_filling': False,   'shadow': False,},
        '3': {'resize_ratio': 4,     'morphology_open': False,    'morphology_close': True,   'connected_component': False,    'hole_filling': True,   'shadow': False,},
        '4': {'resize_ratio': 4,     'morphology_open': False,    'morphology_close': False,   'connected_component': False,    'hole_filling': False,   'shadow': False,},
    }

class bgfg_a(object):

    def __init__(self, model='MOG2', speed='faster'):
        # set speed config
        # config: resize_ratio, morphology_open, morphology_close, connected_component, hole_filling, shadow
        self.config = bgfg_config.speeds[speed or 'faster']

        self.fgmask = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.model = model
        self.bgfg = self.init_bgmodel(model)
        self.COMPONENT_SIZE_TH = 255

    def init_bgmodel(self, bg_model):
        #pip install opencv-contrib-python if you want to use bgmodels in cv2.bgsegm and cv2.cuda
        bgfg = None

        if bg_model == 'MOG2':
            bgfg = cv2.createBackgroundSubtractorMOG2()
        elif bg_model == 'KNN':
            bgfg = cv2.createBackgroundSubtractorKNN()
        elif bg_model == 'MOG':
            bgfg = cv2.bgsegm.createBackgroundSubtractorMOG()
        elif bg_model == 'GMG':
            bgfg = cv2.bgsegm.createBackgroundSubtractorGMG()
        elif bg_model == 'CNT': #faster and better, parallelized
            #https://www.theimpossiblecode.com/blog/fastest-background-subtraction-opencv/
            bgfg = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)
        elif bg_model == 'GSOC': #better than LSBP
            bgfg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        elif bg_model == 'LSBP':
            bgfg = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif bg_model == 'cuda_MOG':
            bgfg = cv2.cuda.createBackgroundSubtractorMOG()
        elif bg_model == 'cuda_MOG2':
            bgfg = cv2.cuda.createBackgroundSubtractorMOG2()
        elif bg_model == 'cuda_FGD':
            bgfg = cv2.cuda.createBackgroundSubtractorFGD()
        elif bg_model == 'cuda_GMG':
            bgfg = cv2.cuda.createBackgroundSubtractorGMG()
        else:
            print('no such background model: %s' % bg_model)

        return bgfg

    def apply(self, img):

        # resize input image for speedup
        if self.config['resize_ratio'] > 1:
            new_size = np.divide(img.shape[:2], self.config['resize_ratio']).astype(int)
            input_img = cv2.resize(img, tuple(new_size[:2]))
        else:
            new_size = (img.shape[1], img.shape[0])
            input_img = img

        # set shasow detection on/off
        if self.model not in ['CNT', 'MOG', 'GMG', 'GSOC', 'LSBP']:
            self.bgfg.setDetectShadows(self.config['shadow'])

        # generate raw foreground mask
        fgmask = self.bgfg.apply(input_img)

        # Dilation & Erotion. open: remove noise, close: filling small holes
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        if self.config['morphology_open']:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        if self.config['morphology_close']:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel) #can be replaced by flood filling

        # Set values equal to or above 200 to 0, below 200 to 255. (when has shadows)
        if self.config['shadow']:
            _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Connected component analysis (discard small objects)
        if self.config['connected_component']:
            _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(fgmask)
            for i, _ in enumerate(labels):
                for j, _ in enumerate(labels[0]):
                    # not background and check size
                    if labels[i][j] != 0 and stats[labels[i][j]][4] < self.COMPONENT_SIZE_TH:
                        labels[i][j] = 0
            im_th = np.zeros(new_size[::-1], np.uint8)
            im_th[labels != 0] = 255
            fgmask = im_th
        # _, fgmask = cv2.threshold(np.uint8(labels), 1, 255, cv2.THRESH_BINARY) #get binary fgmask
        # fgmask = np.uint8(labels) #get grayscale fgmask

        if self.config['hole_filling']:
            fgmask = self.hole_filling(fgmask)

        if self.config['resize_ratio'] > 1:
            fgmask = cv2.resize(fgmask, (img.shape[1], img.shape[0]))

        self.fgmask = fgmask
        return fgmask

    def mask_fusion(self, img, fgmask):
        '''
        create color foreground mask
            img: w x h x 3 uint8
            mask: w x h uint8
        '''
        mask = np.zeros(img.shape, np.uint8)
        mask[:] = (255, 0, 0) #bgr
        mask = cv2.bitwise_and(mask, mask, mask=fgmask)

        return cv2.add(img, mask)

    def apply_fusion(self, img):
        fgmask = self.apply(img)
        return self.mask_fusion(img, fgmask)

    def hole_filling(self, fgmask):
        # hole filling by floodfilling. Notice the size needs to be 2 pixels larger than the image.
        mask = np.zeros(np.add(fgmask.shape, 2), np.uint8)

        # Floodfill from point (0, 0)
        im_floodfill = fgmask.copy()
        im_floodfill[0] = 0  # prevent floodfilling the foreground
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the floodfilled foreground.
        fgmask = fgmask | im_floodfill_inv

        return fgmask

    def foreground_bbox(self, img, fgmask, NMS=False):
        # get foreground bboxes
        _num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(fgmask)
        # save as another rectangle format
        # first element in stats is background (should be skipped)
        bboxes = [[x[0], x[1], int(x[0])+int(x[2]), int(x[1])+int(x[3])] for x in stats[1::]]

        # apply Non-Maximum Suppression
        if NMS and bboxes:
            bboxes = self.non_max_suppression_fast(np.array(bboxes))

        # draw bounding boxes
        for _b, bbox in enumerate(bboxes):
            cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 255, 0), 2, cv2.LINE_8)
        return img

    def non_max_suppression_fast(self, boxes, overlapThresh=0):
        '''
        boxes:
            np.array [[x1, y1, x2, y2], ...]
        overlapThresh:
            overlap threshold (pixel)
        src:
            https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        '''
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the integer data type
        return boxes[pick].astype("int")
